#!/usr/bin/env python3
"""Train-selected scalar replay probe for the frozen H1 Point correction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realpde_hybrid_cno_point_h1_runner as h1  # noqa: E402
import realpde_loss_official_v9 as core  # noqa: E402

SEED = 20260901
BATCH = 8
ALPHA_GRID = tuple(round(i / 10, 1) for i in range(11))
TKE_PROTECTION_PCT = 5.0
EXPERIMENT_ID = "T1-ID-HYBRID-CNO-POINT-H1-SCALE-S20260902"
MANIFEST_SHA = "42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347"
H1_CHECKPOINT_SHA = "fb6735bff296cc53f028b894b74691697f7475f5bbfda9ae8ee0dcd70d1e3bd2"
CNO_CHECKPOINT_SHA = "499ec748cc5db1b7f3ad24029a464e921ad31c3d383b69626ec540eba392903e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def status(out: Path, state: str, stage: str, **extra: object) -> None:
    atomic_json(out / "status.json", {
        "status": state, "stage": stage, "pid": os.getpid(),
        "last_update_time": time.time(), "locked_final_accessed": False,
        "codabench": False, **extra,
    })


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def scale_prediction(cno: np.ndarray, h1_prediction: np.ndarray, alpha: float) -> np.ndarray:
    """Apply alpha to H1 uv correction while copying CNO pressure exactly."""
    if cno.shape != h1_prediction.shape or cno.ndim != 5 or cno.shape[-1] != 3:
        raise ValueError(f"expected matching [N,20,H,W,3], got {cno.shape} and {h1_prediction.shape}")
    result = cno.copy()
    result[..., :2] = cno[..., :2] + np.float32(alpha) * (h1_prediction[..., :2] - cno[..., :2])
    return result


def raw_improvements(base_errors: dict[str, float], candidate_errors: dict[str, float]) -> dict[str, float]:
    return {m: (base_errors[m] - candidate_errors[m]) / base_errors[m] * 100.0 for m in ("rel_l2", "tke", "mvpe")}


def select_alpha(rows: list[dict]) -> float | None:
    safe = [r for r in rows if float(r["tke_impr_pct"]) >= -TKE_PROTECTION_PCT]
    if not safe:
        return None
    # Maximum Rel-L2 improvement; within 0.1 percentage point, maximize MVPE;
    # final tie-break is the smaller alpha.
    best_rel = max(float(r["rel_impr_pct"]) for r in safe)
    rel_tied = [r for r in safe if best_rel - float(r["rel_impr_pct"]) < 0.1]
    best_mvpe = max(float(r["mvpe_impr_pct"]) for r in rel_tied)
    mvpe_tied = [r for r in rel_tied if float(r["mvpe_impr_pct"]) == best_mvpe]
    return float(min(mvpe_tied, key=lambda r: float(r["alpha"]))["alpha"])


def dev_gate(base_errors: dict[str, float], candidate_errors: dict[str, float]) -> tuple[bool, dict]:
    improvement = raw_improvements(base_errors, candidate_errors)
    detail = {
        "gate": "Rel-L2>0 and MVPE>0 and TKE degradation<=5%",
        "improvement_pct": improvement,
        "passed": bool(improvement["rel_l2"] > 0 and improvement["mvpe"] > 0 and improvement["tke"] >= -5.0),
    }
    return bool(detail["passed"]), detail


def load_head(checkpoint: Path, device: torch.device) -> h1.Local3PointHead:
    if sha256(checkpoint) != H1_CHECKPOINT_SHA:
        raise ValueError(f"FAIL_H1_CHECKPOINT_SHA: {sha256(checkpoint)}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("head_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise TypeError("H1 checkpoint has no head_state_dict")
    head = h1.Local3PointHead().to(device)
    head.load_state_dict(state, strict=True); head.eval()
    return head


@torch.no_grad()
def replay_split(cno: torch.nn.Module, head: h1.Local3PointHead, paths: list[Path], device: torch.device, out: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    packed = h1.base.PackedDataset(paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    data = h1.base.loader(packed, list(range(len(packed))), workers=0, drop_last=False)
    cno_rows: list[np.ndarray] = []; h1_rows: list[np.ndarray] = []; targets: list[np.ndarray] = []; elapsed = 0.0
    cno.eval(); head.eval()
    for x, y, _, _ in data:
        x = x.to(device, non_blocking=True)
        started = time.perf_counter()
        cno_pred = cno(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        h1_pred = h1.hybrid_forward(cno_pred, x, head)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        cno_rows.append(cno_pred.cpu().numpy().astype(np.float32))
        h1_rows.append(h1_pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    cno_arr, h1_arr, target_arr = np.concatenate(cno_rows), np.concatenate(h1_rows), np.concatenate(targets)
    np.savez_compressed(out / f"{split}_replay.npz", cno=cno_arr, h1=h1_arr, target=target_arr)
    return cno_arr, h1_arr, target_arr, elapsed / max(len(packed), 1)


def score(kit_root: Path, prediction: np.ndarray, target: np.ndarray, mean_t: float, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    result = core.score_bundle(kit_root, prediction, target, mean_t, out)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def scan_train(cno: np.ndarray, h1_prediction: np.ndarray, target: np.ndarray, mean_t: float, kit_root: Path, out: Path) -> tuple[list[dict], dict]:
    baseline = score(kit_root, cno, target, mean_t, out / "train_score_cno")
    base_errors = baseline["raw_errors"]
    rows: list[dict] = []
    for alpha in ALPHA_GRID:
        pred = scale_prediction(cno, h1_prediction, alpha)
        result = score(kit_root, pred, target, mean_t, out / f"train_score_alpha_{alpha:.1f}")
        errors = result["raw_errors"]; impr = raw_improvements(base_errors, errors)
        rows.append({"alpha": alpha, "rel_l2": errors["rel_l2"], "rel_impr_pct": impr["rel_l2"], "tke": errors["tke"], "tke_impr_pct": impr["tke"], "mvpe": errors["mvpe"], "mvpe_impr_pct": impr["mvpe"], "tke_safe": impr["tke"] >= -5.0})
    alpha_star = select_alpha(rows)
    selection = {"selection_split": "train", "alpha_grid": list(ALPHA_GRID), "tke_protection_pct": 5.0, "selection_primary": "max_rel_l2_improvement", "alpha_star": alpha_star, "dev_used_for_selection": False, "train_baseline_errors": base_errors}
    atomic_json(out / "alpha_selection.json", selection)
    write_csv(out / "alpha_train_scan.csv", rows)
    return rows, selection


def dev_eval(alpha_star: float, cno: np.ndarray, h1_prediction: np.ndarray, target: np.ndarray, mean_t: float, kit_root: Path, out: Path) -> tuple[list[dict], dict]:
    records = [("FROZEN_CNO", 0.0, cno), ("H1_ORIGINAL_ALPHA_1", 1.0, h1_prediction), ("H1_SCALED_ALPHA_STAR", alpha_star, scale_prediction(cno, h1_prediction, alpha_star))]
    baseline = score(kit_root, cno, target, mean_t, out / "dev_score_cno"); base_errors = baseline["raw_errors"]; rows = []
    for name, alpha, pred in records:
        result = baseline if name == "FROZEN_CNO" else score(kit_root, pred, target, mean_t, out / f"dev_score_{name.lower()}")
        errors = result["raw_errors"]; impr = raw_improvements(base_errors, errors)
        rows.append({"model": name, "alpha": alpha, "rel_l2": errors["rel_l2"], "rel_impr_pct": impr["rel_l2"], "tke": errors["tke"], "tke_impr_pct": impr["tke"], "mvpe": errors["mvpe"], "mvpe_impr_pct": impr["mvpe"], "official_v9_subscores": json.dumps(result["official_v9_subscores"], sort_keys=True)})
    scaled = next(row for row in rows if row["model"] == "H1_SCALED_ALPHA_STAR")
    passed, gate = dev_gate(base_errors, {"rel_l2": scaled["rel_l2"], "tke": scaled["tke"], "mvpe": scaled["mvpe"]})
    gate["decision"] = "GO_H1_SCALE_CALIBRATION" if passed else "STOP_H1_SCALE_CALIBRATION"
    atomic_json(out / "dev_gate.json", gate); write_csv(out / "dev_metrics.csv", rows)
    return rows, gate


def run(args: argparse.Namespace) -> str:
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"output directory not empty: {out}")
    out.mkdir(parents=True); status(out, "RUNNING", "initializing")
    set_seed(args.seed); device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    manifest_sha = sha256(args.manifest.resolve())
    if manifest_sha != MANIFEST_SHA: raise ValueError(f"manifest SHA mismatch: {manifest_sha}")
    train_paths = h1.resolve_paths(args.manifest, args.data_root, "train"); dev_paths = h1.resolve_paths(args.manifest, args.data_root, "dev")
    if args.smoke:
        train_paths, dev_paths = train_paths[:1], dev_paths[:1]
    cno = h1.load_frozen_cno(args.kit_root, args.cno_checkpoint, device); head = load_head(args.h1_checkpoint, device)
    if any(p.requires_grad for p in cno.parameters()): raise RuntimeError("CNO is not frozen")
    metadata = {"experiment_id": EXPERIMENT_ID, "source_h1_experiment": "T1-ID-HYBRID-CNO-POINT-H1-S20260902", "cno_checkpoint_sha256": sha256(args.cno_checkpoint), "h1_checkpoint_sha256": sha256(args.h1_checkpoint), "manifest_sha256": manifest_sha, "runner_sha256": sha256(Path(__file__)), "h1_runner_sha256": sha256(HERE / "realpde_hybrid_cno_point_h1_runner.py"), "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True, capture_output=True, check=False).stdout.strip(), "seed": args.seed, "batch": BATCH, "pipeline": "B3_PACKED", "device": str(device), "train_trajectories": len(train_paths), "dev_trajectories": len(dev_paths), "alpha_grid": list(ALPHA_GRID), "tke_protection_pct": 5.0, "optimizer_step": False, "model_training": False, "locked_final_accessed": False, "codabench": False}
    atomic_json(out / "run_metadata.json", metadata)
    status(out, "RUNNING", "train_replay")
    train_cno, train_h1, train_target, train_mean_t = replay_split(cno, head, train_paths, device, out, "train")
    if not (np.allclose(scale_prediction(train_cno, train_h1, 0.0), train_cno, atol=1e-6, rtol=0) and np.allclose(scale_prediction(train_cno, train_h1, 1.0), train_h1, atol=1e-6, rtol=0)):
        raise RuntimeError("FAIL_H1_REPLAY_EQUIVALENCE")
    atomic_json(out / "replay_sanity.json", {"alpha_0_max_abs_diff": float(np.max(np.abs(scale_prediction(train_cno, train_h1, 0.0) - train_cno))), "alpha_1_max_abs_diff": float(np.max(np.abs(scale_prediction(train_cno, train_h1, 1.0) - train_h1))), "alpha_0_pressure_exact": bool(np.array_equal(scale_prediction(train_cno, train_h1, 0.0)[..., 2], train_cno[..., 2])), "alpha_1_pressure_exact": bool(np.array_equal(scale_prediction(train_cno, train_h1, 1.0)[..., 2], train_cno[..., 2]))})
    if args.smoke:
        status(out, "DONE", "smoke_complete", smoke_pass=True, dev_accessed=False)
        (out / "SMOKE_PASS").touch()
        return "SMOKE_PASS"
    status(out, "RUNNING", "train_alpha_scan")
    _, selection = scan_train(train_cno, train_h1, train_target, train_mean_t, args.kit_root, out)
    alpha_star = selection["alpha_star"]
    if alpha_star is None or alpha_star <= 0:
        decision = "NO_NONZERO_SAFE_ALPHA"; atomic_json(out / "summary.json", {"decision": decision, "alpha_star": alpha_star, "dev_accessed": False, "locked_final_accessed": False, "codabench": False}); write_report(out, decision, metadata, selection, None, None); status(out, "DONE", "no_nonzero_safe_alpha", decision=decision); (out / decision).touch(); return decision
    status(out, "RUNNING", "dev_eval", alpha_star=alpha_star)
    dev_cno, dev_h1, dev_target, dev_mean_t = replay_split(cno, head, dev_paths, device, out, "dev")
    dev_rows, gate = dev_eval(float(alpha_star), dev_cno, dev_h1, dev_target, dev_mean_t, args.kit_root, out)
    decision = gate["decision"]; atomic_json(out / "summary.json", {"decision": decision, "alpha_star": alpha_star, "dev_accessed": True, "dev_gate": gate, "locked_final_accessed": False, "codabench": False}); write_report(out, decision, metadata, selection, dev_rows, gate); status(out, "DONE", "complete", decision=decision, alpha_star=alpha_star); (out / decision).touch(); return decision


def write_report(out: Path, decision: str, metadata: dict, selection: dict, dev_rows: list[dict] | None, gate: dict | None) -> None:
    lines = ["# H1 Residual Scale Probe", "", f"Decision: `{decision}`", "", "No neural-network training was performed. The existing H1 Point correction was replayed as `Y_alpha = Y_CNO + alpha * (Y_H1 - Y_CNO)` with pressure copied from CNO.", "", "## Train-only selection", "", "```json", json.dumps(selection, indent=2), "```"]
    if gate is not None: lines += ["", "## Dev gate", "", "```json", json.dumps(gate, indent=2), "```"]
    if dev_rows is not None:
        lines += ["", "| Model | alpha | Rel-L2 | TKE error | MVPE | Rel impr % | TKE impr % | MVPE impr % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| {r['model']} | {float(r['alpha']):.1f} | {r['rel_l2']:.8f} | {r['tke']:.8f} | {r['mvpe']:.8f} | {r['rel_impr_pct']:.3f} | {r['tke_impr_pct']:.3f} | {r['mvpe_impr_pct']:.3f} |" for r in dev_rows]
    lines += ["", "## Boundary", "", "- model training: **NO**", "- optimizer.step: **NO**", "- dev used for alpha selection: **NO**", f"- dev accessed for frozen evaluation: **{'YES' if dev_rows is not None else 'NO'}**", "- locked-final accessed: **NO**", "- Codabench: **NO**", "- H2 / joint training: **NOT EXECUTED**", "", "## Metadata", "", "```json", json.dumps(metadata, indent=2), "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme = lines[:lines.index("## Metadata")] + ["", "See report.md and the CSV/JSON artifacts for complete evidence."]
    (out / "README_FOR_CHATGPT.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--kit-root", type=Path, required=True); parser.add_argument("--cno-checkpoint", type=Path, required=True); parser.add_argument("--h1-checkpoint", type=Path, required=True); parser.add_argument("--out-dir", type=Path, required=True); parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--device", default="cuda"); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try: run(args)
    except Exception:
        args.out_dir.mkdir(parents=True, exist_ok=True); status(args.out_dir, "FAILED", "exception", traceback=traceback.format_exc()); raise


if __name__ == "__main__": main()
