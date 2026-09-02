#!/usr/bin/env python3
"""Bounded POINT-LOCAL3-BALANCED-L001 runner.

This standalone runner changes exactly one modeling quantity relative to the
frozen LOCAL3 run: the differentiable TKE coefficient is 0.001.  It starts
from a seeded random initialization, trains 1500 updates, evaluates the
registered dev gate once, and continues the same optimizer state to 7500 only
when that gate passes.  It never touches locked-final or Codabench.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realpde_loss_official_v9 as core  # noqa: E402
import realpde_point_v1_local3_runner as base  # noqa: E402

# Re-export frozen implementation names so tests and handoff tooling can
# verify the contract without importing the older runner directly.
Local3MLP = base.Local3MLP
PackedDataset = base.PackedDataset

SEED = 20260901
BATCH = 8
LAMBDA_TKE = 0.001
LR = 1e-4
SCREEN_UPDATES = 1500
FULL_UPDATES = 7500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)


def write_status(out: Path, status: str, stage: str, **extra: object) -> None:
    atomic_json(out / "status.json", {
        "status": status,
        "stage": stage,
        "pid": os.getpid(),
        "last_update_time": time.time(),
        "locked_final_accessed": False,
        "codabench": False,
        **extra,
    })


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fixed_order(n: int, count: int, seed: int) -> list[int]:
    if n <= 0:
        raise ValueError("cannot construct an order for an empty dataset")
    rng = np.random.default_rng(seed)
    base_order = np.arange(n, dtype=np.int64)
    rng.shuffle(base_order)
    return [int(base_order[i % n]) for i in range(count)]


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_paths(manifest: Path, data_root: Path, split: str) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if split not in payload:
        raise KeyError(f"manifest has no split {split!r}")
    paths = [data_root / row["file"] for row in payload[split]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {split} files under --data-root: {missing[:3]}")
    return paths


def model_loss(model: torch.nn.Module, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    pred = model(x)
    parts = core.loss_parts(pred, y)
    mse = parts["mse"]
    tke = parts["tke"]
    weighted = LAMBDA_TKE * tke
    return pred, mse, tke, mse + weighted


def flatten_grads(grads: Iterable[Tensor | None], params: Sequence[torch.nn.Parameter]) -> Tensor:
    pieces: list[Tensor] = []
    for grad, param in zip(grads, params):
        if grad is not None:
            pieces.append(grad.detach().reshape(-1))
    if not pieces:
        return params[0].new_zeros(1)
    return torch.cat(pieces)


def gradient_row(model: torch.nn.Module, x: Tensor, y: Tensor, snapshot: str, batch_index: int) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    _, mse, tke, total = model_loss(model, x, y)
    gm = torch.autograd.grad(mse, params, retain_graph=True, allow_unused=True)
    gt = torch.autograd.grad(LAMBDA_TKE * tke, params, allow_unused=True)
    gm_flat = flatten_grads(gm, params)
    gt_flat = flatten_grads(gt, params)
    nm, nt = torch.linalg.vector_norm(gm_flat), torch.linalg.vector_norm(gt_flat)
    ratio = nt / nm if float(nm) > 0.0 else torch.full_like(nt, float("nan"))
    denom = nm * nt
    cosine = torch.dot(gm_flat, gt_flat) / denom if float(denom) > 0.0 else torch.full_like(denom, float("nan"))
    return {
        "snapshot": snapshot,
        "batch": int(batch_index),
        "mse": float(mse.detach().cpu()),
        "tke_raw": float(tke.detach().cpu()),
        "weighted_tke": float((LAMBDA_TKE * tke).detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        "grad_mse_norm": float(nm.detach().cpu()),
        "grad_tke_norm": float(nt.detach().cpu()),
        "grad_ratio": float(ratio.detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
        "finite": bool(torch.isfinite(total).item() and torch.isfinite(nm).item() and torch.isfinite(nt).item()),
    }


def gradient_snapshot(model: torch.nn.Module, dataset: torch.utils.data.Dataset, order: Sequence[int], device: torch.device, label: str, batches: int = 4) -> tuple[list[dict], dict]:
    data = base.loader(dataset, list(order[:batches * BATCH]), workers=0, drop_last=True)
    rows: list[dict] = []
    model.eval()
    for batch_index, (x, y, _, _) in enumerate(data):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        rows.append(gradient_row(model, x, y, label, batch_index))
    model.train()
    finite = [r for r in rows if math.isfinite(r["grad_ratio"]) and math.isfinite(r["cosine"])]
    summary = {
        "snapshot": label,
        "batches": len(rows),
        "mean_grad_ratio": float(np.mean([r["grad_ratio"] for r in finite])) if finite else float("nan"),
        "median_grad_ratio": float(np.median([r["grad_ratio"] for r in finite])) if finite else float("nan"),
        "mean_cosine": float(np.mean([r["cosine"] for r in finite])) if finite else float("nan"),
        "median_cosine": float(np.median([r["cosine"] for r in finite])) if finite else float("nan"),
        "all_finite": len(rows) == len(finite),
    }
    return rows, summary


def train_updates(model: torch.nn.Module, optimizer: torch.optim.Optimizer, dataset: torch.utils.data.Dataset, order: Sequence[int], device: torch.device, out: Path, start: int, target: int) -> tuple[list[dict], float]:
    data = base.loader(dataset, list(order), workers=0, drop_last=True)
    iterator = iter(data)
    for _ in range(start % max(len(data), 1)):
        next(iterator)
    rows: list[dict] = []
    started = time.monotonic()
    model.train()
    for update in range(start + 1, target + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(data)
            x, y, _, _ = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, mse, tke, total = model_loss(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        finite = bool(torch.isfinite(total).item() and math.isfinite(grad_norm))
        row = {
            "update": update,
            "mse": float(mse.detach().cpu()),
            "tke_raw": float(tke.detach().cpu()),
            "weighted_tke": float((LAMBDA_TKE * tke).detach().cpu()),
            "total_loss": float(total.detach().cpu()),
            "gradient_norm": grad_norm,
            "finite": finite,
            "wall_seconds": time.monotonic() - started,
        }
        rows.append(row)
        if update % 20 == 0 or update == target:
            write_status(out, "RUNNING", "train", current_update=update, target_update=target, loss_weight=LAMBDA_TKE)
        if not finite:
            raise FloatingPointError(f"non-finite training state at update {update}: {row}")
    return rows, time.monotonic() - started


def screening_gate(persist: dict, candidate: dict) -> tuple[bool, dict]:
    base_errors, cand_errors = persist["raw_errors"], candidate["raw_errors"]
    improvement = {metric: (base_errors[metric] - cand_errors[metric]) / base_errors[metric] * 100.0 for metric in ("rel_l2", "tke", "mvpe")}
    passed = improvement["rel_l2"] > 0.0 and improvement["mvpe"] > 0.0 and improvement["tke"] >= -10.0
    return passed, {"improvement_pct": improvement, "passed": passed, "gate": "Rel-L2>0 and MVPE>0 and TKE degradation<=10%"}


def final_gate(persist: dict, candidate: dict, bootstrap: dict) -> tuple[str, dict]:
    base_errors, cand_errors = persist["raw_errors"], candidate["raw_errors"]
    improvement = {metric: (base_errors[metric] - cand_errors[metric]) / base_errors[metric] * 100.0 for metric in ("rel_l2", "tke", "mvpe")}
    stable = all(bootstrap[metric]["macro_mean_improvement_pct"] > 0.0 for metric in ("rel_l2", "tke", "mvpe"))
    if improvement["rel_l2"] >= 10.0 and improvement["mvpe"] >= 10.0 and improvement["tke"] >= -5.0 and stable:
        decision = "STRONG_GO_BALANCED_LOCAL_POINT"
    elif improvement["rel_l2"] >= 5.0 and improvement["mvpe"] >= 5.0 and improvement["tke"] >= -10.0:
        decision = "GO_BALANCED_LOCAL_POINT"
    else:
        decision = "STOP_POINT_MLP_FAMILY"
    return decision, {"improvement_pct": improvement, "bootstrap_direction_stable": stable}


def paired_bootstrap(persist_csv: Path, candidate_csv: Path, seed: int) -> tuple[list[dict], dict]:
    def read(path: Path) -> dict[str, dict[str, float]]:
        with path.open() as handle:
            return {row["trajectory_id"]: {metric: float(row[metric]) for metric in ("rel_l2", "tke", "mvpe")} for row in csv.DictReader(handle)}
    persist, candidate = read(persist_csv), read(candidate_csv)
    names = sorted(set(persist) & set(candidate))
    if len(names) != 16:
        raise ValueError(f"expected 16 paired dev trajectories, got {len(names)}")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    summary: dict = {}
    for metric in ("rel_l2", "tke", "mvpe"):
        values = np.asarray([(persist[name][metric] - candidate[name][metric]) / persist[name][metric] * 100.0 for name in names])
        samples = np.asarray([values[rng.integers(0, len(names), len(names))].mean() for _ in range(10000)])
        summary[metric] = {
            "macro_mean_improvement_pct": float(values.mean()),
            "bootstrap_95_low_pct": float(np.percentile(samples, 2.5)),
            "bootstrap_95_high_pct": float(np.percentile(samples, 97.5)),
            "trajectory_win_rate": float(np.mean(values > 0.0)),
        }
        rows.extend({"metric": metric, "trajectory_id": name, "improvement_pct": float(value)} for name, value in zip(names, values))
    return rows, summary


def horizon_rows(pred: np.ndarray, target: np.ndarray) -> list[dict]:
    rows = []
    for index in range(pred.shape[1]):
        p = pred[:, index, ..., :2].reshape(pred.shape[0], -1)
        y = target[:, index, ..., :2].reshape(target.shape[0], -1)
        rows.append({"horizon": index + 1, "velocity_rel_l2_micro": float(np.linalg.norm(p - y) / max(np.linalg.norm(y), 1e-12))})
    return rows


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, update: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": update, "lr": LR, "lambda_tke": LAMBDA_TKE, "seed": SEED}, path)
    return sha256(path)


def metrics_rows(phase: str, persist: dict, candidate: dict) -> list[dict]:
    rows = []
    for model_name, result in (("PERSIST", persist), ("POINT-LOCAL3-BALANCED-L001", candidate)):
        raw = result["raw_errors"]
        rows.append({"phase": phase, "model": model_name, "rel_l2": raw["rel_l2"], "tke": raw["tke"], "mvpe": raw["mvpe"], "mean_t_neural_s": result.get("mean_t_neural_s")})
    return rows


def write_reports(out: Path, decision: str, screen: dict, phase_metrics: list[dict], metadata: dict, bootstrap: dict | None = None) -> None:
    lines = ["# POINT-LOCAL3-BALANCED-L001", "", f"Decision: `{decision}`", "", "This is a bounded clean Point-MLP loss-balance experiment. Only the TKE coefficient changed (`0.05` → `0.001`); the new run starts from random initialization.", "", "## Frozen protocol", "", "- LOCAL3 replicate-padded 3×3 raw u/v history + normalized grid x/y; MLP `362→256→256→256→128→40`, GELU; raw residual output and p=0.", "- B3_PACKED, batch=8, seed=20260901, AdamW lr=1e-4, fixed seeded-shuffled train order; train 1500 updates and continue to 7500 only if the screen gate passes.", "- Gradient snapshots are train-only; no old LOCAL3 checkpoint is reused.", "", "## Screening gate", "", "```json", json.dumps(screen, indent=2), "```", "", "## Dev metrics", "", "| Phase | Model | Rel-L2 | TKE | MVPE | mean neural s/window |", "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['phase']} | {r['model']} | {r['rel_l2']:.6f} | {r['tke']:.6f} | {r['mvpe']:.6f} | {r['mean_t_neural_s']!s} |" for r in phase_metrics]
    if bootstrap is not None:
        lines += ["", "## Paired trajectory bootstrap", "", "```json", json.dumps(bootstrap, indent=2), "```"]
    lines += ["", "## Boundary", "", "- locked-final accessed: **NO**", "- Codabench: **NO**", "- old LOCAL3 checkpoint reused: **NO**", "- optimizer.step occurred only in the registered training updates; gradient snapshots used no optimizer.step.", "", "## Provenance", "", "```json", json.dumps(metadata, indent=2), "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme_lines = lines[:lines.index("## Provenance")] + ["", "See `report.md`, `training_curve.csv`, `gradient_snapshot.csv`, `screening_gate.json`, and `run_metadata.json` for complete evidence."]
    (out / "README_FOR_CHATGPT.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> str:
    out = args.out_dir
    # Detached shell redirection creates run.log before Python starts; that
    # single file is an allowed empty-run sentinel. Any other prior artifact
    # still fails closed to avoid overwriting an existing experiment.
    if out.exists() and any(path.name != "run.log" for path in out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    write_status(out, "RUNNING", "initializing", current_update=0, target_update=SCREEN_UPDATES, loss_weight=LAMBDA_TKE)
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_paths = resolve_paths(args.manifest, args.data_root, "train")
    dev_paths = resolve_paths(args.manifest, args.data_root, "dev")
    train_ds = base.PackedDataset(train_paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    order = fixed_order(len(train_ds), FULL_UPDATES * BATCH, args.seed)
    atomic_json(out / "fixed_order.json", {"seed": args.seed, "batch": BATCH, "windows": len(train_ds), "indices": order})
    metadata = {
        "experiment_id": "T1-ID-POINT-LOCAL3-BALANCED-L001-S20260902",
        "seed": args.seed,
        "batch": BATCH,
        "lambda_old": 0.05,
        "lambda_tke": LAMBDA_TKE,
        "lambda_selection": "0.05 / 44.764 = approximately 0.00112, frozen at 0.001",
        "lr": LR,
        "pipeline": "B3_PACKED",
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "train_windows": len(train_ds),
        "cache_bytes": int(train_ds.cache_bytes),
        "cache_build_seconds": float(train_ds.cache_build_seconds),
        "device": str(device),
        "manifest_sha256": sha256(args.manifest),
        "runner_sha256": sha256(Path(__file__)),
        "source_runner_sha256": sha256(HERE / "realpde_point_v1_local3_runner.py"),
        "locked_final_accessed": False,
        "codabench": False,
        "old_checkpoint_reused": False,
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True, capture_output=True, check=False).stdout.strip(),
    }
    atomic_json(out / "run_metadata.json", metadata)
    if args.smoke:
        model = base.Local3MLP().to(device)
        smoke_order = order[:BATCH]
        smoke_loader = base.loader(train_ds, smoke_order, workers=0, drop_last=True)
        x, y, _, _ = next(iter(smoke_loader))
        pred = model(x.to(device))
        assert tuple(pred.shape) == (BATCH, 20, 32, 64, 3), tuple(pred.shape)
        atomic_json(out / "smoke.json", {"passed": True, "prediction_shape": list(pred.shape), "cache_bytes": int(train_ds.cache_bytes), "data_equivalence": "B3_PACKED contract reused"})
        write_status(out, "DONE", "smoke_complete", smoke_pass=True)
        (out / "DONE").touch()
        return "SMOKE_PASS"

    model = base.Local3MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    init_rows, init_summary = gradient_snapshot(model, train_ds, order, device, "initialization")
    all_snap_rows = init_rows
    all_snap_summary = [init_summary]
    write_status(out, "RUNNING", "phase_a_train", current_update=0, target_update=SCREEN_UPDATES, loss_weight=LAMBDA_TKE)
    history_a, seconds_a = train_updates(model, optimizer, train_ds, order, device, out, 0, SCREEN_UPDATES)
    snap_rows_1500, snap_summary_1500 = gradient_snapshot(model, train_ds, order, device, "update_1500")
    all_snap_rows += snap_rows_1500
    all_snap_summary.append(snap_summary_1500)
    write_csv(out / "training_curve.csv", history_a)
    write_csv(out / "gradient_snapshot.csv", all_snap_rows)
    atomic_json(out / "gradient_snapshot.json", {"lambda_tke": LAMBDA_TKE, "snapshots": all_snap_summary})
    checkpoint_sha = save_checkpoint(out / "last@1500.pt", model, optimizer, SCREEN_UPDATES)
    atomic_json(out / "checkpoint_metadata.json", {"last@1500_sha256": checkpoint_sha, "random_initialization": True, "old_checkpoint_reused": False})
    write_status(out, "RUNNING", "phase_a_dev", current_update=SCREEN_UPDATES, target_update=SCREEN_UPDATES, loss_weight=LAMBDA_TKE)
    candidate_a = base.evaluate_dev(model, dev_paths, "B3_PACKED", args.kit_root, out / "dev@1500_candidate", device)
    persist_a = base.evaluate_dev(base.persistence_model().to(device), dev_paths, "B3_PACKED", args.kit_root, out / "dev@1500_persist", device)
    passed, screen = screening_gate(persist_a, candidate_a)
    atomic_json(out / "screening_gate.json", screen)
    phase_metrics = metrics_rows("1500", persist_a, candidate_a)
    write_csv(out / "metrics.csv", phase_metrics)
    if not passed:
        decision = "STOP_BALANCED_LOCAL3_EARLY"
        atomic_json(out / "summary.json", {"decision": decision, "screening_gate": screen, "train_seconds": seconds_a, "last_checkpoint_sha256": checkpoint_sha, "locked_final_accessed": False, "codabench": False})
        write_reports(out, decision, screen, phase_metrics, metadata)
        write_status(out, "DONE", "phase_a_stop", decision=decision, current_update=SCREEN_UPDATES, target_update=SCREEN_UPDATES)
        (out / decision).touch()
        return decision

    write_status(out, "RUNNING", "phase_b_train", current_update=SCREEN_UPDATES, target_update=FULL_UPDATES, loss_weight=LAMBDA_TKE)
    history_b, seconds_b = train_updates(model, optimizer, train_ds, order, device, out, SCREEN_UPDATES, FULL_UPDATES)
    write_csv(out / "training_curve.csv", history_a + history_b)
    all_snap_rows += gradient_snapshot(model, train_ds, order, device, "update_7500")[0]
    write_csv(out / "gradient_snapshot.csv", all_snap_rows)
    checkpoint_sha_full = save_checkpoint(out / "last@7500.pt", model, optimizer, FULL_UPDATES)
    write_status(out, "RUNNING", "phase_b_dev", current_update=FULL_UPDATES, target_update=FULL_UPDATES, loss_weight=LAMBDA_TKE)
    candidate_b = base.evaluate_dev(model, dev_paths, "B3_PACKED", args.kit_root, out / "dev@7500_candidate", device)
    persist_b = base.evaluate_dev(base.persistence_model().to(device), dev_paths, "B3_PACKED", args.kit_root, out / "dev@7500_persist", device)
    bootstrap_rows, bootstrap_summary = paired_bootstrap(out / "dev@7500_persist" / "trajectory_metrics.csv", out / "dev@7500_candidate" / "trajectory_metrics.csv", args.seed)
    write_csv(out / "paired_trajectory_bootstrap.csv", bootstrap_rows)
    atomic_json(out / "paired_trajectory_bootstrap.json", bootstrap_summary)
    pred_target = np.load(out / "dev@7500_candidate" / "predictions_targets.npz")
    write_csv(out / "horizon_metrics.csv", horizon_rows(pred_target["prediction"], pred_target["target"]))
    spatial = base.spatial_diagnostic(pred_target["prediction"], pred_target["target"], out)
    phase_metrics += metrics_rows("7500", persist_b, candidate_b)
    write_csv(out / "metrics.csv", phase_metrics)
    decision, final_detail = final_gate(persist_b, candidate_b, bootstrap_summary)
    atomic_json(out / "summary.json", {"decision": decision, "screening_gate": screen, "final_gate": final_detail, "phase_a_train_seconds": seconds_a, "phase_b_train_seconds": seconds_b, "last_checkpoint_sha256": checkpoint_sha_full, "spatial": spatial, "locked_final_accessed": False, "codabench": False})
    write_reports(out, decision, screen, phase_metrics, metadata, bootstrap_summary)
    write_status(out, "DONE", "complete", decision=decision, current_update=FULL_UPDATES, target_update=FULL_UPDATES)
    (out / "DONE").touch()
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_status(args.out_dir, "FAILED", "exception", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
