#!/usr/bin/env python3
"""Bounded H1: frozen CLEAN FE-00 CNO plus a LOCAL3 point residual head."""
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
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realpde_loss_official_v9 as core  # noqa: E402
import realpde_point_v1_local3_runner as base  # noqa: E402

SEED = 20260901
BATCH = 8
LR = 1e-4
WEIGHT_DECAY = 0.0
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
        "status": status, "stage": stage, "pid": os.getpid(),
        "last_update_time": time.time(), "locked_final_accessed": False,
        "codabench": False, **extra,
    })


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def fixed_order(n: int, count: int, seed: int) -> list[int]:
    if n <= 0: raise ValueError("empty dataset")
    rng = np.random.default_rng(seed); indices = np.arange(n, dtype=np.int64); rng.shuffle(indices)
    return [int(indices[i % n]) for i in range(count)]


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


class Local3PointHead(nn.Module):
    """402-input shared head: CNO future uv (40) + LOCAL3 history (360) + xy (2)."""
    def __init__(self) -> None:
        super().__init__()
        dims = (402, 256, 256, 256, 128, 40); layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if b != 40: layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)

    def forward(self, x: Tensor, cno: Tensor) -> Tensor:
        b, t, h, w, _ = x.shape
        if (t, h, w) != (20, 32, 64):
            raise ValueError(f"H1 frozen for input [B,20,32,64,3], got {tuple(x.shape)}")
        uv = x[..., :2].permute(0, 1, 4, 2, 3).reshape(b * t, 2, h, w)
        uv = F.pad(uv, (1, 1, 1, 1), mode="replicate")
        patches = F.unfold(uv, kernel_size=3, stride=1).transpose(1, 2).reshape(b, t, h * w, 18)
        patches = patches.permute(0, 2, 1, 3).reshape(b, h * w, 360)
        center = cno[..., :2].permute(0, 2, 3, 1, 4).reshape(b, h * w, 40)
        xx = torch.linspace(-1., 1., w, device=x.device, dtype=x.dtype)
        yy = torch.linspace(-1., 1., h, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        position = torch.stack((gx, gy), dim=-1).reshape(1, h * w, 2).expand(b, -1, -1)
        features = torch.cat((position, center, patches), dim=-1)
        delta = self.net(features.reshape(b * h * w, 402)).reshape(b, h * w, 20, 2)
        return delta.reshape(b, h, w, 20, 2).permute(0, 3, 1, 2, 4)


class HybridDataset(Dataset):
    def __init__(self, packed: Dataset, cno_cache: np.ndarray) -> None:
        self.packed = packed; self.cno_cache = cno_cache
        self.refs = getattr(packed, "refs", None)

    def __len__(self) -> int: return len(self.packed)

    def __getitem__(self, index: int):
        x, y, meta, _ = self.packed[index]
        return x, y, torch.from_numpy(np.asarray(self.cno_cache[index])), meta, torch.tensor(index)


def freeze_backbone(backbone: nn.Module) -> None:
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)


def make_optimizer(head: nn.Module, backbone: nn.Module) -> torch.optim.Optimizer:
    head_ids = {id(p) for p in head.parameters()}
    backbone_ids = {id(p) for p in backbone.parameters()}
    if head_ids & backbone_ids: raise RuntimeError("head/backbone parameters overlap")
    return torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def hybrid_forward(cno: Tensor, x: Tensor, head: Local3PointHead) -> Tensor:
    delta = head(x, cno)
    output = cno.clone()
    output[..., :2] = cno[..., :2] + delta
    return output


def resolve_paths(manifest: Path, data_root: Path, split: str) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if split not in payload: raise KeyError(split)
    paths = [data_root / row["file"] for row in payload[split]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing: raise FileNotFoundError(f"missing {split} files: {missing[:3]}")
    return paths


def load_frozen_cno(kit_root: Path, checkpoint: Path, device: torch.device) -> nn.Module:
    # FE-00 stores the CNO inside FEModel, so its checkpoint keys are prefixed
    # with ``cno.``.  Unwrap only that known format; never relax strict loading.
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict): raise TypeError("FE-00 checkpoint has no state dict")
    if state and all(str(key).startswith("cno.") for key in state):
        state = {str(key)[4:]: value for key, value in state.items()}
    model = CNO3d(in_dim=3, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    model.load_state_dict(state, strict=True)
    freeze_backbone(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CNO freeze check failed")
    return model


@torch.no_grad()
def build_cno_cache(model: nn.Module, packed: Dataset, order: Sequence[int], device: torch.device, cache_path: Path, out: Path) -> tuple[np.ndarray, float]:
    n = len(packed)
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(n, 20, 32, 64, 3))
    data = base.loader(packed, list(order), workers=0, drop_last=False)
    started = time.monotonic(); model.eval(); seen = 0
    for x, _, _, index in data:
        x = x.to(device, non_blocking=True)
        prediction = model(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1).cpu().numpy().astype(np.float32)
        indices = index.numpy().astype(np.int64)
        cache[indices] = prediction; seen += len(indices)
        if seen % (BATCH * 100) == 0: write_status(out, "RUNNING", "cache_build", cached_windows=seen, target_windows=n)
    cache.flush(); return cache, time.monotonic() - started


@torch.no_grad()
def evaluate_split(model: nn.Module, head: Local3PointHead | None, paths: list[Path], kit_root: Path, out: Path, device: torch.device) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    packed = base.PackedDataset(paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    data = base.loader(packed, list(range(len(packed))), workers=0, drop_last=False)
    model.eval();
    if head is not None: head.eval()
    predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []; elapsed = 0.0
    for x, y, _, _ in data:
        x = x.to(device, non_blocking=True)
        started = time.perf_counter()
        cno = model(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        prediction = cno if head is None else hybrid_forward(cno, x, head)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        predictions.append(prediction.cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    np.savez_compressed(out / "predictions_targets.npz", prediction=pred, target=target)
    result = core.score_bundle(kit_root, pred, target, elapsed / len(packed), out)
    rows, anatomy = core.trajectory_rows(packed, pred, target, kit_root)
    write_csv(out / "trajectory_metrics.csv", rows)
    result.update({"windows": len(packed), "trajectories": len(rows), "trajectory_anatomy": anatomy, "timing_scope": "CNO plus optional Point Head inference; excludes HDF5/DataLoader I/O"})
    atomic_json(out / "evaluation.json", result); return result


def train_updates(head: Local3PointHead, optimizer: torch.optim.Optimizer, dataset: HybridDataset, order: Sequence[int], device: torch.device, out: Path, start: int, target: int) -> tuple[list[dict], float]:
    data = base.loader(dataset, list(order), workers=0, drop_last=True); iterator = iter(data)
    for _ in range(start % max(len(data), 1)): next(iterator)
    history: list[dict] = []; started = time.monotonic(); head.train()
    for update in range(start + 1, target + 1):
        try: x, y, cno, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(data); x, y, cno, _, _ = next(iterator)
        x, y, cno = x.to(device, non_blocking=True), y.to(device, non_blocking=True), cno.to(device, non_blocking=True)
        prediction = hybrid_forward(cno, x, head)
        mse = (prediction[..., :2] - y[..., :2]).square().mean()
        optimizer.zero_grad(set_to_none=True); mse.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)); optimizer.step()
        finite = bool(torch.isfinite(mse).item() and math.isfinite(gradient_norm))
        if not finite: raise FloatingPointError(f"non-finite head state at update {update}")
        history.append({"update": update, "mse": float(mse.detach().cpu()), "total_loss": float(mse.detach().cpu()), "gradient_norm": gradient_norm, "finite": finite, "wall_seconds": time.monotonic() - started})
        if update % 20 == 0 or update == target: write_status(out, "RUNNING", "train", current_update=update, target_update=target, loss="uv_mse_only")
    return history, time.monotonic() - started


def screening_gate(base_result: dict, candidate: dict) -> tuple[bool, dict]:
    b, c = base_result["raw_errors"], candidate["raw_errors"]
    improvement = {metric: (b[metric] - c[metric]) / b[metric] * 100.0 for metric in ("rel_l2", "tke", "mvpe")}
    passed = improvement["rel_l2"] > 0.0 and improvement["mvpe"] > 0.0 and improvement["tke"] >= -5.0
    return passed, {"improvement_pct": improvement, "passed": passed, "gate": "Rel-L2>0 and MVPE>0 and TKE degradation<=5%"}


def paired_bootstrap(base_csv: Path, candidate_csv: Path, seed: int) -> tuple[list[dict], dict]:
    def read(path: Path) -> dict[str, dict[str, float]]:
        with path.open() as handle: return {row["trajectory_id"]: {m: float(row[m]) for m in ("rel_l2", "tke", "mvpe")} for row in csv.DictReader(handle)}
    base_rows, candidate_rows = read(base_csv), read(candidate_csv); names = sorted(set(base_rows) & set(candidate_rows))
    if len(names) != 16: raise ValueError(f"expected 16 paired dev trajectories, got {len(names)}")
    rng = np.random.default_rng(seed); rows: list[dict] = []; summary: dict = {}
    for metric in ("rel_l2", "tke", "mvpe"):
        values = np.asarray([(base_rows[name][metric] - candidate_rows[name][metric]) / base_rows[name][metric] * 100.0 for name in names])
        samples = np.asarray([values[rng.integers(0, len(names), len(names))].mean() for _ in range(10000)])
        summary[metric] = {"macro_mean_improvement_pct": float(values.mean()), "bootstrap_95_low_pct": float(np.percentile(samples, 2.5)), "bootstrap_95_high_pct": float(np.percentile(samples, 97.5)), "trajectory_win_rate": float(np.mean(values > 0.0))}
        rows.extend({"metric": metric, "trajectory_id": name, "improvement_pct": float(value)} for name, value in zip(names, values))
    return rows, summary


def final_gate(base_result: dict, candidate: dict, bootstrap: dict) -> tuple[str, dict]:
    b, c = base_result["raw_errors"], candidate["raw_errors"]
    improvement = {metric: (b[metric] - c[metric]) / b[metric] * 100.0 for metric in ("rel_l2", "tke", "mvpe")}
    stable = all(bootstrap[m]["macro_mean_improvement_pct"] > 0.0 for m in ("rel_l2", "tke", "mvpe"))
    if improvement["rel_l2"] >= 5 and improvement["mvpe"] >= 5 and improvement["tke"] >= -3 and stable: decision = "STRONG_GO_HYBRID_POINT"
    elif improvement["rel_l2"] >= 3 and improvement["mvpe"] >= 3 and improvement["tke"] >= -5: decision = "GO_HYBRID_POINT"
    else: decision = "STOP_HYBRID_POINT_H1"
    return decision, {"improvement_pct": improvement, "bootstrap_direction_supportive": stable}


def write_report(out: Path, decision: str, gate: dict, rows: list[dict], metadata: dict, bootstrap: dict | None = None) -> None:
    lines = ["# H1 Frozen CLEAN CNO + LOCAL3 Point Residual Head", "", f"Decision: `{decision}`", "", "The FE-00 CLEAN CNO backbone is frozen; only a zero-initialized LOCAL3 point residual head was optimized with uv MSE. This result is incremental value relative to this specific frozen FE-00 CNO reference, not a general CNO architecture comparison.", "", "## Screening gate", "", "```json", json.dumps(gate, indent=2), "```", "", "## Metrics", "", "| Phase | Model | Rel-L2 | TKE | MVPE | mean neural s/window |", "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['phase']} | {r['model']} | {r['rel_l2']:.8f} | {r['tke']:.8f} | {r['mvpe']:.8f} | {r['mean_t_neural_s']!s} |" for r in rows]
    if bootstrap is not None: lines += ["", "## Paired trajectory bootstrap", "", "```json", json.dumps(bootstrap, indent=2), "```"]
    lines += ["", "## Boundary", "", "- locked-final accessed: **NO**", "- Codabench: **NO**", "- sim_real_ft / OFFICIAL_WARM_START: **NO**", "- CNO joint training: **NO**", "- optimizer contains Point Head parameters only", "", "## Provenance", "", "```json", json.dumps(metadata, indent=2), "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "README_FOR_CHATGPT.md").write_text("\n".join(lines[:lines.index("## Provenance")] + ["", "See report.md and the CSV/JSON artifacts for full evidence."]) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> str:
    out = args.out_dir
    if out.exists() and any(path.name != "run.log" for path in out.iterdir()): raise FileExistsError(f"output directory not empty: {out}")
    out.mkdir(parents=True, exist_ok=True); write_status(out, "RUNNING", "initializing", current_update=0, target_update=SCREEN_UPDATES)
    set_seed(args.seed); device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file(): raise FileNotFoundError(f"CLEAN FE-00 checkpoint not found: {checkpoint}")
    train_paths, dev_paths = resolve_paths(args.manifest, args.data_root, "train"), resolve_paths(args.manifest, args.data_root, "dev")
    packed = base.PackedDataset(train_paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    order = fixed_order(len(packed), FULL_UPDATES * BATCH, args.seed); atomic_json(out / "fixed_order.json", {"seed": args.seed, "batch": BATCH, "windows": len(packed), "indices": order})
    model = load_frozen_cno(args.kit_root, checkpoint, device); head = Local3PointHead().to(device); optimizer = make_optimizer(head, model)
    point_count = sum(p.numel() for p in head.parameters()); cno_count = sum(p.numel() for p in model.parameters())
    metadata = {"experiment_id": "T1-ID-HYBRID-CNO-POINT-H1-S20260902", "registry_reference": "T1-ID-FE-N2-30M-S20260901 / FE-00 CNO-only", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "initialization_family": "CLEAN sim_pretrain -> local train PIV FE-00", "research_cleanliness": "CLEAN FE-specific reference", "frozen_cno_parameter_count": cno_count, "point_head_trainable_parameter_count": point_count, "manifest_sha256": sha256(args.manifest), "seed": args.seed, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY, "pipeline": "B3_PACKED", "device": str(device), "train_trajectories": len(train_paths), "dev_trajectories": len(dev_paths), "train_windows": len(packed), "locked_final_accessed": False, "codabench": False, "sim_real_ft": False, "cno_joint_training": False, "runner_sha256": sha256(Path(__file__)), "source_runner_sha256": sha256(HERE / "realpde_point_v1_local3_runner.py"), "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True, capture_output=True, check=False).stdout.strip()}
    atomic_json(out / "run_metadata.json", metadata)
    if args.smoke:
        smoke_packed = base.PackedDataset(train_paths[:1], in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
        x, _, _, _ = next(iter(base.loader(smoke_packed, [0], workers=0, drop_last=False))); x = x.to(device)
        with torch.no_grad(): cno = model(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1); hybrid = hybrid_forward(cno, x, head)
        zero_exact = bool(torch.equal(hybrid, cno)); opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}; head_ids = {id(p) for p in head.parameters()}; cno_grad_none = all(p.grad is None for p in model.parameters())
        if not zero_exact: raise RuntimeError("FAIL_ZERO_INIT_EQUIVALENCE")
        atomic_json(out / "smoke.json", {"passed": True, "zero_init_exact": zero_exact, "prediction_shape": list(hybrid.shape), "p_channel_exact": bool(torch.equal(hybrid[..., 2], cno[..., 2])), "cno_parameters_frozen": all(not p.requires_grad for p in model.parameters()), "optimizer_point_only": opt_ids == head_ids and not (opt_ids & {id(p) for p in model.parameters()}), "cno_grad_none": cno_grad_none, "checkpoint_sha256": metadata["checkpoint_sha256"], "locked_final_accessed": False, "codabench": False})
        write_status(out, "DONE", "smoke_complete", smoke_pass=True); (out / "DONE").touch(); return "SMOKE_PASS"

    cache_path = out / "train_cno_predictions.npy"; write_status(out, "RUNNING", "cache_build", target_windows=len(packed), loss="uv_mse_only")
    cno_cache, cache_seconds = build_cno_cache(model, packed, list(range(len(packed))), device, cache_path, out)
    metadata.update({"cno_cache_path": str(cache_path), "cno_cache_bytes": int(cno_cache.nbytes), "cno_inference_cache_seconds": cache_seconds, "backbone_metadata": "backbone_metadata.json"}); atomic_json(out / "run_metadata.json", metadata)
    backbone_meta = {"registry_reference_id": "T1-ID-FE-N2-30M-S20260901", "checkpoint_path": str(checkpoint), "checkpoint_sha256": metadata["checkpoint_sha256"], "initialization_family": metadata["initialization_family"], "research_cleanliness": metadata["research_cleanliness"], "frozen_parameter_count": cno_count, "point_head_trainable_parameter_count": point_count, "cno_parameters_requires_grad_false": all(not p.requires_grad for p in model.parameters()), "optimizer_parameter_count_equals_point_head": len({id(p) for g in optimizer.param_groups for p in g["params"]}) == len({id(p) for p in head.parameters()}), "cno_grad_check": "PASS (no_grad cache and no CNO optimizer parameters)", "sim_real_ft": False}
    atomic_json(out / "backbone_metadata.json", backbone_meta)
    if args.smoke: raise AssertionError("unreachable")
    hybrid_ds = HybridDataset(packed, cno_cache)
    write_status(out, "RUNNING", "phase_a_train", current_update=0, target_update=SCREEN_UPDATES)
    history_a, seconds_a = train_updates(head, optimizer, hybrid_ds, order, device, out, 0, SCREEN_UPDATES); write_csv(out / "training_curve.csv", history_a)
    last_path = out / "last@1500.pt"; torch.save({"head_state_dict": head.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": SCREEN_UPDATES, "lr": LR, "weight_decay": WEIGHT_DECAY}, last_path)
    candidate_a = evaluate_split(model, head, dev_paths, args.kit_root, out / "dev@1500_hybrid", device); baseline_a = evaluate_split(model, None, dev_paths, args.kit_root, out / "dev@1500_frozen_cno", device)
    passed, gate = screening_gate(baseline_a, candidate_a); atomic_json(out / "screening_gate.json", gate)
    metric_rows = [{"phase": "1500", "model": "FROZEN_CNO", "rel_l2": baseline_a["raw_errors"]["rel_l2"], "tke": baseline_a["raw_errors"]["tke"], "mvpe": baseline_a["raw_errors"]["mvpe"], "mean_t_neural_s": baseline_a["mean_t_neural_s"]}, {"phase": "1500", "model": "FROZEN_CNO+LOCAL3_POINT_HEAD", "rel_l2": candidate_a["raw_errors"]["rel_l2"], "tke": candidate_a["raw_errors"]["tke"], "mvpe": candidate_a["raw_errors"]["mvpe"], "mean_t_neural_s": candidate_a["mean_t_neural_s"]}]; write_csv(out / "metrics.csv", metric_rows)
    if not passed:
        decision = "STOP_HYBRID_POINT_H1_EARLY"; atomic_json(out / "summary.json", {"decision": decision, "screening_gate": gate, "phase_a_train_seconds": seconds_a, "last@1500_sha256": sha256(last_path), "locked_final_accessed": False, "codabench": False}); write_report(out, decision, gate, metric_rows, metadata); write_status(out, "DONE", "phase_a_stop", decision=decision, current_update=SCREEN_UPDATES, target_update=SCREEN_UPDATES); (out / decision).touch(); return decision

    write_status(out, "RUNNING", "phase_b_train", current_update=SCREEN_UPDATES, target_update=FULL_UPDATES); history_b, seconds_b = train_updates(head, optimizer, hybrid_ds, order, device, out, SCREEN_UPDATES, FULL_UPDATES); write_csv(out / "training_curve.csv", history_a + history_b)
    final_path = out / "last@7500.pt"; torch.save({"head_state_dict": head.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": FULL_UPDATES, "lr": LR, "weight_decay": WEIGHT_DECAY}, final_path)
    candidate_b = evaluate_split(model, head, dev_paths, args.kit_root, out / "dev@7500_hybrid", device); baseline_b = evaluate_split(model, None, dev_paths, args.kit_root, out / "dev@7500_frozen_cno", device)
    boot_rows, boot_summary = paired_bootstrap(out / "dev@7500_frozen_cno" / "trajectory_metrics.csv", out / "dev@7500_hybrid" / "trajectory_metrics.csv", args.seed); write_csv(out / "paired_trajectory_bootstrap.csv", boot_rows); atomic_json(out / "paired_trajectory_bootstrap.json", boot_summary)
    final_npz = np.load(out / "dev@7500_hybrid" / "predictions_targets.npz"); horizon = []
    for i in range(20):
        p = final_npz["prediction"][:, i, ..., :2].reshape(final_npz["prediction"].shape[0], -1); y = final_npz["target"][:, i, ..., :2].reshape(final_npz["target"].shape[0], -1); horizon.append({"horizon": i + 1, "velocity_rel_l2_micro": float(np.linalg.norm(p - y) / max(np.linalg.norm(y), 1e-12))})
    write_csv(out / "horizon_metrics.csv", horizon); spatial = base.spatial_diagnostic(final_npz["prediction"], final_npz["target"], out); decision, detail = final_gate(baseline_b, candidate_b, boot_summary)
    metric_rows += [{"phase": "7500", "model": "FROZEN_CNO", "rel_l2": baseline_b["raw_errors"]["rel_l2"], "tke": baseline_b["raw_errors"]["tke"], "mvpe": baseline_b["raw_errors"]["mvpe"], "mean_t_neural_s": baseline_b["mean_t_neural_s"]}, {"phase": "7500", "model": "FROZEN_CNO+LOCAL3_POINT_HEAD", "rel_l2": candidate_b["raw_errors"]["rel_l2"], "tke": candidate_b["raw_errors"]["tke"], "mvpe": candidate_b["raw_errors"]["mvpe"], "mean_t_neural_s": candidate_b["mean_t_neural_s"]}]; write_csv(out / "metrics.csv", metric_rows); atomic_json(out / "summary.json", {"decision": decision, "screening_gate": gate, "final_gate": detail, "phase_a_train_seconds": seconds_a, "phase_b_train_seconds": seconds_b, "last@7500_sha256": sha256(final_path), "spatial": spatial, "locked_final_accessed": False, "codabench": False}); write_report(out, decision, gate, metric_rows, metadata, boot_summary); write_status(out, "DONE", "complete", decision=decision, current_update=FULL_UPDATES, target_update=FULL_UPDATES); (out / "DONE").touch(); return decision


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--kit-root", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--out-dir", type=Path, required=True); parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--device", default="cuda"); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    try: run(args)
    except Exception:
        args.out_dir.mkdir(parents=True, exist_ok=True); write_status(args.out_dir, "FAILED", "exception", traceback=traceback.format_exc()); raise


if __name__ == "__main__": main()
