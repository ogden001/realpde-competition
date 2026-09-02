#!/usr/bin/env python3
"""Frozen Clean Point-V0 screening for RealPDE Track 1.

This is deliberately a standalone test of positional, pointwise temporal
prediction.  It neither loads a CNO checkpoint nor accesses CFD, metadata,
neighbourhood values, locked-final, or Codabench.  The only learned variants
are trained for the pre-registered last@7500 protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

import realpde_loss_official_v9 as core


VARIANTS = ("PERSIST", "POINT-DIRECT", "POINT-RESIDUAL")
LOSS_WEIGHTS = {"rel": 0.0, "mse": 1.0, "tke": 0.05, "mvpe": 0.0, "mean": 0.0, "fluct": 0.0}


def write_status(out: Path, *, status: str, stage: str, **extra: object) -> None:
    """Atomically publish runner state for detached-job recovery."""
    value = {"status": status, "stage": stage, "pid": os.getpid(), "last_update_time": time.time(), **extra}
    temp = out / ".status.tmp"
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(out / "status.json")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PointMLP(nn.Module):
    """Shared MLP with positional indices but no spatial-field input."""
    def __init__(self) -> None:
        super().__init__()
        dims = (42, 256, 256, 256, 128, 40)
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if b != dims[-1]: layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, residual: bool) -> Tensor:
        # x is B,T,H,W,3.  The input field values are only from this point.
        b, t, h, w, _ = x.shape
        if t != 20: raise ValueError(f"Point-V0 is frozen for 20 input frames, got {t}")
        xx = torch.linspace(-1.0, 1.0, w, dtype=x.dtype, device=x.device)
        yy = torch.linspace(-1.0, 1.0, h, dtype=x.dtype, device=x.device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        position = torch.stack((gx, gy), dim=-1).expand(b, -1, -1, -1)
        history = x[..., :2].permute(0, 2, 3, 1, 4).reshape(b, h, w, 40)
        point = torch.cat((position, history), dim=-1).reshape(b * h * w, 42)
        uv = self.net(point).reshape(b, h, w, 20, 2).permute(0, 3, 1, 2, 4)
        if residual: uv = uv + x[:, -1:, :, :, :2]
        # Official submission shape is 3-channel, with unavailable PIV pressure zeroed.
        return torch.cat((uv, torch.zeros_like(uv[..., :1])), dim=-1)


def persistence(x: Tensor) -> Tensor:
    uv = x[:, -1:, :, :, :2].expand(-1, 20, -1, -1, -1)
    return torch.cat((uv, torch.zeros_like(uv[..., :1])), dim=-1)


def make_loader(paths: list[Path], args: argparse.Namespace, shuffle: bool) -> tuple[core.H5WindowDataset, DataLoader]:
    ds = core.H5WindowDataset(paths, max_windows_per_trajectory=args.max_windows)
    generator = torch.Generator().manual_seed(args.seed) if shuffle else None
    return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                          pin_memory=True, generator=generator, persistent_workers=False)


def manifest_paths(manifest_path: Path, split: str, data_root: Path) -> list[Path]:
    """Resolve frozen manifest filenames through an explicit runtime data root."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split not in manifest: raise KeyError(f"split {split!r} absent from {manifest_path}")
    paths = [data_root / row["file"] for row in manifest[split]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError(f"manifest files absent under --data-root: {missing[:3]}")
    return paths


def forward_variant(model: PointMLP | None, x: Tensor, variant: str) -> Tensor:
    if variant == "PERSIST": return persistence(x)
    assert model is not None
    return model(x, residual=variant == "POINT-RESIDUAL")


def _timed_forward(model: PointMLP | None, x: Tensor, variant: str, repeats: int, device: torch.device) -> tuple[Tensor, float]:
    # Feature construction, MLP, reshape, residual restore and p=0 are all timed.
    with torch.no_grad():
        for _ in range(2): forward_variant(model, x, variant)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        pred = None
        for _ in range(repeats): pred = forward_variant(model, x, variant)
        if device.type == "cuda": torch.cuda.synchronize()
    assert pred is not None
    return pred, (time.perf_counter() - started) / repeats


def evaluate(model: PointMLP | None, variant: str, paths: list[Path], args: argparse.Namespace,
             device: torch.device, kit_root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds, data = make_loader(paths, args, shuffle=False)
    if model is not None: model.eval()
    predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []; elapsed = 0.0
    for x, y, _, _ in data:
        x = x.to(device, non_blocking=True)
        pred, seconds = _timed_forward(model, x, variant, args.timing_repeats, device)
        elapsed += seconds
        predictions.append(pred.cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    np.savez_compressed(out / "dev_predictions_targets.npz", prediction=prediction, target=target)
    result = core.score_bundle(kit_root, prediction, target, elapsed / len(ds), out)
    rows, anatomy = core.trajectory_rows(ds, prediction, target, kit_root)
    with (out / "trajectory_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result |= {"windows": len(ds), "trajectories": len(rows), "trajectory_anatomy": anatomy,
               "timing_scope": "in-memory full Point-V0 inference path; excludes DataLoader/H5 I/O"}
    core.json_dump(out / "evaluation.json", result)
    return result


def verify_tke_replay(paths: list[Path], args: argparse.Namespace, kit_root: Path, device: torch.device, out: Path) -> dict:
    """Check differentiable TKE against the supplied scorer on a fixed persistence batch."""
    _, data = make_loader(paths[:1], args, shuffle=False)
    x, y, _, _ = next(iter(data)); x = x.to(device)
    pred = persistence(x)
    torch_value = float(core.loss_parts(pred, y.to(device))["tke"].detach().cpu())
    import sys
    sys.path.insert(0, str(kit_root)); import scoring
    scorer_value = float(np.mean(scoring.tke_rel_l2_per_sample(pred.cpu().numpy(), y.numpy(), scoring.measured_channels(y.numpy()))))
    difference = abs(torch_value - scorer_value)
    result = {"torch_tke": torch_value, "scorer_tke": scorer_value, "absolute_difference": difference,
              "tolerance": 5e-6, "passed": difference <= 5e-6}
    core.json_dump(out / "tke_replay.json", result)
    if not result["passed"]: raise RuntimeError(f"TKE replay failed: {result}")
    return result


def train(model: PointMLP, variant: str, train_paths: list[Path], args: argparse.Namespace, device: torch.device, out: Path) -> dict:
    ds, loader = make_loader(train_paths, args, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    iterator = iter(loader); started = time.monotonic(); last = {}
    model.train()
    for step in range(1, args.updates + 1):
        try: x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader); x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = forward_variant(model, x, variant)
        parts = core.loss_parts(pred, y)
        loss = sum(LOSS_WEIGHTS[k] * parts[k] for k in LOSS_WEIGHTS)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        last = {k: float(v.detach().cpu()) for k, v in parts.items()} | {"total": float(loss.detach().cpu())}
        if step % 250 == 0 or step == args.updates:
            write_status(out, status="RUNNING", stage="train", current_variant=variant, current_update=step, target_update=args.updates)
    return {"actual_updates": args.updates, "train_windows": len(ds), "train_seconds": time.monotonic() - started,
            "last_loss_parts": last}


def read_rows(path: Path) -> dict[str, dict[str, float]]:
    with path.open() as f: return {r["trajectory_id"]: {k: float(r[k]) for k in ("rel_l2", "tke", "mvpe")} for r in csv.DictReader(f)}


def bootstrap(persist: dict[str, dict[str, float]], candidate: dict[str, dict[str, float]], seed: int) -> tuple[list[dict], dict]:
    names = sorted(set(persist) & set(candidate))
    if len(names) != 16: raise ValueError(f"expected 16 paired dev trajectories, got {len(names)}")
    rng = np.random.default_rng(seed); rows: list[dict] = []; summary = {}
    for metric in ("rel_l2", "tke", "mvpe"):
        improvement = np.asarray([(persist[n][metric] - candidate[n][metric]) / persist[n][metric] * 100.0 for n in names])
        means = np.asarray([improvement[rng.integers(0, len(names), len(names))].mean() for _ in range(10000)])
        summary[metric] = {"macro_mean_improvement_pct": float(improvement.mean()),
                           "bootstrap_95_low_pct": float(np.percentile(means, 2.5)),
                           "bootstrap_95_high_pct": float(np.percentile(means, 97.5)),
                           "trajectory_win_rate": float(np.mean(improvement > 0.0))}
        rows += [{"metric": metric, "trajectory_id": n, "improvement_pct": float(v)} for n, v in zip(names, improvement)]
    return rows, summary


def horizon_metrics(pred: np.ndarray, target: np.ndarray, label: str) -> list[dict]:
    rows = []
    for index in range(20):
        p, y = pred[:, index, ..., :2].reshape(pred.shape[0], -1), target[:, index, ..., :2].reshape(target.shape[0], -1)
        rows.append({"model": label, "horizon": index + 1, "velocity_rel_l2_micro": float(np.linalg.norm(p - y) / max(np.linalg.norm(y), 1e-12))})
    return rows


def spatial_map(pred: np.ndarray, target: np.ndarray, out: Path) -> dict:
    uvp, uvy = pred[..., :2], target[..., :2]
    valid = np.isfinite(uvp).all(axis=(0, 1, 4)) & np.isfinite(uvy).all(axis=(0, 1, 4)) & np.any(np.abs(uvy) > 0.0, axis=(0, 1, 4))
    err = np.sqrt(np.mean(np.sum((uvp - uvy) ** 2, axis=-1), axis=(0, 1)))
    rows = [{"row": i, "column": j, "velocity_rmse": float(err[i, j]), "valid": bool(valid[i, j])}
            for i in range(err.shape[0]) for j in range(err.shape[1])]
    with (out / "spatial_error_e2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return {"valid_pixels": int(valid.sum()), "total_pixels": int(valid.size), "mask": "finite u/v predictions and targets, with at least one nonzero target velocity across dev windows/horizons"}


def plots(out: Path, horizon: list[dict], spatial_csv: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable; CSV diagnostics were written"]
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    for label in VARIANTS:
        r = [x for x in horizon if x["model"] == label]
        ax.plot([x["horizon"] for x in r], [x["velocity_rel_l2_micro"] for x in r], label=label)
    ax.set(xlabel="forecast horizon", ylabel="velocity Rel-L2 (window micro)"); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "horizon_error.png", dpi=160); plt.close(fig)
    arr = np.full((32, 64), np.nan)
    with spatial_csv.open() as f:
        for r in csv.DictReader(f):
            if r["valid"] == "True": arr[int(r["row"])][int(r["column"])] = float(r["velocity_rmse"])
    fig, ax = plt.subplots(); im = ax.imshow(arr, origin="lower"); fig.colorbar(im, ax=ax, label="velocity RMSE")
    ax.set(title="POINT-RESIDUAL spatial error (valid pixels)"); fig.tight_layout(); fig.savefig(figdir / "spatial_error_e2.png", dpi=160); plt.close(fig)
    return []


def gate(persist: dict, residual: dict) -> tuple[str, dict]:
    base, cand = persist["raw_errors"], residual["raw_errors"]
    imp = {m: (base[m] - cand[m]) / base[m] * 100.0 for m in ("rel_l2", "tke", "mvpe")}
    if imp["rel_l2"] >= 10 and imp["mvpe"] >= 10 and imp["tke"] >= -5: decision = "STRONG_GO_POINT_V1"
    elif imp["rel_l2"] >= 5 and imp["mvpe"] >= 5 and imp["tke"] >= -10: decision = "GO_POINT_V1"
    else: decision = "STOP_PURE_POINT"
    return decision, {"raw_micro_improvement_pct": imp, "gate_model": "POINT-RESIDUAL"}


def write_review(out: Path, results: dict, stability: dict, decision: str, gate_detail: dict, spatial: dict, plot_notes: list[str]) -> None:
    metrics = []
    for v in VARIANTS:
        r = results[v]; x = r["raw_errors"]
        metrics.append({"model": v, "rel_l2": x["rel_l2"], "tke": x["tke"], "mvpe": x["mvpe"], "mean_t_neural_s": r["mean_t_neural_s"]})
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    lines = ["# Point-V0 review", "", f"## Final decision: `{decision}`", "", "This is a standalone clean screening, not a strict CNO causal comparison. No Point-V1 is launched by this run.",
             "", "## Protocol compliance", "", "- 50/16/16 manifest; train and dev only; locked final not accessed.", "- seed `20260901`; batch=8 complete windows; 7500 updates; AdamW lr=1e-5; last@7500 only.",
             "- `sim_real_ft` used: **NO**. CFD, Re/AoA, neighbourhoods and global encoders used: **NO**.", "- Point definition: normalized grid positional coordinates plus that point's 20-frame u/v history; output p=0.",
             "- Loss: raw-space `MSE + 0.05*TKE`; full field restored before TKE.", "", "## Raw v9 dev metrics", "", "| Model | Rel-L2 | TKE | MVPE | mean inference s/window |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['rel_l2']:.6f} | {r['tke']:.6f} | {r['mvpe']:.6f} | {r['mean_t_neural_s']:.6g} |" for r in metrics]
    lines += ["", "## Gate", "", "```json", json.dumps(gate_detail, indent=2), "```", "", "## Stability evidence", "", "```json", json.dumps(stability, indent=2), "```", "", "## Diagnostics", "", "- Horizon diagnostics use pre-registered window-micro velocity Rel-L2 in `horizon_metrics.csv`.", f"- Spatial map: {json.dumps(spatial)}."]
    if plot_notes: lines += ["- " + note for note in plot_notes]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "README_FOR_CHATGPT.md").write_text("\n".join(lines[:28]) + "\n")


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    if args.updates != 7500 and not args.smoke: raise ValueError("frozen full protocol requires --updates 7500")
    if args.batch_size != 8 and not args.smoke: raise ValueError("frozen full protocol requires --batch-size 8")
    if args.eval_split != "dev": raise ValueError("Point-V0 is frozen to dev only")
    if not (args.kit_root / "scoring.py").is_file(): raise FileNotFoundError(args.kit_root / "scoring.py")
    args.out_dir.mkdir(parents=True); write_status(args.out_dir, status="RUNNING", stage="initializing", current_variant=None, current_update=0, target_update=args.updates); set_seed(args.seed)
    train_paths = manifest_paths(args.manifest, "train", args.data_root)
    dev_paths = manifest_paths(args.manifest, "dev", args.data_root)
    if args.smoke:
        train_paths, dev_paths = train_paths[:1], dev_paths[:1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = {"experiment": "T1-ID-POINT-V0-7500-S20260901", "seed": args.seed, "batch_size": args.batch_size, "updates": args.updates,
            "checkpoint_rule": "last@7500", "loss_weights": LOSS_WEIGHTS, "manifest": str(args.manifest.resolve()), "data_root": str(args.data_root.resolve()),
            "manifest_sha256": core.sha256(args.manifest), "kit_scorer_sha256": core.sha256(args.kit_root / "scoring.py"),
            "script": str(Path(__file__).resolve()), "script_sha256": core.sha256(Path(__file__)),
            "loss_implementation": str(Path(core.__file__).resolve()), "loss_implementation_sha256": core.sha256(Path(core.__file__)),
            "device": str(device), "smoke": args.smoke, "locked_final_accessed": False, "sim_real_ft_used": False}
    core.json_dump(args.out_dir / "run_metadata.json", meta)
    write_status(args.out_dir, status="RUNNING", stage="v9_loss_replay")
    verify_tke_replay(train_paths, args, args.kit_root, device, args.out_dir)
    results: dict[str, dict] = {}
    for variant in VARIANTS:
        set_seed(args.seed)
        sub = args.out_dir / variant
        model = None if variant == "PERSIST" else PointMLP().to(device)
        write_status(args.out_dir, status="RUNNING", stage="persist_evaluation" if model is None else "train", current_variant=variant, current_update=0, target_update=args.updates if model else 0)
        train_info = {"actual_updates": 0, "train_windows": 0} if model is None else train(model, variant, train_paths, args, device, args.out_dir)
        if model is not None: torch.save({"model_state_dict": model.state_dict(), "variant": variant, "iteration": args.updates}, sub.parent / f"{variant}_last.pth")
        write_status(args.out_dir, status="RUNNING", stage="dev_evaluation", current_variant=variant, current_update=args.updates if model else 0, target_update=args.updates if model else 0)
        result = evaluate(model, variant, dev_paths, args, device, args.kit_root, sub)
        result["training"] = train_info; results[variant] = result
    # Smoke only validates the tensor/loss/scorer path and intentionally does not make a decision.
    if args.smoke:
        core.json_dump(args.out_dir / "summary.json", {"metadata": meta, "results": results, "status": "smoke_passed"}); write_status(args.out_dir, status="DONE", stage="smoke_complete"); (args.out_dir / "DONE").touch(); return
    write_status(args.out_dir, status="RUNNING", stage="bootstrap_and_diagnostics")
    persist_rows = read_rows(args.out_dir / "PERSIST" / "trajectory_metrics.csv")
    stability = {}; bootstrap_rows = []
    for variant in ("POINT-DIRECT", "POINT-RESIDUAL"):
        rows, summary = bootstrap(persist_rows, read_rows(args.out_dir / variant / "trajectory_metrics.csv"), args.seed)
        bootstrap_rows += [{"model": variant, **r} for r in rows]; stability[variant] = summary
    with (args.out_dir / "paired_trajectory_bootstrap.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bootstrap_rows[0])); w.writeheader(); w.writerows(bootstrap_rows)
    horizon = []
    for variant in VARIANTS:
        z = np.load(args.out_dir / variant / "dev_predictions_targets.npz")
        horizon += horizon_metrics(z["prediction"], z["target"], variant)
    with (args.out_dir / "horizon_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(horizon[0])); w.writeheader(); w.writerows(horizon)
    z = np.load(args.out_dir / "POINT-RESIDUAL" / "dev_predictions_targets.npz")
    spatial = spatial_map(z["prediction"], z["target"], args.out_dir)
    notes = plots(args.out_dir, horizon, args.out_dir / "spatial_error_e2.csv")
    decision, gate_detail = gate(results["PERSIST"], results["POINT-RESIDUAL"])
    core.json_dump(args.out_dir / "summary.json", {"metadata": meta, "results": results, "stability": stability, "gate": gate_detail, "decision": decision, "spatial": spatial})
    write_review(args.out_dir, results, stability, decision, gate_detail, spatial, notes)
    write_status(args.out_dir, status="DONE", stage="complete", current_variant="POINT-RESIDUAL", current_update=args.updates, target_update=args.updates, decision=decision)
    (args.out_dir / "DONE").touch()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--eval-split", default="dev")
    p.add_argument("--seed", type=int, default=20260901); p.add_argument("--updates", type=int, default=7500)
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--timing-repeats", type=int, default=5); p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.updates < 1 or args.batch_size < 1 or args.timing_repeats < 1: p.error("updates, batch size and timing repeats must be positive")
    try:
        run(args)
    except Exception:
        if args.out_dir.exists():
            write_status(args.out_dir, status="FAILED", stage="exception", traceback=traceback.format_exc())
        raise


if __name__ == "__main__": main()
