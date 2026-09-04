#!/usr/bin/env python3
"""Bounded Residual Corrector + Adaptive Uncertainty experiment.

The tensor helpers in this file are deliberately metadata-free: inference sees
only Past20, the backbone Future20 prediction, and normalized tensor indices.
The CLI is added below the pure helpers so the same semantics are used by unit
tests, training, and the packaged submission wrapper.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
import textwrap
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _diff(value: Tensor, dim: int) -> Tensor:
    """First derivative with unit spacing and one-sided boundaries."""
    out = torch.empty_like(value)
    first = [slice(None)] * value.ndim
    second = [slice(None)] * value.ndim
    first[dim], second[dim] = 0, 1
    out[tuple(first)] = value[tuple(second)] - value[tuple(first)]
    last = [slice(None)] * value.ndim
    before = [slice(None)] * value.ndim
    last[dim], before[dim] = -1, -2
    out[tuple(last)] = value[tuple(last)] - value[tuple(before)]
    middle = [slice(None)] * value.ndim
    right = [slice(None)] * value.ndim
    left = [slice(None)] * value.ndim
    middle[dim], right[dim], left[dim] = slice(1, -1), slice(2, None), slice(None, -2)
    if value.shape[dim] > 2:
        out[tuple(middle)] = (value[tuple(right)] - value[tuple(left)]) * 0.5
    return out


def flow_features(flow: Tensor) -> Tensor:
    """Return raw u/v plus the frozen 10 deterministic flow/index features."""
    if flow.ndim != 5 or flow.shape[-1] < 2:
        raise ValueError("flow must have shape [B,T,H,W,C>=2]")
    u, v = flow[..., 0], flow[..., 1]
    if min(u.shape[-2:]) < 2:
        raise ValueError("flow spatial dimensions must be at least 2")
    du_dt = torch.zeros_like(u)
    dv_dt = torch.zeros_like(v)
    du_dt[:, 1:] = u[:, 1:] - u[:, :-1]
    dv_dt[:, 1:] = v[:, 1:] - v[:, :-1]
    du_dy, du_dx = _diff(u, -2), _diff(u, -1)
    dv_dy, dv_dx = _diff(v, -2), _diff(v, -1)
    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy
    strain = torch.sqrt(du_dx.square() + dv_dy.square() + 0.5 * (du_dy + dv_dx).square())
    b, t, h, w = u.shape
    x = torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype).view(1, 1, 1, w).expand(b, t, h, w)
    y = torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype).view(1, 1, h, 1).expand(b, t, h, w)
    time_index = torch.linspace(-1.0, 1.0, t, device=flow.device, dtype=flow.dtype).view(1, t, 1, 1).expand(b, t, h, w)
    extras = [torch.sqrt(u.square() + v.square()), 0.5 * (u.square() + v.square()), du_dt, dv_dt,
              vorticity, divergence, strain, x, y, time_index]
    return torch.cat([flow[..., :2], *(item.unsqueeze(-1) for item in extras)], dim=-1)


def adaptive_features(past: Tensor, base_future: Tensor) -> Tensor:
    """Construct the 42-channel corrector feature tensor from causal inputs."""
    if past.ndim != 5 or base_future.ndim != 5 or past.shape[-1] < 3 or base_future.shape[-1] < 3:
        raise ValueError("past and base_future must be [B,T,H,W,C>=3]")
    if past.shape[0] != base_future.shape[0] or past.shape[2:] != base_future.shape[2:]:
        raise ValueError("past and base_future batch/spatial shapes must match")
    future_flow = flow_features(base_future[..., :2])
    last = flow_features(past[:, -1:, ..., :2]).expand(-1, base_future.shape[1], -1, -1, -1)
    if past.shape[1] < 2:
        raise ValueError("past requires at least two frames")
    velocity = past[:, -1:, ..., :2] + (past[:, -1:, ..., :2] - past[:, -2:-1, ..., :2]) * torch.arange(
        1, base_future.shape[1] + 1, device=past.device, dtype=past.dtype
    ).view(1, -1, 1, 1, 1)
    linear = flow_features(velocity)
    delta_last = base_future[..., :3] - past[:, -1:, ..., :3]
    delta_linear = base_future[..., :3] - torch.cat([velocity, torch.zeros_like(velocity[..., :1])], dim=-1)
    return torch.cat([future_flow, last, linear, delta_last, delta_linear], dim=-1)


class _ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(channels, channels, 3, padding=1), nn.GELU(), nn.Conv3d(channels, channels, 3, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class ResidualCorrector3D(nn.Module):
    def __init__(self, in_channels: int = 42, hidden: int = 64, blocks: int = 2, max_delta: float = 0.04):
        super().__init__()
        if blocks < 1 or hidden < 1 or max_delta <= 0:
            raise ValueError("invalid corrector configuration")
        self.max_delta = float(max_delta)
        self.input = nn.Conv3d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*(_ResidualBlock3D(hidden) for _ in range(blocks)))
        self.output = nn.Conv3d(hidden, 3, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 5 or features.shape[1] != self.input.in_channels:
            raise ValueError("corrector input must be [B,in_channels,T,H,W]")
        raw = self.output(self.blocks(torch.nn.functional.gelu(self.input(features))))
        delta = self.max_delta * torch.tanh(raw / self.max_delta)
        return torch.cat([delta[:, :2], torch.zeros_like(delta[:, 2:3])], dim=1)


class AdaptiveUncertaintyHead(nn.Module):
    def __init__(self, in_channels: int = 15, hidden: int = 32, blocks: int = 2):
        super().__init__()
        if blocks < 1 or hidden < 1:
            raise ValueError("invalid uncertainty-head configuration")
        self.input = nn.Conv3d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*(_ResidualBlock3D(hidden) for _ in range(blocks)))
        self.output = nn.Conv3d(hidden, 2, 1)

    def forward(self, features: Tensor) -> Tensor:
        raw = self.output(self.blocks(torch.nn.functional.gelu(self.input(features))))
        return torch.exp(raw).clamp(1e-4, 1.0)


def _rel(p: Tensor, y: Tensor) -> Tensor:
    p, y = p.reshape(p.shape[0], -1), y.reshape(y.shape[0], -1)
    return (torch.linalg.norm(p - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-8)).mean()


def _tke(x: Tensor) -> Tensor:
    uv = x[..., :2] - x[..., :2].mean(dim=1, keepdim=True)
    return 0.5 * uv.square().mean(dim=1).sum(dim=-1)


def corrector_loss(prediction: Tensor, target: Tensor, delta: Tensor) -> dict[str, Tensor]:
    uv, yuv = prediction[..., :2], target[..., :2]
    temporal = _rel(uv[:, 1:] - uv[:, :-1], yuv[:, 1:] - yuv[:, :-1])
    grad_p = uv[..., 1:, :] - uv[..., :-1, :]
    grad_y = yuv[..., 1:, :] - yuv[..., :-1, :]
    return {
        "point": _rel(uv, yuv), "mse": (uv - yuv).square().mean(), "tke": _rel(_tke(uv), _tke(yuv)),
        "temporal": temporal, "grad": _rel(grad_p, grad_y), "p_zero": prediction[..., 2].square().mean(),
        "residual_mse": (delta[..., :2] - (target[..., :2] - prediction[..., :2])).square().mean(),
        "delta_penalty": delta[..., :2].square().mean(),
    }


def gaussian_nll(target: Tensor, prediction: Tensor, sigma: Tensor) -> Tensor:
    if target.shape != prediction.shape or sigma.shape != target.shape[:-1] + (2,):
        raise ValueError("target/prediction/sigma shapes are incompatible")
    return (torch.log(sigma) + 0.5 * ((target - prediction) / sigma).square()).mean()


def evaluate_corrector_gate(*, baseline: dict[str, float], candidate: dict[str, float], trajectory_tke_degradations: Iterable[float]) -> dict[str, object]:
    rel_improvement = 1.0 - candidate["rel_l2"] / baseline["rel_l2"]
    mvpe_improvement = 1.0 - candidate["mvpe"] / baseline["mvpe"]
    tke_degradation = candidate["tke"] / baseline["tke"] - 1.0
    over_15 = sum(float(value) > 0.15 for value in trajectory_tke_degradations)
    checks = {"rel_l2_improvement_ge_2pct": rel_improvement >= 0.02, "mvpe_improvement_ge_2pct": mvpe_improvement >= 0.02,
              "aggregate_tke_degradation_le_2pct": tke_degradation <= 0.02, "trajectory_tke_over_15pct_le_2": over_15 <= 2}
    return {"status": "PASS" if all(checks.values()) else "CORRECTOR_NO_GO", "checks": checks,
            "rel_l2_improvement": rel_improvement, "mvpe_improvement": mvpe_improvement,
            "aggregate_tke_degradation": tke_degradation, "trajectory_count_over_15pct": over_15}


def fixed_calibration_grid() -> list[tuple[float, float]]:
    return [(floor, mult) for floor in (0.0, 0.0025, 0.005, 0.0075) for mult in (0.5, 1, 1.5, 2, 2.5, 3, 4)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_probe(path: Path, *, corrector: ResidualCorrector3D | None, base_head: AdaptiveUncertaintyHead | None,
                corrected_head: AdaptiveUncertaintyHead | None, metadata: dict, base_state_dict: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "corrector_state_dict": None if corrector is None else corrector.state_dict(),
        "base_head_state_dict": None if base_head is None else base_head.state_dict(),
        "corrected_head_state_dict": None if corrected_head is None else corrected_head.state_dict(),
        "base_state_dict": base_state_dict,
        "metadata": metadata,
    }, path)


def _train_module(module: nn.Module, batches, updates: int, lr: float,
                 weight_decay: float, loss_fn, log_path: Path) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
    iterator = iter(batches())
    rows: list[dict[str, float]] = []
    with log_path.open("a", encoding="utf-8") as log:
        for update in range(1, updates + 1):
            try:
                features, target = next(iterator)
            except StopIteration:
                iterator = iter(batches())
                features, target = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            value = loss_fn(module(features), target)
            value.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            row = {"update": float(update), "loss": float(value.detach().cpu()), "lr": float(optimizer.param_groups[0]["lr"])}
            rows.append(row)
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
    return rows


def run_training(*, data_root: Path, kit_root: Path, checkpoint: Path, manifest: Path, out_dir: Path,
                 updates_corrector: int = 2400, updates_head: int = 1400, batch_size: int = 8,
                 workers: int = 2, seed: int = 20260905, full: bool = False) -> dict:
    """Run one frozen validation-family or all-82 training stage.

    The caller supplies all paths.  Full mode is deliberately explicit and
    uses the fixed 3960 corrector budget; no full-data checkpoint selection is
    performed here.
    """
    import realpde_b1_p0a_n2 as base_api
    import realpde_loss_official_v9 as core
    from realpde_p0_data import H5WindowDataset, read_grid
    from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig
    if out_dir.exists():
        raise FileExistsError(out_dir)
    torch.manual_seed(seed); np.random.seed(seed)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if full:
        paths = sorted(data_root.glob("*.h5")); train_paths, dev_paths = paths, []
        updates_corrector = 3960
    else:
        train_paths = [data_root / row["file"] for row in payload["train"]]
        dev_paths = [data_root / row["file"] for row in payload["dev"]]
    if not train_paths or any(not p.is_file() for p in train_paths):
        raise FileNotFoundError("training trajectory missing")
    x_grid, y_grid = read_grid(train_paths[0], sub_sample=2)
    cfg = P0FeatureConfig(include_p0_a=True, include_p0_b=False,
                          dx=float(x_grid[0, 1] - x_grid[0, 0]), dy=float(y_grid[1, 0] - y_grid[0, 0]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    builder = P0FeatureBuilder(cfg).to(device)
    args = argparse.Namespace(batch_size=batch_size, workers=workers, max_windows=None, seed=seed)
    base = base_api.load_model(kit_root, checkpoint, builder, device).eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    ds = H5WindowDataset(train_paths, include_pressure=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    out_dir.mkdir(parents=True)
    raw_log = out_dir / "training.log"
    corrector = ResidualCorrector3D(hidden=64, blocks=2).to(device)
    head = AdaptiveUncertaintyHead(hidden=32, blocks=2).to(device)

    def batches_corrector():
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                base_pred = base_api.forward(base, builder, x)
                feats = adaptive_features(x, base_pred).permute(0, 4, 1, 2, 3)
            yield feats, (base_pred, y)

    def corr_loss(pred_delta, pair):
        base_pred, y = pair
        return sum({"point": 1.0, "mse": .05, "tke": .12, "temporal": .04, "grad": .02, "p_zero": .01}.get(k, 0.0) * v
                   for k, v in corrector_loss(base_pred + pred_delta.permute(0, 2, 3, 4, 1), y, pred_delta.permute(0, 2, 3, 4, 1)).items()) + .15 * corrector_loss(base_pred + pred_delta.permute(0, 2, 3, 4, 1), y, pred_delta.permute(0, 2, 3, 4, 1))["residual_mse"] + .05 * corrector_loss(base_pred + pred_delta.permute(0, 2, 3, 4, 1), y, pred_delta.permute(0, 2, 3, 4, 1))["delta_penalty"]

    corr_rows = _train_module(corrector, batches_corrector, updates_corrector, 1e-4, 1e-5, corr_loss, raw_log)

    def batches_head():
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                base_pred = base_api.forward(base, builder, x)
                uv_features = torch.cat([x, flow_features(base_pred[..., :2])], dim=-1).permute(0, 4, 1, 2, 3)
                delta = corrector(adaptive_features(x, base_pred).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
                final = base_pred + delta
            yield uv_features, (final[..., :2], y[..., :2])

    def nll_loss(sigma, pair):
        pred, target = pair
        return gaussian_nll(target, pred, sigma.permute(0, 2, 3, 4, 1))

    head_rows = _train_module(head, batches_head, updates_head, 1e-3, 1e-5, nll_loss, out_dir / "head_training.log")
    metadata = {"seed": seed, "full": full, "updates_corrector": updates_corrector, "updates_head": updates_head,
                "batch_size": batch_size, "device": str(device), "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint), "manifest": str(manifest), "train_windows": len(ds),
                "dx": cfg.dx, "dy": cfg.dy}
    _save_probe(out_dir / "probe.pth", corrector=corrector, base_head=head, corrected_head=None, metadata=metadata,
                base_state_dict={key: value.detach().cpu() for key, value in base.state_dict().items()})
    (out_dir / "metrics.json").write_text(json.dumps({"corrector": corr_rows, "head": head_rows, "metadata": metadata}, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run a tiny tensor smoke")
    parser.add_argument("--train", action="store_true", help="run a fixed-budget probe stage")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--kit-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--package", action="store_true", help="package a trained probe")
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--package-out", type=Path)
    parser.add_argument("--bound-abs", type=float, default=0.0025)
    parser.add_argument("--bound-rel", type=float, default=1.0)
    args = parser.parse_args()
    if args.self_test:
        torch.manual_seed(0)
        past = torch.randn(1, 20, 4, 6, 3)
        future = torch.randn(1, 20, 4, 6, 3)
        features = adaptive_features(past, future)
        corrector = ResidualCorrector3D()
        sigma = AdaptiveUncertaintyHead()(torch.randn(1, 15, 20, 4, 6))
        print(json.dumps({"features": list(features.shape), "sigma": list(sigma.shape), "grid": len(fixed_calibration_grid())}))
    elif args.train:
        required = (args.data_root, args.kit_root, args.checkpoint, args.manifest, args.out_dir)
        if any(value is None for value in required):
            parser.error("--train requires --data-root --kit-root --checkpoint --manifest --out-dir")
        result = run_training(data_root=args.data_root, kit_root=args.kit_root, checkpoint=args.checkpoint,
                              manifest=args.manifest, out_dir=args.out_dir, batch_size=args.batch_size,
                              workers=args.workers, full=args.full)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.package:
        if args.probe is None or args.package_out is None:
            parser.error("--package requires --probe and --package-out")
        args.package_out.mkdir(parents=True, exist_ok=False)
        shutil.copy2(Path(__file__), args.package_out / "adaptive_runtime.py")
        shutil.copy2(HERE / "realpde_p0_features.py", args.package_out / "realpde_p0_features.py")
        if args.kit_root is None:
            parser.error("--package requires --kit-root")
        shutil.copytree(args.kit_root / "rpde_baselines", args.package_out / "rpde_baselines")
        payload = torch.load(args.probe, map_location="cpu", weights_only=False)
        torch.save(payload, args.package_out / "model.pth")
        (args.package_out / "submission.py").write_text(textwrap.dedent(f'''\
            import numpy as np, torch
            from pathlib import Path
            from adaptive_runtime import P0FeatureBuilder, P0FeatureConfig, ResidualCorrector3D, AdaptiveUncertaintyHead, adaptive_features, flow_features
            _payload = torch.load(Path(__file__).with_name("model.pth"), map_location="cpu", weights_only=False)
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _meta = _payload["metadata"]
            _cfg = P0FeatureConfig(include_p0_a=True, include_p0_b=False, dx=_meta["dx"], dy=_meta["dy"])
            _builder = P0FeatureBuilder(_cfg).to(_device)
            _base = torch.nn.Module()\n+            from rpde_baselines.model.cno import CNO3d\n+            _base = CNO3d(in_dim=20, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(_device)\n+            _base.load_state_dict(_payload["base_state_dict"], strict=True)\n+            _corrector = ResidualCorrector3D().to(_device); _corrector.load_state_dict(_payload["corrector_state_dict"], strict=True)\n+            _head = AdaptiveUncertaintyHead().to(_device); _head.load_state_dict(_payload["base_head_state_dict"], strict=True)\n+            for _m in (_base, _corrector, _head): _m.eval()\n+            def predict(input_array, metadata=None):\n+                x = np.asarray(input_array, dtype=np.float32)\n+                if x.ndim != 5 or x.shape[1:] != (20,32,64,3): raise ValueError("expected (N,20,32,64,3)")\n+                with torch.inference_mode():\n+                    tx = torch.from_numpy(x).to(_device); bp = _base(_builder(tx).permute(0,4,1,2,3)).permute(0,2,3,4,1); bp[...,2] = 0\n+                    delta = _corrector(adaptive_features(tx,bp).permute(0,4,1,2,3)).permute(0,2,3,4,1); pred = (bp + delta).cpu().numpy().astype(np.float32)\n+                    pred[...,2] = 0; sigma = _head(torch.cat([tx, flow_features(bp[...,:2])],dim=-1).permute(0,4,1,2,3)).permute(0,2,3,4,1).cpu().numpy().astype(np.float32)\n+                half = ({args.bound_abs!r} + {args.bound_rel!r} * sigma).astype(np.float32); half[...,2] = 0\n+                return {{"prediction":pred, "lower":(pred-half).astype(np.float32), "upper":(pred+half).astype(np.float32)}}\n+        '''), encoding="utf-8")
        print(json.dumps({"status": "PACKAGED", "out": str(args.package_out)}))


if __name__ == "__main__":
    main()
