#!/usr/bin/env python3
"""Probe whether the Future20 tail cliff is caused by the CNO time boundary.

This is an inference-only diagnostic.  It loads an existing P0-A checkpoint,
checks whether its unchanged wrapper/model accepts a 40-frame temporal input,
then compares the normal Past20 -> Future20 call with a Past20 + 20 zero-frame
call, retaining only output frames 1..20 from the latter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


FRAME_FIELDS = [
    "horizon",
    "baseline_rel_l2",
    "guardband_rel_l2",
    "improvement",
    "baseline_rmse",
    "guardband_rmse",
    "baseline_pred_speed",
    "guardband_pred_speed",
    "target_speed",
]
EXPECTED_WINDOWS = 659
EXPECTED_SHAPE = (EXPECTED_WINDOWS, 20, 32, 64, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifest_paths(manifest: Path, data_root: Path, split: str) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if split not in payload or not isinstance(payload[split], list):
        raise ValueError(f"manifest lacks list split {split!r}")
    paths = [data_root / row["file"] for row in payload[split]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {split} files: {missing[:3]}")
    expected = 50 if split == "train" else 16
    if len(paths) != expected:
        raise ValueError(f"expected {expected} {split} trajectories, got {len(paths)}")
    return paths


def p0a_config_from_checkpoint(payload: dict) -> P0FeatureConfig:
    if payload.get("feature_set") != "P0-A":
        raise ValueError(f"checkpoint feature_set is {payload.get('feature_set')!r}, expected 'P0-A'")
    saved = dict(payload.get("feature_config", {}))
    saved["include_p0_a"] = True
    saved["include_p0_b"] = False
    return P0FeatureConfig(**saved)


def load_model(kit_root: Path, checkpoint: Path, config: P0FeatureConfig, device: torch.device) -> tuple[torch.nn.Module, P0FeatureBuilder]:
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d

    builder = P0FeatureBuilder(config).to(device)
    model = CNO3d(
        in_dim=len(builder.feature_names), out_dim=3, out_dim_mult=1, in_size=64, N_layers=3
    ).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    return model.eval(), builder.eval()


def forward_p0a(model: torch.nn.Module, builder: P0FeatureBuilder, x: torch.Tensor) -> torch.Tensor:
    features = builder(x)
    return model(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


def frame_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    pred_uv = prediction[..., :2]
    target_uv = target[..., :2]
    delta = pred_uv - target_uv
    numerator = np.linalg.norm(delta.reshape(delta.shape[0], delta.shape[1], -1), axis=-1)
    denominator = np.linalg.norm(target_uv.reshape(target_uv.shape[0], target_uv.shape[1], -1), axis=-1)
    rel_l2 = np.mean(numerator / np.maximum(denominator, 1e-12), axis=0)
    rmse = np.mean(np.sqrt(np.mean(delta * delta, axis=(2, 3, 4))), axis=0)
    pred_speed = np.mean(np.sqrt(np.sum(pred_uv * pred_uv, axis=-1)), axis=(0, 2, 3))
    target_speed = np.mean(np.sqrt(np.sum(target_uv * target_uv, axis=-1)), axis=(0, 2, 3))
    return {
        "rel_l2": rel_l2,
        "rmse": rmse,
        "pred_speed": pred_speed,
        "target_speed": target_speed,
    }


class OutputStats:
    """Streaming per-horizon statistics for the complete output40 tensor."""

    def __init__(self, horizons: int, channels: int = 3) -> None:
        self.count = 0
        self.sum = np.zeros((horizons, channels), dtype=np.float64)
        self.sumsq = np.zeros((horizons, channels), dtype=np.float64)
        self.speed_sum = np.zeros(horizons, dtype=np.float64)

    def update(self, output: np.ndarray) -> None:
        if output.ndim != 5 or output.shape[-1] != self.sum.shape[1]:
            raise ValueError(f"unexpected output shape for stats: {output.shape}")
        flat = output.transpose(1, 0, 2, 3, 4).reshape(output.shape[1], -1, output.shape[-1])
        self.count += flat.shape[1]
        self.sum += flat.sum(axis=1, dtype=np.float64)
        self.sumsq += np.square(flat, dtype=np.float64).sum(axis=1, dtype=np.float64)
        speed = np.sqrt(np.sum(np.square(output[..., :2]), axis=-1))
        self.speed_sum += speed.transpose(1, 0, 2, 3).reshape(output.shape[1], -1).sum(axis=1, dtype=np.float64)

    def rows(self) -> list[dict[str, float | int]]:
        mean = self.sum / self.count
        variance = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
        speed_mean = self.speed_sum / self.count
        rows = []
        for horizon in range(mean.shape[0]):
            rows.append({
                "horizon": horizon + 1,
                "mean_u": float(mean[horizon, 0]),
                "mean_v": float(mean[horizon, 1]),
                "mean_p": float(mean[horizon, 2]),
                "std_u": float(np.sqrt(variance[horizon, 0])),
                "std_v": float(np.sqrt(variance[horizon, 1])),
                "std_p": float(np.sqrt(variance[horizon, 2])),
                "speed_mean": float(speed_mean[horizon]),
            })
        return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_unsupported(out_dir: Path, evidence: dict[str, object], error: BaseException) -> None:
    evidence = dict(evidence)
    evidence.update({
        "guardband_supported": False,
        "status": "GUARDBAND_PROBE_NOT_DIRECTLY_SUPPORTED",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    })
    (out_dir / "guardband_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "frame_metrics.csv", FRAME_FIELDS, [])
    (out_dir / "README.md").write_text(
        "# Future40 Guard-Band Inference Probe\n\n"
        "结论标签：`GUARDBAND_PROBE_NOT_DIRECTLY_SUPPORTED`\n\n"
        "1. checkpoint / commit / split\n\n"
        f"checkpoint: `{evidence.get('checkpoint')}`\n\n"
        f"execution commit: `{evidence.get('execution_commit')}`\n\n"
        "split: fixed 50 train / 16 dev; no locked-final access.\n\n"
        "2. T=40 是否直接支持\n\n"
        "不支持。未修改网络参数、模型结构或 checkpoint；在 preflight 阶段停止。\n\n"
        f"error: `{evidence['error_type']}: {evidence['error_message']}`\n\n"
        "具体 shape/module/error 见 `guardband_evidence.json`。\n\n"
        "3. h1-10 / h11-18 / h19-20 结果\n\n"
        "未执行 A/B replay。\n\n"
        "4. h19/h20 improvement\n\n"
        "未执行。\n\n"
        "5. 是否支持 right-boundary hypothesis\n\n"
        "不能由本次 probe 判定；标签为 `GUARDBAND_PROBE_NOT_DIRECTLY_SUPPORTED`。\n\n"
        "6. 下一步建议\n\n"
        "不要修改 architecture 或训练；如需继续，应先由研究负责人明确批准另一种不改变模型的边界诊断。\n",
        encoding="utf-8",
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_paths = manifest_paths(args.manifest, args.data_root, "train")
    dev_paths = manifest_paths(args.manifest, args.data_root, "dev")
    dataset = H5WindowDataset(dev_paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    if len(dataset) != EXPECTED_WINDOWS:
        raise ValueError(f"expected {EXPECTED_WINDOWS} dev windows, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu")
    config = p0a_config_from_checkpoint(checkpoint_payload)
    evidence: dict[str, object] = {
        "checkpoint": args.checkpoint_label,
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_iteration": checkpoint_payload.get("iteration"),
        "manifest": args.manifest.name,
        "manifest_sha256": sha256(args.manifest),
        "kit_root": args.kit_root.name,
        "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "execution_commit": args.execution_commit or execution_commit(),
        "data_root": args.data_root.name,
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "dev_windows": len(dataset),
        "batch_size": args.batch_size,
        "device": str(device),
        "input_shape_baseline": [args.batch_size, 20, 32, 64, 3],
        "input_shape_guardband": [args.batch_size, 40, 32, 64, 3],
        "feature_config": vars(config),
        "locked_final_access": False,
    }
    if checkpoint_payload.get("iteration") != args.expected_iteration:
        raise ValueError(
            f"checkpoint iteration is {checkpoint_payload.get('iteration')!r}, "
            f"expected {args.expected_iteration}"
        )
    try:
        model, builder = load_model(args.kit_root, args.checkpoint, config, device)
        first_past, _, _, _ = dataset[0]
        first_x = first_past.unsqueeze(0).to(device)
        baseline_probe = forward_p0a(model, builder, first_x)
        guard_input = torch.cat([first_x, torch.zeros_like(first_x)], dim=1)
        guard_probe = forward_p0a(model, builder, guard_input)
        if tuple(baseline_probe.shape) != (1, 20, 32, 64, 3):
            raise RuntimeError(f"baseline preflight output shape is {tuple(baseline_probe.shape)}")
        if tuple(guard_probe.shape) != (1, 40, 32, 64, 3):
            raise RuntimeError(f"guard-band preflight output shape is {tuple(guard_probe.shape)}")
    except Exception as error:
        write_unsupported(args.out_dir, evidence, error)
        return

    evidence.update({
        "guardband_supported": True,
        "status": "COMPLETED",
        "baseline_probe_output_shape": list(baseline_probe.shape),
        "guardband_probe_output_shape": list(guard_probe.shape),
    })
    baseline_parts: list[np.ndarray] = []
    guardband_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    stats = OutputStats(horizons=40)
    for past, target, _, _ in loader:
        x = past.to(device, non_blocking=True)
        baseline = forward_p0a(model, builder, x)
        guard_input = torch.cat([x, torch.zeros_like(x)], dim=1)
        output40 = forward_p0a(model, builder, guard_input)
        if tuple(baseline.shape[1:]) != (20, 32, 64, 3):
            raise RuntimeError(f"baseline output shape is {tuple(baseline.shape)}")
        if tuple(output40.shape[1:]) != (40, 32, 64, 3):
            raise RuntimeError(f"output40 shape is {tuple(output40.shape)}")
        baseline_parts.append(baseline.cpu().numpy().astype(np.float32))
        guardband_parts.append(output40[:, :20].cpu().numpy().astype(np.float32))
        target_parts.append(target.numpy().astype(np.float32))
        stats.update(output40.cpu().numpy().astype(np.float32))

    baseline_all = np.concatenate(baseline_parts, axis=0)
    guardband_all = np.concatenate(guardband_parts, axis=0)
    target_all = np.concatenate(target_parts, axis=0)
    for name, value in {"baseline": baseline_all, "guardband": guardband_all, "target": target_all}.items():
        if value.shape != EXPECTED_SHAPE:
            raise RuntimeError(f"{name} aggregate shape is {value.shape}, expected {EXPECTED_SHAPE}")
    baseline_metrics = frame_metrics(baseline_all, target_all)
    guardband_metrics = frame_metrics(guardband_all, target_all)
    improvement = (baseline_metrics["rel_l2"] - guardband_metrics["rel_l2"]) / np.maximum(baseline_metrics["rel_l2"], 1e-12)
    rows = []
    for index in range(20):
        rows.append({
            "horizon": index + 1,
            "baseline_rel_l2": float(baseline_metrics["rel_l2"][index]),
            "guardband_rel_l2": float(guardband_metrics["rel_l2"][index]),
            "improvement": float(improvement[index]),
            "baseline_rmse": float(baseline_metrics["rmse"][index]),
            "guardband_rmse": float(guardband_metrics["rmse"][index]),
            "baseline_pred_speed": float(baseline_metrics["pred_speed"][index]),
            "guardband_pred_speed": float(guardband_metrics["pred_speed"][index]),
            "target_speed": float(baseline_metrics["target_speed"][index]),
        })
    write_csv(args.out_dir / "frame_metrics.csv", FRAME_FIELDS, rows)
    stats_rows = stats.rows()
    write_csv(args.out_dir / "output40_stats.csv", list(stats_rows[0]), stats_rows)

    def mean_slice(values: np.ndarray, start: int, stop: int) -> float:
        return float(np.mean(values[start:stop]))

    mean_abs_h1_h15 = float(np.mean(np.abs(improvement[:15])))
    h19_h20_improvement = improvement[18:20]
    strong_support = bool(np.all(h19_h20_improvement >= 0.10) and mean_abs_h1_h15 <= 0.03)
    label = "RIGHT_BOUNDARY_HYPOTHESIS_SUPPORTED" if strong_support else "RIGHT_BOUNDARY_HYPOTHESIS_NOT_SUPPORTED"
    summary = {
        "label": label,
        "baseline_rel_l2_mean_h1_h10": mean_slice(baseline_metrics["rel_l2"], 0, 10),
        "guardband_rel_l2_mean_h1_h10": mean_slice(guardband_metrics["rel_l2"], 0, 10),
        "improvement_mean_h1_h10": mean_slice(improvement, 0, 10),
        "baseline_rel_l2_mean_h11_h18": mean_slice(baseline_metrics["rel_l2"], 10, 18),
        "guardband_rel_l2_mean_h11_h18": mean_slice(guardband_metrics["rel_l2"], 10, 18),
        "improvement_mean_h11_h18": mean_slice(improvement, 10, 18),
        "baseline_rel_l2_mean_h19_h20": mean_slice(baseline_metrics["rel_l2"], 18, 20),
        "guardband_rel_l2_mean_h19_h20": mean_slice(guardband_metrics["rel_l2"], 18, 20),
        "improvement_mean_h19_h20": mean_slice(improvement, 18, 20),
        "h19_improvement": float(improvement[18]),
        "h20_improvement": float(improvement[19]),
        "h19_over_h18_baseline": float(baseline_metrics["rel_l2"][18] / baseline_metrics["rel_l2"][17]),
        "h20_over_h18_baseline": float(baseline_metrics["rel_l2"][19] / baseline_metrics["rel_l2"][17]),
        "h19_over_h18_guardband": float(guardband_metrics["rel_l2"][18] / guardband_metrics["rel_l2"][17]),
        "h20_over_h18_guardband": float(guardband_metrics["rel_l2"][19] / guardband_metrics["rel_l2"][17]),
        "mean_signed_improvement_h1_h15": float(np.mean(improvement[:15])),
        "mean_abs_improvement_h1_h15": mean_abs_h1_h15,
        "gate": {
            "late_both_at_least_10_percent": bool(np.all(h19_h20_improvement >= 0.10)),
            "early_mean_absolute_change_at_most_3_percent": bool(mean_abs_h1_h15 <= 0.03),
        },
    }
    evidence["summary"] = summary
    evidence["output40_stats_rows"] = 40
    (args.out_dir / "guardband_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    stats = {row["horizon"]: row for row in stats_rows}
    tail_reference = stats[38]
    tail = {
        "h39_vs_h38": {
            "speed_mean_ratio": float(stats[39]["speed_mean"] / max(abs(tail_reference["speed_mean"]), 1e-12)),
            "std_u_ratio": float(stats[39]["std_u"] / max(abs(tail_reference["std_u"]), 1e-12)),
            "std_v_ratio": float(stats[39]["std_v"] / max(abs(tail_reference["std_v"]), 1e-12)),
        },
        "h40_vs_h38": {
            "speed_mean_ratio": float(stats[40]["speed_mean"] / max(abs(tail_reference["speed_mean"]), 1e-12)),
            "std_u_ratio": float(stats[40]["std_u"] / max(abs(tail_reference["std_u"]), 1e-12)),
            "std_v_ratio": float(stats[40]["std_v"] / max(abs(tail_reference["std_v"]), 1e-12)),
        },
    }
    (args.out_dir / "output40_tail_comparison.json").write_text(json.dumps(tail, indent=2) + "\n", encoding="utf-8")
    pct = lambda value: f"{100.0 * value:.3f}%"
    (args.out_dir / "README.md").write_text(
        "# Future40 Guard-Band Inference Probe\n\n"
        f"结论标签：`{label}`\n\n"
        "1. checkpoint / commit / split\n\n"
        f"checkpoint: `{evidence['checkpoint']}` (iteration `{evidence['checkpoint_iteration']}`; SHA-256 `{evidence['checkpoint_sha256']}`)\n\n"
        f"execution commit: `{evidence['execution_commit']}`\n\n"
        "split: fixed 50 train / 16 dev, 659 windows; no locked-final access.\n\n"
        "2. T=40 是否直接支持\n\n"
        "支持。未修改网络参数、模型结构或 checkpoint；同一 P0-A wrapper 直接完成 `[B,40,32,64,3] -> [B,40,32,64,3]`。\n\n"
        "3. h1-10 / h11-18 / h19-20 结果\n\n"
        f"Rel-L2 mean h1-h10: A `{baseline_metrics['rel_l2'][:10].mean():.8f}`, B `{guardband_metrics['rel_l2'][:10].mean():.8f}`, improvement `{pct(summary['improvement_mean_h1_h10'])}`.\n\n"
        f"Rel-L2 mean h11-h18: A `{baseline_metrics['rel_l2'][10:18].mean():.8f}`, B `{guardband_metrics['rel_l2'][10:18].mean():.8f}`, improvement `{pct(summary['improvement_mean_h11_h18'])}`.\n\n"
        f"Rel-L2 mean h19-h20: A `{baseline_metrics['rel_l2'][18:20].mean():.8f}`, B `{guardband_metrics['rel_l2'][18:20].mean():.8f}`, improvement `{pct(summary['improvement_mean_h19_h20'])}`.\n\n"
        f"h1-h15 mean signed improvement `{pct(summary['mean_signed_improvement_h1_h15'])}`; mean absolute change `{pct(summary['mean_abs_improvement_h1_h15'])}`.\n\n"
        "4. h19/h20 improvement\n\n"
        f"h19: A `{baseline_metrics['rel_l2'][18]:.8f}` -> B `{guardband_metrics['rel_l2'][18]:.8f}`, improvement `{pct(summary['h19_improvement'])}`.\n\n"
        f"h20: A `{baseline_metrics['rel_l2'][19]:.8f}` -> B `{guardband_metrics['rel_l2'][19]:.8f}`, improvement `{pct(summary['h20_improvement'])}`.\n\n"
        "5. 是否支持 right-boundary hypothesis\n\n"
        f"`{label}`. Gate uses both h19/h20 improvements >=10% and h1-h15 mean absolute Rel-L2 change <=3%; no parameter tuning was performed.\n\n"
        "6. 下一步建议\n\n"
        f"Review the paired frame table and complete output40 statistics. Output40 tail comparison is in `output40_tail_comparison.json`; h39/h40 simple mean/std/speed statistics are in `output40_stats.csv` (no Future21-40 ground truth was used).\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", default="P0-A/N2 validation update 30900 model_last.pth")
    parser.add_argument("--expected-iteration", type=int, default=30900)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--execution-commit", default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
