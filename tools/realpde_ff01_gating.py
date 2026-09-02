#!/usr/bin/env python3
"""CLEAN FF-01 latent FiLM/gating screening for Track 1.

The raw CNO path remains unchanged until its bottleneck.  A fixed-width,
zero-initialized 2-D condition encoder produces a per-pixel scale/bias pair;
the pair is resized to the bottleneck spatial grid and broadcast over the
20-frame latent time axis.  G0, G1 and G2 therefore have identical graph and
conditioner parameter counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


FEATURE_GROUPS = {
    "G0": ("u_last", "v_last", "zero_0", "zero_1", "zero_2", "zero_3", "zero_4", "zero_5", "zero_6", "zero_7"),
    "G1": ("mean_u_20", "mean_v_20", "std_u_20", "std_v_20", "delta_u", "delta_v", "zero_0", "zero_1", "zero_2", "zero_3"),
    "G2": ("mean_u_20", "mean_v_20", "std_u_20", "std_v_20", "delta_u", "delta_v", "du_dx_pixel", "du_dy_pixel", "dv_dx_pixel", "dv_dy_pixel"),
}
FEATURE_INPUT_WIDTH = 10


def pixel_derivative(field: Tensor, axis: int) -> Tensor:
    """Spacing-one finite difference for [B,H,W] fields."""
    if field.ndim != 3 or axis not in (1, 2):
        raise ValueError("pixel_derivative expects [B,H,W] and axis 1 or 2")
    out = torch.empty_like(field)
    if axis == 2:
        out[:, :, 0] = field[:, :, 1] - field[:, :, 0]
        out[:, :, -1] = field[:, :, -1] - field[:, :, -2]
        out[:, :, 1:-1] = 0.5 * (field[:, :, 2:] - field[:, :, :-2])
    else:
        out[:, 0, :] = field[:, 1, :] - field[:, 0, :]
        out[:, -1, :] = field[:, -1, :] - field[:, -2, :]
        out[:, 1:-1, :] = 0.5 * (field[:, 2:, :] - field[:, :-2, :])
    return out


def build_prior_features(x: Tensor, group: str) -> Tensor:
    """Build one of the frozen ten-channel [B,C,H,W] prior packages.

    ``x`` is the runtime tensor [B,20,H,W,3].  Only its current u/v input is
    read; the pressure placeholder and all target frames are ignored.
    """
    if group not in FEATURE_GROUPS:
        raise ValueError(f"unknown FF-01 group: {group}")
    if x.ndim != 5 or x.shape[1] != 20 or x.shape[-1] < 2:
        raise ValueError("FF-01 expects [B,20,H,W,3] runtime input")
    u, v = x[..., 0], x[..., 1]
    last_u, last_v = u[:, -1], v[:, -1]
    values = {
        "u_last": last_u,
        "v_last": last_v,
        "mean_u_20": u.mean(dim=1),
        "mean_v_20": v.mean(dim=1),
        "std_u_20": u.std(dim=1, unbiased=False),
        "std_v_20": v.std(dim=1, unbiased=False),
        "delta_u": last_u - u[:, -2],
        "delta_v": last_v - v[:, -2],
        "du_dx_pixel": pixel_derivative(last_u, 2),
        "du_dy_pixel": pixel_derivative(last_u, 1),
        "dv_dx_pixel": pixel_derivative(last_v, 2),
        "dv_dy_pixel": pixel_derivative(last_v, 1),
    }
    zeros = torch.zeros_like(last_u)
    channels = [values.get(name, zeros) for name in FEATURE_GROUPS[group]]
    return torch.stack(channels, dim=1)


class ConditionEncoder(nn.Module):
    """Small fixed-width 2-D encoder whose final projection starts at zero."""

    def __init__(self, latent_channels: int, in_channels: int = FEATURE_INPUT_WIDTH, hidden_channels: int = 16):
        super().__init__()
        self.input = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.projection = nn.Conv2d(hidden_channels, 2 * latent_channels, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: Tensor, spatial_size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        features = F.interpolate(features, size=spatial_size, mode="bilinear", align_corners=False)
        params = self.projection(self.activation(self.input(features)))
        return params.chunk(2, dim=1)


class CNOFiLM(nn.Module):
    """Official CNO with one fixed bottleneck FiLM insertion."""

    def __init__(self, base: nn.Module, feature_group: str, hidden_channels: int = 16):
        super().__init__()
        if feature_group not in FEATURE_GROUPS:
            raise ValueError(f"unknown FF-01 group: {feature_group}")
        self.base = base
        self.feature_group = feature_group
        latent_channels = int(base.encoder_features[base.N_layers])
        self.conditioner = ConditionEncoder(latent_channels, hidden_channels=hidden_channels)

    def apply_conditioning(self, z: Tensor, features: Tensor) -> Tensor:
        gamma, beta = self.conditioner(features, (z.shape[-2], z.shape[-1]))
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)
        return z * (1.0 + gamma) + beta

    def forward(self, runtime_input: Tensor) -> Tensor:
        if runtime_input.ndim != 5 or runtime_input.shape[-1] < 2:
            raise ValueError("CNOFiLM expects [B,20,H,W,3] runtime input")
        features = build_prior_features(runtime_input, self.feature_group)
        x = runtime_input.permute(0, 4, 1, 2, 3)
        base = self.base
        skip = []

        x = base.lift(x)
        for i in range(base.N_layers):
            y = x
            for j in range(base.N_res):
                y = base.res_nets[i * base.N_res + j](y)
            skip.append(y)
            x = base.encoder[i](x)

        for j in range(base.N_res_neck):
            x = base.res_nets[-j - 1](x)

        # The one and only authorized insertion point: final encoder latent,
        # after the neck residuals and immediately before decoder expansion.
        x = self.apply_conditioning(x, features)

        for i in range(base.N_layers):
            if i == 0:
                x = base.ED_expansion[base.N_layers - i](x)
            else:
                x = torch.cat((x, base.ED_expansion[base.N_layers - i](skip[-i])), dim=1)
            if base.add_inv:
                x = base.decoder_inv[i](x)
            x = base.decoder[i](x)

        x = torch.cat((x, base.ED_expansion[0](skip[0])), dim=1)
        x = base.project(x)
        if base.out_dim_mult > 1:
            x = x.reshape(x.shape[0], -1, x.shape[2], x.shape[3], base.out_dim // base.out_dim_mult)
        return x.permute(0, 2, 3, 4, 1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def git_provenance(repo: Path) -> dict:
    status = git_value(repo, "status", "--short")
    tracked_diff = subprocess.check_output(["git", "-C", str(repo), "diff", "--binary"], text=False)
    return {
        "code_commit": git_value(repo, "rev-parse", "HEAD"),
        "working_tree_status": status or "CLEAN",
        "working_tree_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cno(kit_root: Path, checkpoint: Path, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d  # type: ignore

    base = CNO3d(in_dim=3, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model_state_dict", state)
    base.load_state_dict(state, strict=True)
    return base


def make_loader(paths: list[Path], args: argparse.Namespace, shuffle: bool):
    from realpde_p0_data import H5WindowDataset  # type: ignore

    dataset = H5WindowDataset(
        paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        include_pressure=False,
    )
    generator = torch.Generator().manual_seed(args.seed) if shuffle else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=False,
    )
    return dataset, loader


@torch.no_grad()
def evaluate(model: nn.Module, paths: list[Path], args: argparse.Namespace, device: torch.device, kit_root: Path, out: Path) -> dict:
    from realpde_loss_official_v9 import score_bundle, trajectory_rows  # type: ignore

    out.mkdir(parents=True, exist_ok=True)
    dataset, loader = make_loader(paths, args, shuffle=False)
    model.eval()
    predictions, targets = [], []
    elapsed = 0.0
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        predictions.append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    scored = score_bundle(kit_root, pred, target, elapsed / len(dataset), out)
    rows, anatomy = trajectory_rows(dataset, pred, target, kit_root)
    with (out / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scored.update({"windows": len(dataset), "trajectories": len(rows), "trajectory_anatomy": anatomy})
    scored["inference_seconds"] = elapsed
    return scored


def compare_trajectories(candidate_dir: Path, control_dir: Path | None) -> dict:
    candidate_rows = list(csv.DictReader((candidate_dir / "trajectory_metrics.csv").open()))
    candidate = {row["trajectory_id"]: row for row in candidate_rows}
    if control_dir is None:
        return {"matched_raw_control": None}
    control_rows = list(csv.DictReader((control_dir / "trajectory_metrics.csv").open()))
    control = {row["trajectory_id"]: row for row in control_rows}
    if set(candidate) != set(control):
        raise ValueError("candidate and matched Raw-Control trajectory sets differ")
    metrics = ("rel_l2", "tke", "mvpe")
    delta = {metric: float(np.mean([float(candidate[k][metric]) - float(control[k][metric]) for k in candidate])) for metric in metrics}
    win_rate = {metric: float(np.mean([float(candidate[k][metric]) < float(control[k][metric]) for k in candidate])) for metric in metrics}
    return {
        "matched_raw_control": str(control_dir),
        "trajectory_macro_delta_candidate_minus_control": delta,
        "trajectory_win_rate_vs_matched_raw_control": win_rate,
    }


def train(args: argparse.Namespace) -> None:
    if args.group not in FEATURE_GROUPS:
        raise ValueError(args.group)
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    repo = Path(args.provenance_repo).resolve() if args.provenance_repo else None
    manifest = args.manifest.resolve()
    init_checkpoint = args.init_checkpoint.resolve()
    kit_root = args.kit_root.resolve()
    if args.eval_split != "dev":
        raise ValueError("FF-01 is frozen to the dev split")
    tool_dir = (repo / "tools") if repo else Path(__file__).resolve().parent
    sys.path.insert(0, str(tool_dir))
    from realpde_loss_official_v9 import loss_parts  # type: ignore
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_paths = [args.data_root / row["file"] for row in manifest_payload["train"]]
    dev_paths = [args.data_root / row["file"] for row in manifest_payload["dev"]]
    if not all(path.is_file() for path in train_paths + dev_paths):
        missing = [str(path) for path in train_paths + dev_paths if not path.is_file()]
        raise FileNotFoundError(f"manifest files missing beneath --data-root: {missing[:3]}")
    base = load_cno(kit_root, init_checkpoint, device)
    model = CNOFiLM(base, args.group).to(device)
    weights = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757, "mean": 0.0, "fluct": 0.0}
    metadata = {
        "experiment_id": args.experiment_id,
        "group": args.group,
        "feature_names": FEATURE_GROUPS[args.group],
        "feature_input_width": FEATURE_INPUT_WIDTH,
        "fusion": "FiLM at final CNO encoder bottleneck after neck residuals and before decoder",
        "conditioner": {"hidden_channels": 16, "kernel": 1, "resize": "bilinear", "final_projection_zero_init": True, "latent_time_broadcast": True},
        "input_protocol": "[B,20,32,64,3] raw u/v plus p=0; output [B,20,32,64,3]",
        "loss_weights": weights,
        "optimizer": {"name": "AdamW", "lr": args.lr, "batch_size": args.batch_size, "workers": args.workers, "seed": args.seed, "optimizer_active_seconds": args.max_train_seconds, "checkpoint_rule": "last@budget; no dev-best selection"},
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "locked_final_accessed": False,
        "private_test_accessed": False,
        "init_checkpoint": str(init_checkpoint),
        "init_checkpoint_sha256": sha256(init_checkpoint),
        "scorer_sha256": sha256(kit_root / "scoring.py"),
        "device": str(device),
        "total_parameters": count_parameters(model),
        "added_parameters": count_parameters(model.conditioner),
        **(git_provenance(repo) if repo else {
            "code_commit": args.code_commit or "UNKNOWN",
            "working_tree_status": args.working_tree_status or "UNKNOWN",
            "working_tree_diff_sha256": args.working_tree_diff_sha256 or "UNKNOWN",
            "runner_sha256": sha256(Path(__file__).resolve()),
        }),
        "start_time": time.time(),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    train_dataset, train_loader = make_loader(train_paths, args, shuffle=True)
    baseline = evaluate(model, dev_paths, args, device, kit_root, out / "eval_initial")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    torch.save({"model_state_dict": model.state_dict(), "iteration": 0, "loss_weights": weights}, out / "model_latest.pth")
    iterator = iter(train_loader)
    train_start = time.monotonic()
    latest_parts: dict[str, float] = {}
    actual_updates = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while time.monotonic() - train_start < args.max_train_seconds:
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.train()
        parts = loss_parts(model(x), y)
        loss = sum(weights[name] * value for name, value in parts.items())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        actual_updates += 1
        latest_parts = {name: float(value.detach().cpu()) for name, value in parts.items()}
        latest_parts["total"] = float(loss.detach().cpu())
    train_seconds = time.monotonic() - train_start
    checkpoint = out / "model_latest.pth"
    torch.save({"model_state_dict": model.state_dict(), "iteration": actual_updates, "loss_weights": weights}, checkpoint)
    final = evaluate(model, dev_paths, args, device, kit_root, out / "eval_last")
    comparison = compare_trajectories(out / "eval_last", args.matched_raw_control_dir)
    metadata.update({
        "end_time": time.time(),
        "train_seconds": train_seconds,
        "actual_updates": actual_updates,
        "train_windows": len(train_dataset),
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "last_train_loss_parts": latest_parts,
    })
    summary = {"metadata": metadata, "initial": baseline, "final": final, "comparison": comparison}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        f"# FF-01 {args.group}",
        "",
        f"Experiment: `{args.experiment_id}`; group `{args.group}`.",
        "",
        "This is CLEAN offline research from sim_pretrain. No locked-final/private-test data or Codabench was used.",
        "",
        "| evaluation | Rel-L2 | TKE | MVPE |",
        "|---|---:|---:|---:|",
        f"| initial | {baseline['raw_errors']['rel_l2']:.6f} | {baseline['raw_errors']['tke']:.6f} | {baseline['raw_errors']['mvpe']:.6f} |",
        f"| last@budget | {final['raw_errors']['rel_l2']:.6f} | {final['raw_errors']['tke']:.6f} | {final['raw_errors']['mvpe']:.6f} |",
        "",
        f"Updates: `{actual_updates}`; optimizer-active seconds: `{train_seconds:.2f}`; checkpoint SHA-256: `{metadata['checkpoint_sha256']}`.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--group", choices=tuple(FEATURE_GROUPS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matched-raw-control-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-train-seconds", type=float, default=1800.0)
    parser.add_argument("--max-windows-per-trajectory", type=int)
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--provenance-repo", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--working-tree-status")
    parser.add_argument("--working-tree-diff-sha256")
    args = parser.parse_args()
    # The dataset paths in the manifest are resolved by the shared loader.  The
    # explicit data-root argument remains part of the command/provenance API;
    # this guard prevents accidental use of a different root in future edits.
    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    train(args)


if __name__ == "__main__":
    main()
