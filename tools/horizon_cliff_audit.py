#!/usr/bin/env python3
"""Read-only Future20 horizon audit for frozen Track 1 development windows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder, P0FeatureConfig


EXPECTED_SHAPE = (659, 20, 32, 64, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def windowwise_horizon_rel_l2(prediction: np.ndarray, target: np.ndarray) -> list[float]:
    """Mean window-wise velocity Rel-L2, independently for each Future20 frame."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(f"prediction/target must share rank-5 shape, got {prediction.shape}/{target.shape}")
    values: list[float] = []
    for horizon in range(prediction.shape[1]):
        pred_h = prediction[:, horizon, ..., :2]
        target_h = target[:, horizon, ..., :2]
        numerator = np.linalg.norm((pred_h - target_h).reshape(pred_h.shape[0], -1), axis=1)
        denominator = np.linalg.norm(target_h.reshape(target_h.shape[0], -1), axis=1)
        values.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return values


def verify_window_alignment(past: np.ndarray, future: np.ndarray, trajectory: np.ndarray, *, start: int) -> dict[str, list[int]]:
    """Assert Past20 / Future20 are exact contiguous slices of one raw trajectory."""
    np.testing.assert_array_equal(past, trajectory[start : start + 20])
    np.testing.assert_array_equal(future, trajectory[start + 20 : start + 40], err_msg="Future20 must begin at start+20")
    return {"future_frame_indices": [start + 20, start + 38, start + 39]}


def paths_from_manifest(manifest: Path, data_root: Path, split: str) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    paths = [data_root / row["file"] for row in payload[split]]
    if len(paths) != (50 if split == "train" else 16) or any(not path.is_file() for path in paths):
        raise ValueError(f"invalid frozen {split} paths")
    return paths


def raw_uv(path: Path, start: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        u = np.asarray(handle["u"][start : start + 40, ::2, ::2], dtype=np.float32)
        v = np.asarray(handle["v"][start : start + 40, ::2, ::2], dtype=np.float32)
    return np.stack([u, v], axis=-1)


def verify_all_target_alignment(dataset: H5WindowDataset) -> dict[str, list[int]]:
    """Check every dev window's target against raw start+20:start+40 u/v."""
    reported: dict[str, list[int]] | None = None
    for index, ref in enumerate(dataset.refs):
        past, future, _, _ = dataset[index]
        raw = raw_uv(ref.path, ref.start)
        details = verify_window_alignment(past.numpy()[..., :2], future.numpy()[..., :2], raw, start=0)
        if index == 0:
            reported = {"first_window": details["future_frame_indices"]}
        if index == len(dataset) - 1:
            assert reported is not None
            reported["last_window"] = details["future_frame_indices"]
    assert reported is not None
    return reported


def load_cno(kit_root: Path, *, in_dim: int, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    model = CNO3d(in_dim=in_dim, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    state = torch.load(checkpoint, map_location="cpu").get("model_state_dict", torch.load(checkpoint, map_location="cpu"))
    model.load_state_dict(state, strict=True)
    return model.eval()


def forward_p0a(model: torch.nn.Module, builder: P0FeatureBuilder, x: torch.Tensor) -> torch.Tensor:
    return model(builder(x).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


def p0a_config_from_checkpoint(payload: dict) -> P0FeatureConfig:
    """Restore historical P0-A checkpoints which serialized only grid spacing."""
    if payload.get("feature_set") != "P0-A":
        raise ValueError("checkpoint is not P0-A")
    saved = dict(payload.get("feature_config", {}))
    saved["include_p0_a"] = True
    saved["include_p0_b"] = False
    return P0FeatureConfig(**saved)


def prepare_out_dir(path: Path) -> None:
    """Permit an empty container bind mount, never overwrite prior evidence."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=True)


def horizon_csv_fieldnames(*, include_plain: bool) -> list[str]:
    fields = ["horizon", "persist_rel_l2", "official_cno_rel_l2", "p0a_cno_rel_l2"]
    if include_plain:
        fields.append("plain_piv_cno_rel_l2")
    return fields


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    train_paths = paths_from_manifest(args.manifest, args.data_root, "train")
    dev_paths = paths_from_manifest(args.manifest, args.data_root, "dev")
    dataset = H5WindowDataset(dev_paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    if len(dataset) != EXPECTED_SHAPE[0]:
        raise AssertionError(f"expected 659 dev windows, got {len(dataset)}")
    alignment = verify_all_target_alignment(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = load_cno(args.kit_root, in_dim=3, checkpoint=args.official_checkpoint, device=device)
    payload = torch.load(args.p0a_checkpoint, map_location="cpu")
    config = p0a_config_from_checkpoint(payload)
    builder = P0FeatureBuilder(config).to(device)
    p0a = load_cno(args.kit_root, in_dim=len(builder.feature_names), checkpoint=args.p0a_checkpoint, device=device)
    plain = None if args.plain_piv_checkpoint is None else load_cno(args.kit_root, in_dim=3, checkpoint=args.plain_piv_checkpoint, device=device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    persist_all: list[np.ndarray] = []
    official_all: list[np.ndarray] = []
    p0a_all: list[np.ndarray] = []
    plain_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []
    for past, target, _, _ in loader:
        x = past.to(device)
        persist_all.append(np.repeat(past[:, -1:].numpy(), 20, axis=1))
        official_all.append(baseline(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1).cpu().numpy())
        p0a_all.append(forward_p0a(p0a, builder, x).cpu().numpy())
        if plain is not None:
            plain_all.append(plain(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1).cpu().numpy())
        target_all.append(target.numpy())
    persist, official, p0a, target = (np.concatenate(values) for values in (persist_all, official_all, p0a_all, target_all))
    for name, value in {"persist": persist, "official": official, "p0a": p0a, "target": target}.items():
        if value.shape != EXPECTED_SHAPE:
            raise AssertionError(f"{name} has unexpected shape {value.shape}")
    metrics = {"persist": windowwise_horizon_rel_l2(persist, target), "official": windowwise_horizon_rel_l2(official, target), "p0a": windowwise_horizon_rel_l2(p0a, target)}
    if plain is not None:
        plain_prediction = np.concatenate(plain_all)
        if plain_prediction.shape != EXPECTED_SHAPE:
            raise AssertionError(f"plain PIV CNO has unexpected shape {plain_prediction.shape}")
        metrics["plain"] = windowwise_horizon_rel_l2(plain_prediction, target)
    prepare_out_dir(args.out_dir)
    with (args.out_dir / "horizon_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=horizon_csv_fieldnames(include_plain=plain is not None))
        writer.writeheader()
        for h in range(20):
            row = {"horizon": h + 1, "persist_rel_l2": metrics["persist"][h], "official_cno_rel_l2": metrics["official"][h], "p0a_cno_rel_l2": metrics["p0a"][h]}
            if plain is not None: row["plain_piv_cno_rel_l2"] = metrics["plain"][h]
            writer.writerow(row)
    ratios = {name: {"t19_t18": values[18] / values[17], "t20_t18": values[19] / values[17], "t20_mean_t1_t18": values[19] / float(np.mean(values[:18]))} for name, values in metrics.items()}
    evidence = {"shape": list(target.shape), "train_trajectories": len(train_paths), "dev_trajectories": len(dev_paths), "dev_windows": len(dataset), "alignment": alignment, "ratios": ratios, "official_checkpoint_sha256": sha256(args.official_checkpoint), "p0a_checkpoint_sha256": sha256(args.p0a_checkpoint), "manifest_sha256": sha256(args.manifest)}
    if args.plain_piv_checkpoint is not None: evidence["plain_piv_checkpoint_sha256"] = sha256(args.plain_piv_checkpoint)
    (args.out_dir / "audit_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True); parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--p0a-checkpoint", type=Path, required=True); parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plain-piv-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__": main()
