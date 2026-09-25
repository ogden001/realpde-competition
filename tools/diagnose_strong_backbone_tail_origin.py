#!/usr/bin/env python3
"""Diagnose why Exp1 Strong Backbone develops an F19/F20 cliff.

Two zero-training questions are answered in one pass:

1. Checkpoint evolution:
   Does the F19/F20 cliff exist at update 0, or does it grow during training?

2. MF raw-head anatomy:
   Does the tail anomaly already exist in the raw CNO fluctuation head, or is it
   introduced/amplified by the MF temporal zero-mean reconstruction?

Only Clean Train51 / Seen-Dev12 is permitted. No optimizer step is created.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

import realpde_sota_v2_integrated as sota
from realpde_mf01 import MF01CNO
from colleague_80pt.realpde_h5_feature_adapter_train import (
    measured_channels,
    mvpe_rel_l2_per_sample,
    rel_l2_per_sample,
    tke_rel_l2_per_sample,
)
from dw01_by_horizon import aggregate_by_horizon, compute_window_horizon_metrics

SEED = 41
EXPECTED_TRAIN = 51
EXPECTED_DEV = 12
EXPECTED_DEV_WINDOWS = 491
SIM_PRETRAIN_SHA256 = "82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61"
EXPECTED_MILESTONES = (7500, 15000, 20000, 25000, 30000, 31000, 32500, 35000)
TAIL = (18, 19, 20)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty rows")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _uv(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 5 or a.shape[1] != 20 or a.shape[-1] < 2:
        raise ValueError(f"expected [N,20,H,W,C>=2], got {a.shape}")
    return a[..., :2]


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(av - bv) / max(np.linalg.norm(bv), 1e-30))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(av, bv) / max(np.linalg.norm(av) * np.linalg.norm(bv), 1e-30))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def target_fluctuation(target: np.ndarray) -> np.ndarray:
    y = _uv(target)
    return y - y.mean(axis=1, keepdims=True)


def raw_reconstruct(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean_raw, raw_fluct, centered_fluct and reconstructed uv.

    raw is the five-channel output of MF01CNO.cno before factorized_reconstruct:
      [mean_u, mean_v, fluct_u, fluct_v, p].
    """
    a = np.asarray(raw, dtype=np.float64)
    if a.ndim != 5 or a.shape[1] != 20 or a.shape[-1] != 5:
        raise ValueError(f"expected raw [N,20,H,W,5], got {a.shape}")
    mean_raw = a[..., :2]
    raw_fluct = a[..., 2:4]
    mean_field = mean_raw.mean(axis=1, keepdims=True)
    centered = raw_fluct - raw_fluct.mean(axis=1, keepdims=True)
    reconstructed = mean_field + centered
    return mean_raw, raw_fluct, centered, reconstructed


def component_horizon_rows(raw: np.ndarray, target: np.ndarray) -> list[dict[str, object]]:
    mean_raw, raw_fluct, centered, reconstructed = raw_reconstruct(raw)
    tf = target_fluctuation(target)
    mean_field = mean_raw.mean(axis=1, keepdims=True)
    precenter = mean_field + raw_fluct
    y = _uv(target)

    raw_energy = np.sum(raw_fluct ** 2, axis=(0, 2, 3, 4))
    centered_energy = np.sum(centered ** 2, axis=(0, 2, 3, 4))
    target_energy = np.sum(tf ** 2, axis=(0, 2, 3, 4))
    raw_total = float(np.sum(raw_energy))
    centered_total = float(np.sum(centered_energy))
    target_total = float(np.sum(target_energy))

    mean_temporal_center = mean_raw.mean(axis=1, keepdims=True)
    rows: list[dict[str, object]] = []
    for h in range(20):
        rr = raw_fluct[:, h]
        cc = centered[:, h]
        tt = tf[:, h]
        mr = mean_raw[:, h]
        rows.append({
            "horizon": h + 1,
            "raw_fluct_rms": _rms(rr),
            "centered_fluct_rms": _rms(cc),
            "target_fluct_rms": _rms(tt),
            "raw_fluct_amp_ratio": float(np.linalg.norm(rr.reshape(-1)) / max(np.linalg.norm(tt.reshape(-1)), 1e-30)),
            "centered_fluct_amp_ratio": float(np.linalg.norm(cc.reshape(-1)) / max(np.linalg.norm(tt.reshape(-1)), 1e-30)),
            "raw_fluct_cosine": _cos(rr, tt),
            "centered_fluct_cosine": _cos(cc, tt),
            "raw_fluct_rel_l2": _rel(rr, tt),
            "centered_fluct_rel_l2": _rel(cc, tt),
            "raw_energy_share": float(raw_energy[h] / max(raw_total, 1e-30)),
            "centered_energy_share": float(centered_energy[h] / max(centered_total, 1e-30)),
            "target_energy_share": float(target_energy[h] / max(target_total, 1e-30)),
            "precenter_frame_rel_l2": _rel(precenter[:, h], y[:, h]),
            "reconstructed_frame_rel_l2": _rel(reconstructed[:, h], y[:, h]),
            "mean_raw_deviation_rms": _rms(mr - mean_temporal_center[:, 0]),
        })
    return rows


def whole_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p = np.asarray(prediction, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    if p.shape[-1] == 2:
        p = np.concatenate([p, np.zeros_like(p[..., :1])], axis=-1)
    channels = measured_channels(y)
    return {
        "rel_l2": float(np.mean(rel_l2_per_sample(p, y, channels))),
        "tke": float(np.mean(tke_rel_l2_per_sample(p, y, channels))),
        "mvpe": float(np.mean(mvpe_rel_l2_per_sample(p, y))),
    }


def cliff_summary(horizon_rows: list[dict[str, object]], component_rows: list[dict[str, object]]) -> dict[str, float]:
    hmap = {int(row["horizon"]): row for row in horizon_rows}
    cmap = {int(row["horizon"]): row for row in component_rows}

    def growth(a: int, b: int, key: str) -> float:
        x, z = float(hmap[a][key]), float(hmap[b][key])
        return 100.0 * (z - x) / max(abs(x), 1e-30)

    return {
        "f18_rel": float(hmap[18]["frame_rel_l2"]),
        "f19_rel": float(hmap[19]["frame_rel_l2"]),
        "f20_rel": float(hmap[20]["frame_rel_l2"]),
        "rel_growth_f18_f19_pct": growth(18, 19, "frame_rel_l2"),
        "rel_growth_f19_f20_pct": growth(19, 20, "frame_rel_l2"),
        "rel_growth_f18_f20_pct": growth(18, 20, "frame_rel_l2"),
        "raw_fluct_amp_ratio_f18": float(cmap[18]["raw_fluct_amp_ratio"]),
        "raw_fluct_amp_ratio_f19": float(cmap[19]["raw_fluct_amp_ratio"]),
        "raw_fluct_amp_ratio_f20": float(cmap[20]["raw_fluct_amp_ratio"]),
        "centered_fluct_amp_ratio_f18": float(cmap[18]["centered_fluct_amp_ratio"]),
        "centered_fluct_amp_ratio_f19": float(cmap[19]["centered_fluct_amp_ratio"]),
        "centered_fluct_amp_ratio_f20": float(cmap[20]["centered_fluct_amp_ratio"]),
        "centered_fluct_cosine_f18": float(cmap[18]["centered_fluct_cosine"]),
        "centered_fluct_cosine_f19": float(cmap[19]["centered_fluct_cosine"]),
        "centered_fluct_cosine_f20": float(cmap[20]["centered_fluct_cosine"]),
        "raw_tail_energy_share_f19_f20": float(cmap[19]["raw_energy_share"]) + float(cmap[20]["raw_energy_share"]),
        "centered_tail_energy_share_f19_f20": float(cmap[19]["centered_energy_share"]) + float(cmap[20]["centered_energy_share"]),
        "target_tail_energy_share_f19_f20": float(cmap[19]["target_energy_share"]) + float(cmap[20]["target_energy_share"]),
        "centering_f20_rel_delta_pct": 100.0 * (
            float(cmap[20]["reconstructed_frame_rel_l2"]) - float(cmap[20]["precenter_frame_rel_l2"])
        ) / max(float(cmap[20]["precenter_frame_rel_l2"]), 1e-30),
    }


def discover_milestones(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    pattern = re.compile(r"model_update_(\d{5})\.pth$")
    for path in checkpoint_dir.glob("model_update_*.pth"):
        m = pattern.match(path.name)
        if m:
            found[int(m.group(1))] = path
    missing = [u for u in EXPECTED_MILESTONES if u not in found]
    if missing:
        raise FileNotFoundError(f"missing frozen Strong Backbone milestone checkpoints: {missing}")
    unexpected = sorted(set(found) - set(EXPECTED_MILESTONES))
    if unexpected:
        raise ValueError(f"unexpected milestone checkpoints in directory: {unexpected}")
    return [(u, found[u]) for u in EXPECTED_MILESTONES]


def load_model_for_update(
    *,
    update: int,
    path: Path,
    builder,
    kit_root: Path,
    device: torch.device,
    init_direct: bool,
) -> tuple[MF01CNO, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MF01CNO(kit_root, len(builder.feature_names), device)
    if init_direct:
        if update != 0:
            raise ValueError("init_direct is only valid for update 0")
        sota.init_mf_from_direct(model, payload, len(builder.feature_names))
        meta = {"iteration": 0, "source": "official_sim_real_direct_to_mf_init"}
    else:
        iteration = int(payload.get("iteration", -1))
        if iteration != update:
            raise ValueError(f"checkpoint iteration mismatch: filename/update={update}, payload={iteration}")
        if str(payload.get("feature_set", "")) != "P0-A":
            raise ValueError(f"checkpoint {path} is not frozen P0-A")
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint {path} missing model_state_dict")
        model.load_state_dict(state, strict=True)
        meta = {
            "iteration": iteration,
            "feature_set": payload.get("feature_set"),
            "selection_score": payload.get("selection_score"),
            "source": "strong_backbone_milestone",
        }
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, meta


@torch.inference_mode()
def evaluate_checkpoint(model, builder, loader, device: torch.device) -> dict[str, object]:
    preds: list[np.ndarray] = []
    raws: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: list[str] = []
    starts: list[int] = []

    for x, y, batch_names, batch_starts in loader:
        x = x.to(device, non_blocking=True)
        features = builder(x)
        raw = model.cno(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        mean_raw = raw[..., :2]
        fluct_raw = raw[..., 2:4]
        mean_field = mean_raw.mean(dim=1, keepdim=True).expand_as(mean_raw)
        centered = fluct_raw - fluct_raw.mean(dim=1, keepdim=True)
        uv = mean_field + centered

        pred = model(features).clone()
        parity = float((pred[..., :2] - uv).abs().max().cpu())
        if parity > 2e-6:
            raise RuntimeError(f"MF reconstruction parity failed: {parity}")
        pred[..., 2] = 0.0

        preds.append(pred.cpu().numpy().astype(np.float32))
        raws.append(raw.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
        names.extend([str(v) for v in batch_names])
        if torch.is_tensor(batch_starts):
            starts.extend([int(v) for v in batch_starts.cpu().tolist()])
        else:
            starts.extend([int(v) for v in batch_starts])

    pred_np = np.concatenate(preds, axis=0)
    raw_np = np.concatenate(raws, axis=0)
    target_np = np.concatenate(targets, axis=0)
    if pred_np.shape[0] != EXPECTED_DEV_WINDOWS:
        raise ValueError(f"expected {EXPECTED_DEV_WINDOWS} dev windows, got {pred_np.shape[0]}")

    window_rows = compute_window_horizon_metrics(pred_np, target_np, names, starts)
    horizon_rows = aggregate_by_horizon(window_rows, experiment="strong_backbone_checkpoint", trajectories=EXPECTED_DEV)
    component_rows = component_horizon_rows(raw_np, target_np)
    metrics = whole_metrics(pred_np, target_np)
    return {
        "metrics": metrics,
        "horizon_rows": horizon_rows,
        "component_rows": component_rows,
        "cliff": cliff_summary(horizon_rows, component_rows),
    }


def classify_evolution(curve: list[dict[str, object]]) -> dict[str, object]:
    if len(curve) < 3 or int(curve[0]["update"]) != 0:
        raise ValueError("curve must begin at update 0 and contain milestones")
    init = curve[0]
    final = curve[-1]
    init_cliff = float(init["rel_growth_f18_f20_pct"])
    final_cliff = float(final["rel_growth_f18_f20_pct"])
    raw_init = float(init["raw_fluct_amp_ratio_f20"])
    raw_final = float(final["raw_fluct_amp_ratio_f20"])
    center_final = float(final["centered_fluct_amp_ratio_f20"])
    centering_delta = float(final["centering_f20_rel_delta_pct"])

    return {
        "update0_rel_growth_f18_f20_pct": init_cliff,
        "final_rel_growth_f18_f20_pct": final_cliff,
        "cliff_growth_from_init_pp": final_cliff - init_cliff,
        "raw_fluct_amp_ratio_f20_update0": raw_init,
        "raw_fluct_amp_ratio_f20_final": raw_final,
        "centered_fluct_amp_ratio_f20_final": center_final,
        "final_centering_f20_rel_delta_pct": centering_delta,
        "diagnostic_flags": {
            "cliff_present_at_init": init_cliff >= 15.0,
            "cliff_grows_materially_during_training": final_cliff >= init_cliff + 15.0,
            "raw_head_f20_amplitude_grows_materially": raw_final >= raw_init * 1.25,
            "mf_centering_materially_worsens_f20": centering_delta >= 10.0,
        },
        "interpretation_rule": (
            "Cliff present at update0 supports architecture/boundary initialization effects. "
            "A cliff that grows during training supports objective/optimization-induced temporal energy allocation. "
            "If raw fluctuation F20 is already abnormal before centering, the anomaly is in the learned CNO/MF raw head; "
            "if raw is smooth but centered/reconstructed F20 becomes abnormal, MF zero-mean reconstruction is implicated."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    for path in (args.manifest, args.data_root, args.init_checkpoint, args.checkpoint_dir, args.kit_root):
        if not path.exists():
            raise FileNotFoundError(path)

    init_sha = sha256(args.init_checkpoint)
    if init_sha != SIM_PRETRAIN_SHA256:
        raise RuntimeError(f"wrong official sim_real checkpoint SHA: {init_sha}")

    train = sota.split_paths(args.manifest, "train", args.data_root)
    dev = sota.split_paths(args.manifest, "dev", args.data_root)
    if (len(train), len(dev)) != (EXPECTED_TRAIN, EXPECTED_DEV):
        raise ValueError(f"Clean split must be {EXPECTED_TRAIN}/{EXPECTED_DEV}, got {len(train)}/{len(dev)}")
    if set(p.name for p in train) & set(p.name for p in dev):
        raise ValueError("train/dev overlap")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    builder, _feature_config = sota.build_features(train, device)
    ds, loader = sota.dev_loader(
        dev,
        argparse.Namespace(eval_batch_size=args.eval_batch_size, workers=args.workers),
    )
    if len(ds) != EXPECTED_DEV_WINDOWS:
        raise ValueError(f"expected {EXPECTED_DEV_WINDOWS} dev windows, got {len(ds)}")

    milestones = discover_milestones(args.checkpoint_dir)
    sources: list[tuple[int, Path, bool]] = [(0, args.init_checkpoint, True)]
    sources.extend((update, path, False) for update, path in milestones)

    curve: list[dict[str, object]] = []
    checksums: dict[str, str] = {"update_00000": init_sha}

    for update, path, is_init in sources:
        model, meta = load_model_for_update(
            update=update,
            path=path,
            builder=builder,
            kit_root=args.kit_root,
            device=device,
            init_direct=is_init,
        )
        result = evaluate_checkpoint(model, builder, loader, device)
        row: dict[str, object] = {
            "update": update,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256(path),
            **result["metrics"],
            **result["cliff"],
        }
        curve.append(row)
        checksums[f"update_{update:05d}"] = row["checkpoint_sha256"]

        sub = args.out_dir / f"update_{update:05d}"
        sub.mkdir()
        write_rows(sub / "by_horizon.csv", result["horizon_rows"])
        write_rows(sub / "raw_mf_by_horizon.csv", result["component_rows"])
        dump(sub / "summary.json", {"update": update, "meta": meta, "metrics": result["metrics"], "cliff": result["cliff"]})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_rows(args.out_dir / "checkpoint_curve.csv", curve)
    evolution = classify_evolution(curve)
    dump(args.out_dir / "evolution_summary.json", evolution)
    dump(args.out_dir / "checkpoint_sha256.json", checksums)
    dump(args.out_dir / "run_manifest.json", {
        "status": "REVIEW_REQUIRED",
        "purpose": "Strong Backbone F19/F20 checkpoint evolution and raw MF anatomy",
        "seed": SEED,
        "train_trajectories_in_manifest": len(train),
        "eval_trajectories": len(dev),
        "eval_windows": len(ds),
        "updates": [int(row["update"]) for row in curve],
        "sim_pretrain_sha256": init_sha,
        "optimizer_steps": 0,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "aoa10_accessed": False,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "full_data_refit_started": False,
        "submission_packaging_started": False,
    })
    (args.out_dir / "DONE").touch()


if __name__ == "__main__":
    main()
