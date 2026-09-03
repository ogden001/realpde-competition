#!/usr/bin/env python3
"""Fail-fast checks and result summary for the 2026-09-04 SOTA LR screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

EXPECTED_START_UPDATE = 18860
TARGET_UPDATE = 22960
CONTROL_LR = 1e-5
DECAY_LR = 5e-6
EXPECTED_MANIFEST_SHA256 = "42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347"
EXPECTED_SCORER_SHA256 = "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39"
METRICS = ("rel_l2", "tke", "mvpe")
N2_WEIGHTS = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_resume_payload(payload: dict, expected_update: int = EXPECTED_START_UPDATE) -> None:
    if payload.get("iteration") != expected_update:
        raise ValueError(f"resume checkpoint iteration must be {expected_update}")
    if payload.get("feature_set") != "P0-A" or payload.get("loss_weights") != N2_WEIGHTS:
        raise ValueError("resume checkpoint must be P0-A + N2")
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or not optimizer.get("state") or not optimizer.get("param_groups"):
        raise ValueError("resume checkpoint must contain optimizer state")


def preflight(manifest: Path, kit_root: Path, resume_checkpoint: Path) -> dict:
    scorer = kit_root / "scoring.py"
    for path in (manifest, scorer, resume_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_sha = sha256(manifest)
    scorer_sha = sha256(scorer)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"manifest SHA mismatch: {manifest_sha}")
    if scorer_sha != EXPECTED_SCORER_SHA256:
        raise ValueError(f"scorer SHA mismatch: {scorer_sha}")
    payload = torch.load(resume_checkpoint, map_location="cpu")
    validate_resume_payload(payload)
    optimizer_lrs = sorted({float(group["lr"]) for group in payload["optimizer_state_dict"]["param_groups"]})
    return {
        "state": "PREFLIGHT_OK",
        "manifest_sha256": manifest_sha,
        "scorer_sha256": scorer_sha,
        "resume_checkpoint_sha256": sha256(resume_checkpoint),
        "resume_iteration": payload["iteration"],
        "resume_optimizer_lrs": optimizer_lrs,
        "decision": "READY_FOR_LR_AB",
    }


def validate_run_dir(run_dir: Path, *, expected_update: int, expected_lr: float) -> dict[str, float]:
    status = read_json(run_dir / "status.json")
    metadata = read_json(run_dir / "run_metadata.json")
    metrics_payload = read_json(run_dir / f"dev_{expected_update:05d}.json")
    if (
        status.get("state") != "DONE"
        or status.get("stop_reason") != "updates_complete"
        or status.get("update") != expected_update
    ):
        raise ValueError(f"run did not finish cleanly at update {expected_update}: {status}")
    if metadata.get("initial_update") != EXPECTED_START_UPDATE:
        raise ValueError(f"run initial update must be {EXPECTED_START_UPDATE}")
    lrs = metadata.get("optimizer_lrs")
    if metadata.get("lr") != expected_lr or lrs != [expected_lr]:
        raise ValueError(
            f"run learning rate does not match expected {expected_lr}: "
            f"lr={metadata.get('lr')} optimizer_lrs={lrs}"
        )
    if metrics_payload.get("update") != expected_update:
        raise ValueError("dev metric update mismatch")
    raw = metrics_payload.get("raw_errors", {})
    if any(metric not in raw for metric in METRICS):
        raise ValueError("dev metrics are incomplete")
    return {metric: float(raw[metric]) for metric in METRICS}


def summarize_ab(control: dict[str, float], decay: dict[str, float]) -> dict:
    for metric in METRICS:
        if metric not in control or metric not in decay:
            raise ValueError(f"missing metric {metric}")
    delta = {metric: float(decay[metric] - control[metric]) for metric in METRICS}
    return {
        "decision": "REVIEW_REQUIRED",
        "control_lr_1e5": {metric: float(control[metric]) for metric in METRICS},
        "decay_lr_5e6": {metric: float(decay[metric]) for metric in METRICS},
        "delta_decay_minus_control": delta,
        "improved_metrics": [metric for metric in METRICS if delta[metric] < 0],
        "note": "Lower raw error is better. No automatic winner is selected.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--manifest", type=Path, required=True)
    preflight_parser.add_argument("--kit-root", type=Path, required=True)
    preflight_parser.add_argument("--resume-checkpoint", type=Path, required=True)
    preflight_parser.add_argument("--out", type=Path, required=True)

    check_parser = sub.add_parser("check-run")
    check_parser.add_argument("--run-dir", type=Path, required=True)
    check_parser.add_argument("--expected-update", type=int, required=True)
    check_parser.add_argument("--expected-lr", type=float, required=True)

    summarize_parser = sub.add_parser("summarize")
    summarize_parser.add_argument("--control-dir", type=Path, required=True)
    summarize_parser.add_argument("--decay-dir", type=Path, required=True)
    summarize_parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.manifest, args.kit_root, args.resume_checkpoint)
        write_json(args.out, result)
        print(json.dumps(result, sort_keys=True))
    elif args.command == "check-run":
        result = validate_run_dir(args.run_dir, expected_update=args.expected_update, expected_lr=args.expected_lr)
        print(json.dumps({"state": "RUN_OK", "raw_errors": result}, sort_keys=True))
    else:
        control = validate_run_dir(args.control_dir, expected_update=TARGET_UPDATE, expected_lr=CONTROL_LR)
        decay = validate_run_dir(args.decay_dir, expected_update=TARGET_UPDATE, expected_lr=DECAY_LR)
        result = summarize_ab(control, decay)
        write_json(args.out, result)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
