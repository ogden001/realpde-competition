#!/usr/bin/env python3
"""Unattended serial SOTA-merge comparison supervisor.

Frozen scientific pipeline:
  Stage-A @57k
    -> Stage-B 6k, P00-only, LR=3e-6, carry optimizer
    -> select Stage-B checkpoint by Seen-Dev12 point score only
    -> freeze backbone
    -> original Strong-Backbone residual recipe, 22k
    -> select residual checkpoint by Seen-Dev12 point score only
    -> post-hoc AoA10 audit over every saved milestone
    -> final Seen/AoA evidence
    -> STOP

Runtime/environment failures may be retried from exact recovery checkpoints.
Scientific parameters may not be changed by the supervisor.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

import realpde_sota_v2_integrated as strong
import train_sota_merge_backbone as merge

STATUS = "REVIEW_REQUIRED"
PROTOCOL = "REALPDE_SOTA_MERGE_SERIAL_UNATTENDED_V1"
STAGE_A_UPDATE = 57_000
STAGE_B_UPDATES = 6_000
STAGE_B_INTERVAL = 500
RESIDUAL_UPDATES = 22_000
RESIDUAL_INTERVAL = 1_000
POLL_SECONDS = 120
MAX_ATTEMPTS = 5
TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def safe(path: Path) -> None:
    merge.assert_safe_path(path)


def git_state() -> dict[str, object]:
    branch = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
        text=True,
    ).strip()
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        text=True,
    )
    if branch != "main":
        raise RuntimeError(f"serial experiment must run on main, got {branch}")
    return {"branch": branch, "head": head, "dirty": bool(status.strip()), "status_porcelain": status.splitlines()}


def require_inputs(args: argparse.Namespace) -> dict[str, object]:
    for path in (
        args.data_root, args.manifest, args.aoa_manifest, args.kit_root,
        args.init_checkpoint, args.stage_a_checkpoint, args.out_root,
    ):
        safe(path)
    for path in (
        args.data_root, args.manifest, args.aoa_manifest, args.kit_root,
        args.init_checkpoint, args.stage_a_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    source = torch.load(args.stage_a_checkpoint, map_location="cpu", weights_only=False)
    if source.get("stage") != "A" or source.get("scope") != "clean":
        raise ValueError("source must be clean Stage-A checkpoint")
    if int(source.get("iteration", -1)) != STAGE_A_UPDATE:
        raise ValueError(f"source must be Stage-A @{STAGE_A_UPDATE}")
    if source.get("feature_set") != "P0-A":
        raise ValueError("source must use P0-A features")
    if "optimizer_state_dict" not in source:
        raise ValueError("Stage-A source must carry optimizer state for Stage-B carry policy")

    aoa = json.loads(args.aoa_manifest.read_text(encoding="utf-8"))
    if int(aoa.get("heldout_aoa", -1)) != 10:
        raise ValueError("AoA audit manifest must be heldout AoA10")
    if aoa.get("selection_allowed") is not False:
        raise ValueError("AoA10 manifest must explicitly forbid model selection")
    dev_rows = aoa.get("dev")
    if not isinstance(dev_rows, list) or len(dev_rows) != 18:
        raise ValueError("AoA10 audit requires 18 heldout trajectories")

    return {
        "git": git_state(),
        "gpu": torch.cuda.get_device_name(0),
        "source_stage_a_sha256": strong.sha256(args.stage_a_checkpoint),
        "source_stage_a_update": STAGE_A_UPDATE,
        "aoa_manifest": str(args.aoa_manifest),
        "aoa_selection_allowed": False,
    }


def heartbeat(args: argparse.Namespace, *, phase: str, pid: int | None, attempt: int, extra: dict[str, object] | None = None) -> None:
    payload: dict[str, object] = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "phase": phase,
        "pid": pid,
        "attempt": attempt,
        "timestamp_unix": time.time(),
        "locked_final_accessed": False,
        "private_accessed": False,
        "codabench_accessed": False,
        "sps_started": False,
        "full_data_started": False,
    }
    if extra:
        payload.update(extra)
    dump(args.out_root / "heartbeat.json", payload)


def run_process(
    args: argparse.Namespace,
    *,
    phase: str,
    cmd: list[str],
    log_path: Path,
    attempt: int,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n=== ATTEMPT %d ===\n%s\n" % (attempt, " ".join(cmd)))
        log.flush()
        p = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=REPO,
            start_new_session=True,
        )
        while True:
            rc = p.poll()
            heartbeat(args, phase=phase, pid=p.pid, attempt=attempt)
            if rc is not None:
                return int(rc)
            time.sleep(args.poll_seconds)


def archive_failed_precheckpoint(out_dir: Path, attempt: int) -> None:
    if not out_dir.exists():
        return
    target = out_dir.with_name(out_dir.name + f".failed_precheckpoint_attempt{attempt}")
    suffix = 1
    while target.exists():
        target = out_dir.with_name(out_dir.name + f".failed_precheckpoint_attempt{attempt}_{suffix}")
        suffix += 1
    out_dir.rename(target)


def run_stage_b(args: argparse.Namespace) -> Path:
    out = args.out_root / "stage_b"
    done = out / "DONE_TO_REQUESTED_UPDATE"
    latest = out / "checkpoints" / "model_latest.pth"

    for attempt in range(1, args.max_attempts + 1):
        if done.is_file():
            break
        cmd = [
            sys.executable, "-u", "-B", str(TOOLS / "train_sota_merge_backbone.py"),
            "stage-b",
            "--scope", "clean",
            "--data-root", str(args.data_root),
            "--manifest", str(args.manifest),
            "--init-checkpoint", str(args.init_checkpoint),
            "--kit-root", str(args.kit_root),
            "--out-dir", str(out),
            "--max-updates", str(STAGE_B_UPDATES),
            "--batch-size", "8",
            "--workers", str(args.workers),
            "--prefetch-factor", str(args.prefetch_factor),
            "--eval-interval", str(STAGE_B_INTERVAL),
            "--checkpoint-interval", str(STAGE_B_INTERVAL),
            "--log-interval", "100",
            "--backbone-checkpoint", str(args.stage_a_checkpoint),
            "--optimizer-policy", "carry",
            "--require-cuda",
        ]
        if out.exists():
            if latest.is_file():
                cmd.extend(["--resume-checkpoint", str(latest)])
            else:
                archive_failed_precheckpoint(out, attempt)
        rc = run_process(
            args,
            phase="STAGE_B_TRAIN",
            cmd=cmd,
            log_path=args.out_root / "logs" / "stage_b.log",
            attempt=attempt,
        )
        if rc == 0 and done.is_file():
            break
        heartbeat(args, phase="STAGE_B_RETRY", pid=None, attempt=attempt, extra={"return_code": rc})
        time.sleep(args.retry_cooldown_seconds)
    if not done.is_file():
        raise RuntimeError("Stage-B did not complete after recovery attempts")

    best = out / "checkpoints" / "model_best.pth"
    if not best.is_file():
        raise FileNotFoundError(best)
    p = torch.load(best, map_location="cpu", weights_only=False)
    if p.get("stage") != "B":
        raise ValueError("selected Stage-B checkpoint metadata mismatch")
    if int(p.get("metadata", {}).get("source_stage_a_update", -1)) != STAGE_A_UPDATE:
        raise ValueError("selected Stage-B checkpoint has wrong Stage-A source")
    return best


def run_residual(args: argparse.Namespace, backbone: Path) -> Path:
    out = args.out_root / "residual"
    done = out / "DONE"
    recovery = out / "checkpoints" / "runner_latest.pth"

    for attempt in range(1, args.max_attempts + 1):
        if done.is_file():
            break
        cmd = [
            sys.executable, "-u", "-B", str(TOOLS / "train_sota_merge_serial_residual.py"),
            "--data-root", str(args.data_root),
            "--manifest", str(args.manifest),
            "--kit-root", str(args.kit_root),
            "--backbone-checkpoint", str(backbone),
            "--out-dir", str(out),
            "--updates", str(RESIDUAL_UPDATES),
            "--batch-size", "8",
            "--eval-interval", str(RESIDUAL_INTERVAL),
            "--recovery-every", "500",
            "--eval-batch-size", "32",
            "--workers", str(args.workers),
            "--prefetch-factor", str(args.prefetch_factor),
            "--log-every", "100",
            "--preload-to-ram",
            "--require-cuda",
        ]
        if out.exists():
            if recovery.is_file():
                cmd.append("--resume")
            else:
                archive_failed_precheckpoint(out, attempt)
        rc = run_process(
            args,
            phase="RESIDUAL_TRAIN",
            cmd=cmd,
            log_path=args.out_root / "logs" / "residual.log",
            attempt=attempt,
        )
        if rc == 0 and done.is_file():
            break
        heartbeat(args, phase="RESIDUAL_RETRY", pid=None, attempt=attempt, extra={"return_code": rc})
        time.sleep(args.retry_cooldown_seconds)
    if not done.is_file():
        raise RuntimeError("Residual stage did not complete after recovery attempts")
    best = out / "checkpoints" / "model_best.pth"
    if not best.is_file():
        raise FileNotFoundError(best)
    return best


def eval_backbone(args: argparse.Namespace, checkpoint: Path, out: Path, split_name: str) -> None:
    if (out / "DONE").is_file():
        return
    if out.exists():
        shutil.rmtree(out)
    cmd = [
        sys.executable, "-u", "-B", str(TOOLS / "evaluate_sota_merge_backbone_manifest.py"),
        "--data-root", str(args.data_root),
        "--manifest", str(args.aoa_manifest),
        "--kit-root", str(args.kit_root),
        "--checkpoint", str(checkpoint),
        "--out-dir", str(out),
        "--split-name", split_name,
        "--eval-batch-size", "8",
        "--workers", str(args.workers),
        "--require-cuda",
    ]
    rc = run_process(args, phase="STAGE_B_AOA_AUDIT", cmd=cmd, log_path=args.out_root / "logs" / "stage_b_aoa.log", attempt=1)
    if rc != 0:
        raise RuntimeError(f"backbone AoA evaluation failed: {checkpoint}")


def eval_residual(args: argparse.Namespace, backbone: Path, residual: Path, manifest: Path, out: Path, phase: str) -> None:
    if (out / "DONE").is_file():
        return
    if out.exists():
        shutil.rmtree(out)
    cmd = [
        sys.executable, "-u", "-B", str(TOOLS / "evaluate_clean_residual_checkpoint.py"),
        "--real-root", str(args.data_root),
        "--split-manifest", str(manifest),
        "--backbone-checkpoint", str(backbone),
        "--residual-checkpoint", str(residual),
        "--model-root", str(args.kit_root),
        "--base-model", "sota_v2_mf",
        "--out-dir", str(out),
        "--batch-size", "32",
        "--workers", str(args.workers),
        "--require-cuda",
    ]
    rc = run_process(args, phase=phase, cmd=cmd, log_path=args.out_root / "logs" / f"{phase.lower()}.log", attempt=1)
    if rc != 0:
        raise RuntimeError(f"residual evaluation failed: {residual}")


def stage_b_aoa_curve(args: argparse.Namespace, selected_backbone: Path) -> list[dict[str, object]]:
    root = args.out_root / "stage_b_aoa_curve"
    rows: list[dict[str, object]] = []
    checkpoints = args.out_root / "stage_b" / "checkpoints"
    for update in range(STAGE_B_INTERVAL, STAGE_B_UPDATES + 1, STAGE_B_INTERVAL):
        ck = checkpoints / f"model_update_{update:06d}.pth"
        if not ck.is_file():
            raise FileNotFoundError(ck)
        out = root / f"u{update:06d}"
        eval_backbone(args, ck, out, "AoA10")
        row = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        rows.append(row)

    selected_out = root / "selected_seen_best"
    eval_backbone(args, selected_backbone, selected_out, "AoA10_selected_by_seen_only")
    selected = json.loads((selected_out / "metrics.json").read_text(encoding="utf-8"))
    dump(root / "selected_metrics.json", selected)

    if rows:
        with (root / "curve.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def residual_aoa_curve(args: argparse.Namespace, backbone: Path) -> list[dict[str, object]]:
    root = args.out_root / "residual_aoa_curve"
    rows: list[dict[str, object]] = []
    checkpoints = args.out_root / "residual" / "checkpoints"
    for update in range(RESIDUAL_INTERVAL, RESIDUAL_UPDATES + 1, RESIDUAL_INTERVAL):
        ck = checkpoints / f"model_update_{update:06d}.pth"
        if not ck.is_file():
            raise FileNotFoundError(ck)
        out = root / f"u{update:06d}"
        eval_residual(args, backbone, ck, args.aoa_manifest, out, "RESIDUAL_AOA_AUDIT")
        metrics = json.loads((out / "final_primary_metrics.json").read_text(encoding="utf-8"))
        metrics["update"] = update
        rows.append(metrics)
    if rows:
        with (root / "curve.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def point_from_final(metrics: dict[str, object]) -> float:
    raw = {
        "rel_l2": float(metrics["rel_l2_raw"]),
        "tke": float(metrics["tke_raw"]),
        "mvpe": float(metrics["mvpe_raw"]),
    }
    return merge.point_score(raw)


def final_evidence(args: argparse.Namespace, backbone: Path, residual: Path) -> dict[str, object]:
    seen_out = args.out_root / "final_seen_selected"
    aoa_out = args.out_root / "final_aoa_selected"
    eval_residual(args, backbone, residual, args.manifest, seen_out, "FINAL_SEEN_EVAL")
    eval_residual(args, backbone, residual, args.aoa_manifest, aoa_out, "FINAL_AOA_EVAL")
    seen = json.loads((seen_out / "final_primary_metrics.json").read_text(encoding="utf-8"))
    aoa = json.loads((aoa_out / "final_primary_metrics.json").read_text(encoding="utf-8"))
    return {
        "seen": {**seen, "point_score": point_from_final(seen)},
        "aoa10": {**aoa, "point_score": point_from_final(aoa)},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--aoa-manifest", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--stage-a-checkpoint", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    p.add_argument("--retry-cooldown-seconds", type=int, default=30)
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    preflight = require_inputs(args)
    dump(args.out_root / "preflight.json", {
        "status": STATUS,
        "protocol": PROTOCOL,
        **preflight,
        "scientific_plan": {
            "stage_a_source_update": STAGE_A_UPDATE,
            "stage_b_updates": STAGE_B_UPDATES,
            "stage_b_eval_interval": STAGE_B_INTERVAL,
            "stage_b_optimizer_policy": "carry",
            "stage_b_selection": "Seen-Dev12 point score only",
            "residual_updates": RESIDUAL_UPDATES,
            "residual_eval_interval": RESIDUAL_INTERVAL,
            "residual_selection": "Seen-Dev12 point score only",
            "aoa10_role": "post-hoc audit only; never selects checkpoint",
            "sps": "not started",
            "full_data": "not started",
        },
        "hard_constraints": {
            "locked_final": "FORBIDDEN",
            "private_data": "FORBIDDEN",
            "codabench": "FORBIDDEN",
            "sps": "FORBIDDEN",
            "full_data": "FORBIDDEN",
            "scientific_hyperparameter_mutation_during_retry": "FORBIDDEN",
        },
    })

    heartbeat(args, phase="START", pid=os.getpid(), attempt=0)
    selected_backbone = run_stage_b(args)
    dump(args.out_root / "selected_stage_b.json", {
        "checkpoint": str(selected_backbone),
        "sha256": strong.sha256(selected_backbone),
        "selection_split": "Seen-Dev12",
        "aoa_used_for_selection": False,
    })

    stage_b_aoa_curve(args, selected_backbone)

    selected_residual = run_residual(args, selected_backbone)
    dump(args.out_root / "selected_residual.json", {
        "checkpoint": str(selected_residual),
        "sha256": strong.sha256(selected_residual),
        "selection_split": "Seen-Dev12",
        "aoa_used_for_selection": False,
    })

    residual_aoa_curve(args, selected_backbone)
    evidence = final_evidence(args, selected_backbone, selected_residual)

    stage_b_summary = json.loads((args.out_root / "stage_b" / "summary.json").read_text(encoding="utf-8"))
    residual_summary = json.loads((args.out_root / "residual" / "summary.json").read_text(encoding="utf-8"))
    result = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "state": "DONE",
        "stage_a_source": {
            "update": STAGE_A_UPDATE,
            "checkpoint": str(args.stage_a_checkpoint),
            "sha256": strong.sha256(args.stage_a_checkpoint),
        },
        "stage_b": {
            "best_update_seen": stage_b_summary.get("best_update"),
            "best_point_seen": stage_b_summary.get("best_point_score"),
            "selected_checkpoint": str(selected_backbone),
            "selected_sha256": strong.sha256(selected_backbone),
        },
        "residual": {
            "best_update_seen": residual_summary.get("best_update"),
            "best_point_seen": residual_summary.get("best_point_score"),
            "selected_checkpoint": str(selected_residual),
            "selected_sha256": strong.sha256(selected_residual),
        },
        "final_selected_evidence": evidence,
        "aoa10_used_for_selection": False,
        "locked_final_accessed": False,
        "private_accessed": False,
        "codabench_accessed": False,
        "sps_started": False,
        "full_data_started": False,
        "next_action": "REVIEW_REQUIRED: compare serial result with Joint V1; do not auto-start SPS/full/submission.",
    }
    dump(args.out_root / "FINAL_REPORT.json", result)
    dump(args.out_root / "status.json", {"status": STATUS, "state": "DONE"})
    heartbeat(args, phase="DONE", pid=os.getpid(), attempt=0, extra={
        "seen_point": evidence["seen"]["point_score"],
        "aoa10_point": evidence["aoa10"]["point_score"],
    })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
