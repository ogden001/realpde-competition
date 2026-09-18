#!/usr/bin/env python3
"""One-command, resumeable SOTA-V3 train/refit/package pipeline.

Stages:
  01 dev backbone + AoA
  02 dev frozen-backbone residual corrector
  03 dev teammate35 SPS on base features / final corrected center
  04 all-82 backbone refit using Dev-selected reference update
  05 all-82 residual refit using frozen mapped budget
  06 all-82 SPS head using Dev-selected update + bounds
  07 whitelist package
  08 clean-room real-fixture smoke

No Codabench submission or locked-final/private access is implemented.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = (
    "01_dev_backbone", "02_dev_residual", "03_dev_sps",
    "04_full_backbone", "05_full_residual", "06_full_sps",
    "07_package", "08_smoke",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _completed(stage_dir: Path) -> bool:
    status = stage_dir / "status.json"
    if status.is_file():
        try:
            return _load(status).get("state") == "DONE"
        except Exception:
            return False
    return False


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, check=True
        )


def _python(script: str, *args: object) -> list[str]:
    return [
        sys.executable, str(HERE / script), *(str(x) for x in args)
    ]


def _stage_state(run_root: Path, stage: str, state: str, **extra) -> None:
    _dump(run_root / "pipeline_state.json", {
        "state": state,
        "stage": stage,
        "stages": list(STAGES),
        **extra,
        "locked_final_accessed": False,
        "codabench_accessed": False,
    })


def run(args: argparse.Namespace) -> dict:
    args.run_root.mkdir(parents=True, exist_ok=True)
    _stage_state(
        args.run_root, STAGES[0], "RUNNING",
        execution_commit=args.execution_commit,
    )

    dev_backbone = args.run_root / STAGES[0]
    if not _completed(dev_backbone):
        _run(_python(
            "realpde_sota_v3_backbone.py",
            "--mode", "dev",
            "--manifest", args.manifest,
            "--data-root", args.data_root,
            "--checkpoint", args.warm_checkpoint,
            "--kit-root", args.kit_root,
            "--out-dir", dev_backbone,
            "--micro-batch", args.backbone_micro_batch,
            "--accumulation-steps", args.backbone_accumulation,
            "--workers", args.workers,
            "--require-cuda",
            "--resume",
        ), dev_backbone / "pipeline.log")
    selection = _load(dev_backbone / "selection.json")
    dev_backbone_ckpt = Path(selection["selected_checkpoint"])
    reference_update = int(selection["selected_reference_update"])

    dev_residual = args.run_root / STAGES[1]
    _stage_state(
        args.run_root, STAGES[1], "RUNNING",
        execution_commit=args.execution_commit,
    )
    if not _completed(dev_residual):
        _run(_python(
            "realpde_sota_v3_residual.py",
            "--mode", "dev",
            "--manifest", args.manifest,
            "--data-root", args.data_root,
            "--kit-root", args.kit_root,
            "--backbone-checkpoint", dev_backbone_ckpt,
            "--out-dir", dev_residual,
            "--workers", args.workers,
            "--require-cuda",
            "--resume",
        ), dev_residual / "pipeline.log")
    dev_residual_summary = _load(dev_residual / "summary.json")
    dev_corrector_ckpt = Path(
        dev_residual_summary["corrector_checkpoint"]
    )

    dev_sps = args.run_root / STAGES[2]
    _stage_state(
        args.run_root, STAGES[2], "RUNNING",
        execution_commit=args.execution_commit,
    )
    if not _completed(dev_sps):
        _run(_python(
            "realpde_sota_v3_sps.py",
            "--mode", "dev",
            "--manifest", args.manifest,
            "--data-root", args.data_root,
            "--kit-root", args.kit_root,
            "--backbone-checkpoint", dev_backbone_ckpt,
            "--corrector-checkpoint", dev_corrector_ckpt,
            "--out-dir", dev_sps,
            "--workers", args.workers,
            "--require-cuda",
            "--resume",
        ), dev_sps / "pipeline.log")
    dev_sps_summary = _load(dev_sps / "summary.json")
    selected_sps_updates = int(
        dev_sps_summary["selected_iteration"]
    )
    floor = float(dev_sps_summary["best"]["floor"])
    mult = float(dev_sps_summary["best"]["mult"])

    full_backbone = args.run_root / STAGES[3]
    _stage_state(
        args.run_root, STAGES[3], "RUNNING",
        execution_commit=args.execution_commit,
    )
    if not _completed(full_backbone):
        _run(_python(
            "realpde_sota_v3_backbone.py",
            "--mode", "full",
            "--data-root", args.data_root,
            "--checkpoint", args.warm_checkpoint,
            "--kit-root", args.kit_root,
            "--out-dir", full_backbone,
            "--reference-update", reference_update,
            "--micro-batch", args.backbone_micro_batch,
            "--accumulation-steps", args.backbone_accumulation,
            "--workers", args.workers,
            "--require-cuda",
            "--resume",
        ), full_backbone / "pipeline.log")
    full_backbone_summary = _load(
        full_backbone / "summary.json"
    )
    full_backbone_ckpt = Path(
        full_backbone_summary["checkpoint"]
    )

    full_residual = args.run_root / STAGES[4]
    _stage_state(
        args.run_root, STAGES[4], "RUNNING",
        execution_commit=args.execution_commit,
    )
    if not _completed(full_residual):
        _run(_python(
            "realpde_sota_v3_residual.py",
            "--mode", "full",
            "--data-root", args.data_root,
            "--kit-root", args.kit_root,
            "--backbone-checkpoint", full_backbone_ckpt,
            "--out-dir", full_residual,
            "--workers", args.workers,
            "--require-cuda",
            "--resume",
        ), full_residual / "pipeline.log")
    full_residual_summary = _load(
        full_residual / "summary.json"
    )
    full_corrector_ckpt = Path(
        full_residual_summary["corrector_checkpoint"]
    )

    full_sps = args.run_root / STAGES[5]
    _stage_state(
        args.run_root, STAGES[5], "RUNNING",
        execution_commit=args.execution_commit,
    )
    if not _completed(full_sps):
        _run(_python(
            "realpde_sota_v3_sps.py",
            "--mode", "full",
            "--data-root", args.data_root,
            "--kit-root", args.kit_root,
            "--backbone-checkpoint", full_backbone_ckpt,
            "--corrector-checkpoint", full_corrector_ckpt,
            "--out-dir", full_sps,
            "--selected-updates", selected_sps_updates,
            "--bound-floor", floor,
            "--bound-mult", mult,
            "--workers", args.workers,
            "--require-cuda",
            "--resume",
        ), full_sps / "pipeline.log")
    full_sps_summary = _load(full_sps / "summary.json")
    full_head_ckpt = Path(full_sps_summary["head"])

    package = args.run_root / STAGES[6]
    _stage_state(
        args.run_root, STAGES[6], "RUNNING",
        execution_commit=args.execution_commit,
    )
    package_status = package / "status.json"
    if not package_status.is_file():
        if package.exists():
            shutil.rmtree(package)
        _run(_python(
            "build_sota_v3_package.py",
            "--backbone-checkpoint", full_backbone_ckpt,
            "--corrector-checkpoint", full_corrector_ckpt,
            "--head-checkpoint", full_head_ckpt,
            "--kit-root", args.kit_root,
            "--out-root", package,
            "--execution-commit", args.execution_commit,
        ), args.run_root / "07_package.log")
        _dump(package_status, {
            "state": "DONE", "status": "REVIEW_REQUIRED"
        })
    package_report = _load(package / "package_build.json")

    smoke = args.run_root / STAGES[7]
    _stage_state(
        args.run_root, STAGES[7], "RUNNING",
        execution_commit=args.execution_commit,
    )
    smoke.mkdir(parents=True, exist_ok=True)
    if not _completed(smoke):
        _run(_python(
            "verify_sota_v3_package.py",
            "--zip", package / "submission.zip",
            "--backbone-checkpoint", full_backbone_ckpt,
            "--corrector-checkpoint", full_corrector_ckpt,
            "--kit-root", args.kit_root,
            "--fixture", args.fixture,
            "--out", smoke / "smoke_report.json",
        ), smoke / "pipeline.log")
        _dump(smoke / "status.json", {
            "state": "DONE", "status": "REVIEW_REQUIRED"
        })
    smoke_report = _load(smoke / "smoke_report.json")

    result = {
        "status": "REVIEW_REQUIRED",
        "execution_commit": args.execution_commit,
        "dev": {
            "reference_update": reference_update,
            "backbone_checkpoint": str(dev_backbone_ckpt),
            "corrector_checkpoint": str(dev_corrector_ckpt),
            "residual_evaluation": dev_residual_summary.get(
                "evaluation"
            ),
            "sps_selected_updates": selected_sps_updates,
            "sps_best": dev_sps_summary["best"],
        },
        "full": {
            "backbone_checkpoint": str(full_backbone_ckpt),
            "corrector_checkpoint": str(full_corrector_ckpt),
            "head_checkpoint": str(full_head_ckpt),
        },
        "package": package_report,
        "smoke": smoke_report,
        "locked_final_accessed": False,
        "codabench_accessed": False,
    }
    _dump(args.run_root / "summary.json", result)
    _stage_state(
        args.run_root, "DONE", "REVIEW_REQUIRED",
        execution_commit=args.execution_commit,
        summary=str(args.run_root / "summary.json"),
    )
    print(json.dumps(
        result, indent=2, sort_keys=True, default=str
    ))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True)
    p.add_argument("--warm-checkpoint", type=Path, required=True)
    p.add_argument(
        "--fixture", type=Path, required=True,
        help="released Train/Dev H5 only; never locked-final/private",
    )
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--execution-commit", required=True)
    p.add_argument("--backbone-micro-batch", type=int, default=4)
    p.add_argument("--backbone-accumulation", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
