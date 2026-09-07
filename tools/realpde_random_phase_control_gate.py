#!/usr/bin/env python3
"""Evaluate the pre-registered RW-CTRL equivalence gate at update 1000."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("rel_l2", "tke", "mvpe")


def control_gate(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    """Stop only for the registered matched-control deviation limits."""
    degradation = {
        metric: (float(candidate[metric]) / max(float(baseline[metric]), 1e-12) - 1.0) * 100.0
        for metric in METRICS
    }
    stop = degradation["rel_l2"] > 5.0 or degradation["mvpe"] > 5.0 or degradation["tke"] > 10.0
    return {
        "status": "STOP_REVIEW_REQUIRED" if stop else "CONTINUE",
        "degradation_percent": degradation,
        "thresholds_percent": {"rel_l2": 5.0, "mvpe": 5.0, "tke": 10.0},
    }


def _at_update(summary: Path, update: int) -> dict[str, float]:
    payload = json.loads(summary.read_text(encoding="utf-8"))
    row = next(item for item in payload["history"] if int(item["iteration"]) == update)
    return {metric: float(row[metric]) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--update", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = _at_update(args.reference_summary, args.update)
    candidate = _at_update(args.control_summary, args.update)
    result = {
        "update": args.update,
        "reference_metrics": baseline,
        "control_metrics": candidate,
        "gate": control_gate(baseline, candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["gate"]["status"] != "CONTINUE":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
