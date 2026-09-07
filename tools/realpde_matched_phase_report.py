#!/usr/bin/env python3
"""Build the RW-MA/RW-MB matched Random Phase review package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from realpde_random_phase_report import METRICS, relative_improvements


def classify_matched_phase_decision(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    """Classify only the unambiguous registered @3000 decision branches."""
    improvements = relative_improvements(baseline, candidate)
    degradation = {metric: -value for metric, value in improvements.items()}
    if degradation["rel_l2"] > 5.0 and degradation["mvpe"] > 5.0:
        candidate_status = "NO_GO_CANDIDATE"
        reason = "rel_l2_and_mvpe_degradation_exceed_5_percent"
    elif all(value >= 0.0 for value in improvements.values()) or any(value > 0.0 for value in improvements.values()):
        candidate_status = "NEED_LONG_7500"
        reason = "matched_result_is_flat_or_has_positive_improvement"
    else:
        candidate_status = "REVIEW_REQUIRED"
        reason = "tke_compensation_clause_is_not_numerically_preregistered"
    return {
        "candidate_status": candidate_status,
        "final_status": "REVIEW_REQUIRED",
        "reason": reason,
        "improvements_percent": improvements,
        "degradation_percent": degradation,
    }


def _history(run_dir: Path) -> dict[int, dict[str, float]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {int(row["iteration"]): {metric: float(row[metric]) for metric in METRICS} for row in summary["history"]}


def _trajectory_rows(run_dir: Path, update: int) -> dict[str, dict[str, float]]:
    with (run_dir / f"eval_{update:05d}" / "trajectory_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return {row["trajectory_id"]: {metric: float(row[metric]) for metric in METRICS} for row in csv.DictReader(handle)}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(ma_dir: Path, mb_dir: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ma, mb = _history(ma_dir), _history(mb_dir)
    updates = sorted(set(ma) & set(mb) - {0})
    comparison_rows = []
    trajectory_rows = []
    win_counts: dict[str, dict[str, int]] = {}
    for update in updates:
        improvements = relative_improvements(ma[update], mb[update])
        comparison_rows.append({
            "update": update,
            **{f"rw_ma_{metric}": ma[update][metric] for metric in METRICS},
            **{f"rw_mb_{metric}": mb[update][metric] for metric in METRICS},
            **{f"{metric}_improvement_percent": improvements[metric] for metric in METRICS},
        })
        ma_trajectories, mb_trajectories = _trajectory_rows(ma_dir, update), _trajectory_rows(mb_dir, update)
        per_update = {metric: 0 for metric in METRICS}
        for trajectory_id in sorted(set(ma_trajectories) & set(mb_trajectories)):
            left, right = ma_trajectories[trajectory_id], mb_trajectories[trajectory_id]
            row_improvements = relative_improvements(left, right)
            row = {
                "update": update,
                "trajectory_id": trajectory_id,
                **{f"rw_ma_{metric}": left[metric] for metric in METRICS},
                **{f"rw_mb_{metric}": right[metric] for metric in METRICS},
                **{f"{metric}_improvement_percent": row_improvements[metric] for metric in METRICS},
            }
            for metric in METRICS:
                row[f"{metric}_win"] = int(right[metric] < left[metric])
                per_update[metric] += row[f"{metric}_win"]
            trajectory_rows.append(row)
        win_counts[str(update)] = per_update
    _write_csv(out_dir / "comparison.csv", comparison_rows)
    _write_csv(out_dir / "trajectory_comparison.csv", trajectory_rows)
    formal_update = max(updates)
    ma_summary = json.loads((ma_dir / "summary.json").read_text(encoding="utf-8"))
    mb_summary = json.loads((mb_dir / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "rw_ma_dir": str(ma_dir),
        "rw_mb_dir": str(mb_dir),
        "updates": updates,
        "comparison": comparison_rows,
        "trajectory_win_counts": win_counts,
        "windows_per_epoch": {
            "rw_ma": ma_summary.get("window_audit_summary", {}).get("windows_per_epoch"),
            "rw_mb": mb_summary.get("window_audit_summary", {}).get("windows_per_epoch"),
        },
        "phase_audit": {
            "rw_ma": ma_summary.get("window_audit_summary", {}).get("phase_counts"),
            "rw_mb": mb_summary.get("window_audit_summary", {}).get("phase_counts"),
        },
        "early_gate": mb_summary.get("metadata", {}).get("early_gate"),
        "decision": classify_matched_phase_decision(ma[formal_update], mb[formal_update]),
        "formal_update": formal_update,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rw-ma-dir", type=Path, required=True)
    parser.add_argument("--rw-mb-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.rw_ma_dir, args.rw_mb_dir, args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
