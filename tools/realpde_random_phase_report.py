#!/usr/bin/env python3
"""Build the auditable RW-00/RW-01 comparison package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("rel_l2", "tke", "mvpe")


def relative_improvements(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    """Return positive percentages when the candidate error is lower."""
    return {key: round((float(baseline[key]) - float(candidate[key])) / max(float(baseline[key]), 1e-12) * 100.0, 6) for key in METRICS}


def build_comparison_rows(rw00: dict[str, dict[str, float]], rw01: dict[str, dict[str, float]]) -> list[dict[str, float | int]]:
    updates = sorted(set(rw00) & set(rw01), key=int)
    rows = []
    for update in updates:
        left, right = rw00[update], rw01[update]
        improvements = relative_improvements(left, right)
        rows.append({
            "update": int(update),
            "rw00_rel_l2": float(left["rel_l2"]),
            "rw00_tke": float(left["tke"]),
            "rw00_mvpe": float(left["mvpe"]),
            "rw01_rel_l2": float(right["rel_l2"]),
            "rw01_tke": float(right["tke"]),
            "rw01_mvpe": float(right["mvpe"]),
            "rel_l2_improvement_percent": improvements["rel_l2"],
            "tke_improvement_percent": improvements["tke"],
            "mvpe_improvement_percent": improvements["mvpe"],
        })
    return rows


def classify_formal_gate(baseline: dict[str, float], candidate: dict[str, float], *, rel_wins: int, mvpe_wins: int,
                        trajectory_count: int = 16) -> dict[str, object]:
    improvements = relative_improvements(baseline, candidate)
    tke_degradation = -improvements["tke"]
    win_rate = {"rel_l2": rel_wins / trajectory_count, "mvpe": mvpe_wins / trajectory_count}
    if tke_degradation > 5.0 or (improvements["rel_l2"] <= 0.0 and improvements["mvpe"] <= 0.0):
        status = "STOP"
    elif tke_degradation <= 2.0 and (
        improvements["rel_l2"] >= 2.0
        or improvements["mvpe"] >= 2.0
        or improvements["tke"] >= 2.0
        or (all(value > 0.0 for value in improvements.values()) and max(improvements.values()) >= 3.0)
    ):
        status = "GO"
    elif (1.0 <= improvements["rel_l2"] <= 2.0 and 1.0 <= improvements["mvpe"] <= 2.0
          and tke_degradation <= 2.0 and rel_wins > trajectory_count / 2 and mvpe_wins > trajectory_count / 2):
        status = "BORDERLINE_KEEP"
    else:
        status = "STOP"
    return {
        "status": status,
        "improvements_percent": improvements,
        "tke_degradation_percent": tke_degradation,
        "trajectory_win_rate": win_rate,
        "trajectory_wins": {"rel_l2": rel_wins, "mvpe": mvpe_wins},
    }


def _history(summary_path: Path) -> dict[str, dict[str, float]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        str(row["iteration"]): {metric: float(row[metric]) for metric in METRICS}
        for row in summary["history"]
    }


def _trajectory_rows(run_dir: Path, update: int) -> dict[str, dict[str, float]]:
    path = run_dir / f"eval_{update:05d}" / "trajectory_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["trajectory_id"]: {metric: float(row[metric]) for metric in METRICS}
            for row in csv.DictReader(handle)
        }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(rw00_dir: Path, rw01_dir: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rw00 = _history(rw00_dir / "summary.json")
    rw01 = _history(rw01_dir / "summary.json")
    comparison_rows = build_comparison_rows(rw00, rw01)
    _write_csv(out_dir / "comparison.csv", comparison_rows)
    common_updates = [row["update"] for row in comparison_rows]
    formal_update = 7500 if 7500 in common_updates else max(common_updates)
    rw00_traj = _trajectory_rows(rw00_dir, formal_update)
    rw01_traj = _trajectory_rows(rw01_dir, formal_update)
    trajectory_rows = []
    for trajectory in sorted(set(rw00_traj) & set(rw01_traj)):
        left, right = rw00_traj[trajectory], rw01_traj[trajectory]
        improvements = relative_improvements(left, right)
        trajectory_rows.append({
            "trajectory_id": trajectory,
            **{f"rw00_{metric}": left[metric] for metric in METRICS},
            **{f"rw01_{metric}": right[metric] for metric in METRICS},
            **{f"{metric}_improvement_percent": improvements[metric] for metric in METRICS},
            "rel_l2_win": int(right["rel_l2"] < left["rel_l2"]),
            "tke_win": int(right["tke"] < left["tke"]),
            "mvpe_win": int(right["mvpe"] < left["mvpe"]),
        })
    _write_csv(out_dir / "trajectory_comparison.csv", trajectory_rows)
    formal = next(row for row in comparison_rows if row["update"] == formal_update)
    formal_baseline = {metric: formal[f"rw00_{metric}"] for metric in METRICS}
    formal_candidate = {metric: formal[f"rw01_{metric}"] for metric in METRICS}
    win_counts = {metric: sum(int(row[f"{metric}_win"]) for row in trajectory_rows) for metric in METRICS}
    gate = classify_formal_gate(formal_baseline, formal_candidate, rel_wins=win_counts["rel_l2"],
                                mvpe_wins=win_counts["mvpe"], trajectory_count=len(trajectory_rows))
    milestones = {str(row["update"]): row for row in comparison_rows}
    trend = {
        "at_1000_approximately_baseline": abs(float(milestones.get("1000", {}).get("rel_l2_improvement_percent", 0.0))) <= 1.0,
        "at_3000_small_improvement": 0.0 < float(milestones.get("3000", {}).get("rel_l2_improvement_percent", 0.0)) <= 3.0,
        "at_5000_gap_expands": float(milestones.get("5000", {}).get("rel_l2_improvement_percent", 0.0)) > float(milestones.get("3000", {}).get("rel_l2_improvement_percent", 0.0)),
        "at_7500_clear_improvement": float(milestones.get("7500", {}).get("rel_l2_improvement_percent", 0.0)) >= 2.0,
    }
    summary = {
        "rw00_dir": str(rw00_dir),
        "rw01_dir": str(rw01_dir),
        "formal_update": formal_update,
        "metrics_at_3000": milestones.get("3000"),
        "metrics_at_7500": milestones.get("7500"),
        "trajectory_win_counts": win_counts,
        "gate": gate,
        "learning_curve_pattern": trend,
        "early_gate": json.loads((rw01_dir / "summary.json").read_text(encoding="utf-8")).get("metadata", {}).get("early_gate"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rw00-dir", type=Path, required=True)
    parser.add_argument("--rw01-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.rw00_dir, args.rw01_dir, args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
