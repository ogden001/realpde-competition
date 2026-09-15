#!/usr/bin/env python3
"""CPU-only matched summary and mechanical low-memory final gate."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

MILESTONES = (3000, 4500, 6000, 9000, 12000)
LATE = (6000, 9000, 12000)


def improvement(c0: float, v1: float) -> float:
    return (c0 - v1) / c0 * 100.0 if c0 else float("nan")


def compute_gate(rows: list[dict]) -> dict:
    late = [row for row in rows if int(row["update"]) in LATE]
    if len(late) != 3 or {int(row["update"]) for row in late} != set(LATE):
        raise ValueError("late gate requires exactly 6000, 9000, 12000")
    medians = {
        "median_rel_improvement": _median([float(row["rel_l2_improvement_pct"]) for row in late]),
        "median_tke_improvement": _median([float(row["tke_improvement_pct"]) for row in late]),
        "median_mvpe_improvement": _median([float(row["mvpe_improvement_pct"]) for row in late]),
    }
    median_checks = {
        "median_rel_improvement_ge_2pct": medians["median_rel_improvement"] >= 2.0,
        "median_mvpe_improvement_ge_4pct": medians["median_mvpe_improvement"] >= 4.0,
        "median_tke_improvement_ge_minus1pct": medians["median_tke_improvement"] >= -1.0,
    }
    checkpoint_checks = []
    for row in late:
        rel = float(row["rel_l2_improvement_pct"])
        mvpe = float(row["mvpe_improvement_pct"])
        tke = float(row["tke_improvement_pct"])
        checkpoint_checks.append({"update": int(row["update"]), "pass": rel > 0 and mvpe > 0 and tke >= -1,
                                  "rel_positive": rel > 0, "mvpe_positive": mvpe > 0,
                                  "tke_ge_minus1pct": tke >= -1})
    count_check = sum(item["pass"] for item in checkpoint_checks) >= 2
    final_gate = "MERGE_CANDIDATE" if all(median_checks.values()) and count_check else "PARK"
    return {"late_updates": list(LATE), "medians": medians, "median_checks": median_checks,
            "checkpoint_checks": checkpoint_checks, "at_least_two_late_checkpoints_pass": count_check,
            "FINAL_GATE": final_gate}


def _median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    return float(values[middle]) if len(values) % 2 else float((values[middle - 1] + values[middle]) / 2)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path) -> dict:
    c0 = {int(row["update"]): row for row in _read_rows(root / "C0-LOWMEM" / "aggregate_metrics.csv")}
    v1 = {int(row["update"]): row for row in _read_rows(root / "V1-LOWMEM" / "aggregate_metrics.csv")}
    if set(c0) != set(MILESTONES) or set(v1) != set(MILESTONES):
        raise ValueError("both arms must contain all five fixed milestones")
    matched = []
    for update in MILESTONES:
        c, v = c0[update], v1[update]
        c_rel, v_rel = float(c["rel_l2"]), float(v["rel_l2"])
        c_tke, v_tke = float(c["tke"]), float(v["tke"])
        c_mvpe, v_mvpe = float(c["mvpe"]), float(v["mvpe"])
        matched.append({"update": update, "c0_rel_l2": c_rel, "v1_rel_l2": v_rel,
                        "rel_l2_improvement_pct": improvement(c_rel, v_rel),
                        "c0_tke": c_tke, "v1_tke": v_tke,
                        "tke_improvement_pct": improvement(c_tke, v_tke),
                        "c0_mvpe": c_mvpe, "v1_mvpe": v_mvpe,
                        "mvpe_improvement_pct": improvement(c_mvpe, v_mvpe)})

    stability = []
    for update in MILESTONES:
        c_rows = {row["trajectory"]: row for row in _read_rows(root / "C0-LOWMEM" / f"eval_{update:05d}" / "trajectory_metrics.csv")}
        v_rows = {row["trajectory"]: row for row in _read_rows(root / "V1-LOWMEM" / f"eval_{update:05d}" / "trajectory_metrics.csv")}
        if set(c_rows) != set(v_rows) or len(c_rows) != 16:
            raise ValueError(f"trajectory pairing failed at {update}")
        rel_wins = tke_wins = mvpe_wins = all_three = protected = rel_mvpe = 0
        for name in sorted(c_rows):
            cr, vr = c_rows[name], v_rows[name]
            rel = float(vr["rel_l2"]) < float(cr["rel_l2"])
            tke = float(vr["tke"]) < float(cr["tke"])
            mvpe = float(vr["mvpe"]) < float(cr["mvpe"])
            tke_imp = improvement(float(cr["tke"]), float(vr["tke"]))
            rel_wins += rel; tke_wins += tke; mvpe_wins += mvpe
            all_three += rel and tke and mvpe
            rel_mvpe += rel and mvpe
            protected += rel and mvpe and tke_imp >= -2.0
        stability.append({"update": update, "rel_l2_wins_16": rel_wins, "tke_wins_16": tke_wins,
                          "mvpe_wins_16": mvpe_wins, "all_three_wins": all_three,
                          "rel_mvpe_wins": rel_mvpe,
                          "rel_mvpe_wins_tke_degradation_le_2pct": protected})

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write(root / "matched_metrics.csv", matched)
    _write(summary_dir / "matched_metrics.csv", matched)
    _write(root / "trajectory_stability.csv", stability)
    _write(summary_dir / "trajectory_stability.csv", stability)
    gate = compute_gate(matched)
    (root / "late_checkpoint_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    (summary_dir / "late_checkpoint_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")

    shutil.copy2(root / "C0-LOWMEM" / "training_curve.csv", root / "training_curve_c0.csv")
    shutil.copy2(root / "V1-LOWMEM" / "training_curve.csv", root / "training_curve_v1.csv")
    runtime = {"C0-LOWMEM": json.loads((root / "C0-LOWMEM" / "runtime.json").read_text()),
               "V1-LOWMEM": json.loads((root / "V1-LOWMEM" / "runtime.json").read_text())}
    metadata = {"C0-LOWMEM": json.loads((root / "C0-LOWMEM" / "run_metadata.json").read_text()),
                "V1-LOWMEM": json.loads((root / "V1-LOWMEM" / "run_metadata.json").read_text()),
                "gate": gate}
    (root / "runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")
    (root / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"matched": matched, "stability": stability, "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    print(json.dumps(result["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
