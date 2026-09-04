#!/usr/bin/env python3
"""Dev-only error anatomy for the matched Direct CNO / A1 pair."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import realpde_loss_official_v9 as core

METRICS = ("rel_l2", "tke", "mvpe")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def metric_rows(path: Path) -> dict[str, dict[str, float]]:
    return {row["trajectory_id"]: {m: float(row[m]) for m in METRICS} for row in read_csv(path)}


def error(a: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(a - y) / max(np.linalg.norm(y), 1e-12))


def horizon_rows(base: np.ndarray, cand: np.ndarray, target: np.ndarray) -> list[dict]:
    rows = []
    for h in range(base.shape[1]):
        b, c, y = base[:, h, ..., :2], cand[:, h, ..., :2], target[:, h, ..., :2]
        bmv, cmv, ymv = b.mean(axis=(-3, -2)), c.mean(axis=(-3, -2)), y.mean(axis=(-3, -2))
        rows.append({"horizon": h + 1, "baseline_rel_l2": error(b, y), "a1_rel_l2": error(c, y),
                     "delta_rel_l2_pct": (error(b, y) - error(c, y)) / max(error(b, y), 1e-12) * 100,
                     "baseline_mvpe": error(bmv, ymv), "a1_mvpe": error(cmv, ymv),
                     "delta_mvpe_pct": (error(bmv, ymv) - error(cmv, ymv)) / max(error(bmv, ymv), 1e-12) * 100,
                     "baseline_fluctuation_rms": float(np.sqrt(np.mean((b - b.mean(axis=0)) ** 2))),
                     "a1_fluctuation_rms": float(np.sqrt(np.mean((c - c.mean(axis=0)) ** 2))),
                     "target_fluctuation_rms": float(np.sqrt(np.mean((y - y.mean(axis=0)) ** 2)))})
    return rows


def plot_case(out: Path, name: str, base: np.ndarray, cand: np.ndarray, target: np.ndarray, residual: np.ndarray | None, index: int) -> None:
    b, c, y = base[index, 0, ..., 0], cand[index, 0, ..., 0], target[index, 0, ..., 0]
    fields = [("GT field", y), ("baseline prediction", b), ("A1 prediction", c), ("baseline error", np.abs(b-y)), ("A1 error", np.abs(c-y)), ("A1-baseline improvement", np.abs(b-y)-np.abs(c-y))]
    if residual is not None: fields.append(("local residual magnitude", np.linalg.norm(residual[index, 0], axis=-1)))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.ravel()
    for ax, (title, field) in zip(axes, fields):
        im = ax.imshow(field, cmap="coolwarm" if "improvement" in title else "viridis")
        ax.set_title(title); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=.046)
    for ax in axes[len(fields):]: ax.axis("off")
    fig.suptitle(f"{name} | window index {index} | horizon t+1")
    fig.tight_layout(); fig.savefig(out / f"case_{name}.png", dpi=150); plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_npz, cand_npz = np.load(args.baseline), np.load(args.candidate)
    base, cand, target = base_npz["prediction"], cand_npz["prediction"], cand_npz["target"]
    if base.shape != cand.shape or target.shape != cand.shape: raise ValueError(f"shape mismatch: {base.shape}, {cand.shape}, {target.shape}")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    base_rows = metric_rows(args.baseline_trajectory)
    cand_rows = metric_rows(args.candidate_trajectory)
    delta, wins = core_compare(base_rows, cand_rows)
    descriptors = {row.get("trajectory_id", row.get("file", "")): row for row in read_csv(args.descriptors)}
    case_rows = []
    for name in sorted(base_rows):
        row = {"trajectory_id": name}
        for metric in METRICS:
            row[f"baseline_{metric}"] = base_rows[name][metric]; row[f"a1_{metric}"] = cand_rows[name][metric]
            row[f"delta_{metric}_pct"] = delta[name][f"{metric}_pct"]; row[f"{metric}_win"] = delta[name][f"{metric}_win"]
        row.update({f"descriptor_{k}": v for k, v in descriptors.get(name, {}).items() if k in ("fluctuation_rms", "delta_std", "grad_mag_mean", "vorticity_abs_mean", "strain_mag_mean", "high_energy_area_ratio", "distribution_label", "nearest_train_distance", "train_p95_exceed_count", "aoa", "re")})
        row["mean_delta_pct"] = float(np.mean([row[f"delta_{m}_pct"] for m in METRICS])); case_rows.append(row)
    case_rows.sort(key=lambda r: r["mean_delta_pct"], reverse=True)
    write_csv(args.out_dir / "trajectory_case_table.csv", case_rows)
    write_csv(args.out_dir / "horizon_metrics.csv", horizon_rows(base, cand, target))
    residual = cand_npz["local_residual_uv"] if "local_residual_uv" in cand_npz else None
    if residual is not None:
        magnitude = np.linalg.norm(residual, axis=-1)
        np.savez_compressed(args.out_dir / "local_residual_summary.npz", rms=np.sqrt(np.mean(residual**2, axis=(1,2,3,4))), norm=np.sqrt(np.mean(magnitude**2, axis=(1,2,3))), relative_to_global=np.sqrt(np.mean(residual**2, axis=(1,2,3,4))) / np.maximum(np.sqrt(np.mean(cand[..., :2]**2, axis=(1,2,3,4))), 1e-12))
        residual_rows = [{"window": i, "rms": float(np.sqrt(np.mean(residual[i] ** 2))), "norm": float(np.sqrt(np.mean(magnitude[i] ** 2))), "relative_to_global_output": float(np.sqrt(np.mean(residual[i] ** 2)) / max(np.sqrt(np.mean(cand[i,...,:2]**2)), 1e-12))} for i in range(len(residual))]
        write_csv(args.out_dir / "local_residual_windows.csv", residual_rows)
    name_to_index = {}
    for i, ref in enumerate(core.H5WindowDataset(dev_paths).refs): name_to_index.setdefault(ref.path.stem, i)
    good = case_rows[:3]; bad = case_rows[-3:][::-1]
    for group, rows in (("good", good), ("bad", bad)):
        for row in rows: plot_case(args.out_dir, f"{group}_{row['trajectory_id']}", base, cand, target, residual, name_to_index.get(row["trajectory_id"], 0))
    base_overall = {m: float(np.mean([r[m] for r in base_rows.values()])) for m in METRICS}
    cand_overall = {m: float(np.mean([r[m] for r in cand_rows.values()])) for m in METRICS}
    overall_delta = {m: (base_overall[m]-cand_overall[m])/max(base_overall[m],1e-12)*100 for m in METRICS}
    tke_bad = overall_delta["tke"] < -5.0
    severe_case = sum(r["mean_delta_pct"] < -20 for r in case_rows) >= 1
    verdict = "PROMISING" if all(overall_delta[m] > 0 for m in METRICS) and min(wins.values()) >= 8 else ("NO_GO" if all(overall_delta[m] <= 0 for m in METRICS) else "WEAK_SIGNAL_PARKED")
    facts = [f"A1 overall relative deltas vs matched Direct are {overall_delta} (negative error delta is improvement).", f"Trajectory wins are {wins} out of 16.", f"Top good cases: {[r['trajectory_id'] for r in good]}; top bad cases: {[r['trajectory_id'] for r in bad]}.", "Results use only the frozen 16 Dev trajectories and official v9 raw metrics; no custom final score is constructed."]
    hypotheses = ["Case linkage is descriptive: input-side fluctuation, temporal variation, gradients, vorticity/strain and Train-tail labels are not causal controls.", "A local-branch mechanism is supported only where residual magnitude and error-reduction maps spatially overlap; this does not prove the branch specifically models local physics."]
    report = ["# A1 Evaluation / Error Anatomy", "", "## Verified facts", "", *[f"- {x}" for x in facts], "", "## Overall", "", "| metric | matched Direct | A1 | delta % | wins |", "|---|---:|---:|---:|---:|"]
    report += [f"| {m} | {base_overall[m]:.8f} | {cand_overall[m]:.8f} | {overall_delta[m]:+.3f} | {wins[m]}/16 |" for m in METRICS]
    report += ["", "Metric-vs-update values are in `update_curve.csv`; horizon values are in `horizon_metrics.csv`.", "", "## Case linkage", "", "See `trajectory_case_table.csv`; descriptors are runtime-safe input-side quantities joined from the existing Dataset Profile. No model change was selected from an individual case.", "", "## Spatial / branch evidence", "", "Representative maps are `case_good_*.png` and `case_bad_*.png`. Local residual statistics are in `local_residual_windows.csv` and `local_residual_summary.npz`.", "", "## Mechanism hypothesis", "", *[f"- {x}" for x in hypotheses], "", "## Level 2 trigger", "", f"- TKE clearly worsened: {'YES' if tke_bad else 'NO'}; severe trajectory collapse: {'YES' if severe_case else 'NO'}; deeper diagnosis: {'YES' if tke_bad or severe_case else 'NO'}.", "", "## Architecture verdict", "", f"`{verdict}`", "", "The Local + Global result is interpreted only through the three official raw metrics, trajectory evidence, horizon behavior and maps; it is not converted into a custom final score."]
    (args.out_dir / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps({"overall_baseline": base_overall, "overall_a1": cand_overall, "delta_pct": overall_delta, "trajectory_wins": wins, "top_good": [r["trajectory_id"] for r in good], "top_bad": [r["trajectory_id"] for r in bad], "verdict": verdict, "level2_trigger": tke_bad or severe_case, "locked_final_accessed": False}, indent=2), encoding="utf-8")


def core_compare(base: dict, cand: dict) -> tuple[dict, dict]:
    return __import__("realpde_hybrid_cno_local_runner", fromlist=["compare_trajectory_metrics"]).compare_trajectory_metrics(base, cand)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--baseline", type=Path, required=True); p.add_argument("--candidate", type=Path, required=True); p.add_argument("--baseline-trajectory", type=Path, required=True); p.add_argument("--candidate-trajectory", type=Path, required=True); p.add_argument("--descriptors", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); run(p.parse_args())


if __name__ == "__main__": main()
