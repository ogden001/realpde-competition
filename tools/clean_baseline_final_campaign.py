#!/usr/bin/env python3
"""Single entrypoint for CLEAN_BASELINE_FINAL_CAMPAIGN.

The campaign contains exactly three frozen experiments:
  1. Strong Backbone Clean -> baseline residual, clean 51/12 split.
  2. Pareto-TKE Residual Clean -> matched scalar control + project-TKE arm.
  3. Residual-aware SPS on the frozen clean-baseline predictor.

No full-data refit, package build, locked-final/private access or Codabench path
exists in this runner.  All outputs remain REVIEW_REQUIRED for Sol review.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
COLLEAGUE = TOOLS / "colleague_80pt"
sys.path.insert(0, str(COLLEAGUE))
from run_clean_baseline_v1 import (  # noqa: E402
    SPLIT_MANIFEST, SIM_PRETRAIN_SHA256, training_manifest, holdout_eval_manifest,
    validate_split_payload, verify_split_against_data,
)
from run_incremental_screen import require_clean_main_checkout, require_gpu, sha256  # noqa: E402

STAGE1_SHA = "6aec4edb5e42746080e19baaafa2d6b719bb57befb1ff9035ece7889ec280036"
STAGE2_SHA = "1536fe02e11fc0dc991a38cb1489d0be0752ec8149be7b9f1542a4cb81c2f975"
EXP2_UPDATES = 22_000
BETAS = (0.5, 0.65, 0.8, 1.0)
BASELINE_22K = REPO / "docs/clean_baseline_v1/results/20260924_run1/evidence/stage2_residual/eval_step_22000.json"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}; log={log}")


def point_score(metrics: dict[str, float]) -> float:
    def s(x: float) -> float: return 100.0 / (1.0 + 0.5 * max(float(x), 0.0))
    return sum(s(metrics[k]) for k in ("rel_l2_raw", "tke_raw", "mvpe_raw")) / 3.0


def pct(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate - baseline) / max(abs(baseline), 1e-12)


def metrics_from_eval(path: Path) -> dict[str, float]:
    row = json.loads(path.read_text())[0]
    return {"rel_l2_raw": float(row["rel_l2"]), "tke_raw": float(row["tke"]), "mvpe_raw": float(row["mvpe"])}


def metrics_from_evidence(path: Path) -> dict[str, float]:
    row = json.loads((path / "final_primary_metrics.json").read_text())
    return {k: float(row[k]) for k in ("rel_l2_raw", "tke_raw", "mvpe_raw")}


def relative_gate(candidate: dict[str, float], baseline: dict[str, float], *, kind: str) -> dict[str, object]:
    d = {k: pct(candidate[k], baseline[k]) for k in baseline}
    avg = sum(d.values()) / 3.0
    if kind == "strong":
        checks = {"mean_error_improvement_ge_1pct": avg <= -1.0,
                  "at_least_two_metrics_improve": sum(v < 0 for v in d.values()) >= 2,
                  "no_metric_degrades_gt_1pct": max(d.values()) <= 1.0,
                  "tke_not_worse": d["tke_raw"] <= 0.0}
    elif kind == "pareto":
        checks = {"tke_improvement_ge_2pct": d["tke_raw"] <= -2.0,
                  "rel_degradation_le_0p5pct": d["rel_l2_raw"] <= 0.5,
                  "mvpe_degradation_le_0p5pct": d["mvpe_raw"] <= 0.5,
                  "mean_error_not_worse": avg <= 0.0}
    else: raise ValueError(kind)
    return {"status": "GO" if all(checks.values()) else "NO_GO", "delta_pct": d,
            "mean_delta_pct": avg, "checks": checks}


def prepare(args) -> tuple[Path, Path]:
    args.out_root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(SPLIT_MANIFEST.read_text())
    validate_split_payload(payload)
    audit = verify_split_against_data(args.real_root, payload)
    train_manifest = args.out_root / "train_dev_manifest.json"
    hold_manifest = args.out_root / "holdout_eval_manifest.json"
    dump(train_manifest, training_manifest(payload)); dump(hold_manifest, holdout_eval_manifest(payload))
    dump(args.out_root / "preflight.json", {"status": "REVIEW_REQUIRED", "execution_commit": require_clean_main_checkout(REPO),
        "gpu": require_gpu(), "split_audit": audit, "sim_pretrain_sha256": sha256(args.sim_pretrain),
        "clean_stage1_sha256": sha256(args.clean_stage1), "clean_stage2_sha256": sha256(args.clean_stage2),
        "codabench_accessed": False, "locked_final_accessed": False})
    if sha256(args.sim_pretrain) != SIM_PRETRAIN_SHA256: raise RuntimeError("sim pretrain SHA mismatch")
    if sha256(args.clean_stage1) != STAGE1_SHA: raise RuntimeError("clean Stage1 best SHA mismatch")
    if sha256(args.clean_stage2) != STAGE2_SHA: raise RuntimeError("clean Stage2 best SHA mismatch")
    return train_manifest, hold_manifest


def residual_cmd(args, manifest: Path, checkpoint: Path, out: Path, *, base_model: str, tke: float, gradient: str) -> list[str]:
    model_root = args.kit_root if base_model == "sota_v2_mf" else args.model_root
    return [sys.executable, "-u", "-B", str(COLLEAGUE / "residual_multi.py"), "--real-root", str(args.real_root),
        "--split-manifest", str(manifest), "--checkpoint", str(checkpoint), "--realpdebench-root", str(model_root),
        "--base-model", base_model, "--out-dir", str(out), "--updates", str(EXP2_UPDATES), "--eval-interval", "1000",
        "--batch-size", "8", "--test-batch-size", "32", "--workers", str(args.workers), "--preload-to-ram",
        "--prefetch-factor", "4", "--lr", "0.0002", "--weight-decay", "0.00001", "--hidden", "96", "--blocks", "2",
        "--dropout", "0", "--max-delta", "0.04", "--stride", "1", "--eval-stride", "20", "--train-window-mode", "fixed",
        "--train-alpha", "1.0", "--eval-alphas", "1.0", "--point", "1.0", "--mse", "0.05", "--tke", str(tke),
        "--temporal", "0.03", "--grad", "0.015", "--p-zero", "0.01", "--residual-mse", "0.25", "--delta-penalty", "0.02",
        "--clip-grad", "1.0", "--selection-metric", "point_score", "--gradient-mode", gradient, "--seed", "41"]


def eval_cmd(args, manifest: Path, backbone: Path, residual: Path, out: Path, base_model: str) -> list[str]:
    model_root = args.kit_root if base_model == "sota_v2_mf" else args.model_root
    return [sys.executable, "-u", "-B", str(TOOLS / "evaluate_clean_residual_checkpoint.py"), "--real-root", str(args.real_root),
            "--split-manifest", str(manifest), "--backbone-checkpoint", str(backbone), "--residual-checkpoint", str(residual),
            "--model-root", str(model_root), "--base-model", base_model, "--out-dir", str(out), "--workers", str(args.workers), "--require-cuda"]


def exp1(args) -> None:
    train_manifest = args.out_root / "train_dev_manifest.json"; hold = args.out_root / "holdout_eval_manifest.json"
    root = args.out_root / "exp1_strong_backbone_clean"; root.mkdir(parents=True, exist_ok=False)
    strong = root / "strong_backbone"
    run([sys.executable, "-u", "-B", str(TOOLS / "train_clean_strong_backbone.py"), "--manifest", str(train_manifest),
         "--data-root", str(args.real_root), "--init-checkpoint", str(args.sim_pretrain), "--kit-root", str(args.kit_root),
         "--out-dir", str(strong), "--workers", str(args.workers), "--require-cuda"], root / "strong_backbone.log")
    backbone = strong / "checkpoints/model_best.pth"; residual = root / "residual_22k"
    run(residual_cmd(args, train_manifest, backbone, residual, base_model="sota_v2_mf", tke=0.06, gradient="scalar"), root / "residual_22k.log")
    seen = root / "seen_dev_selected"; run(eval_cmd(args, train_manifest, backbone, residual / "model_best.pth", seen, "sota_v2_mf"), root / "seen_eval.log")
    candidate = metrics_from_evidence(seen); baseline = metrics_from_eval(BASELINE_22K); gate = relative_gate(candidate, baseline, kind="strong")
    summary = {"status": "REVIEW_REQUIRED", "baseline_22k": baseline, "candidate": candidate, "gate": gate,
               "strong_backbone_sha256": sha256(backbone), "residual_best_sha256": sha256(residual / "model_best.pth"),
               "holdout_accessed": False, "codabench_accessed": False}
    if gate["status"] == "GO":
        hout = root / "holdout_aoa10_selected"; run(eval_cmd(args, hold, backbone, residual / "model_best.pth", hout, "sota_v2_mf"), root / "holdout_eval.log")
        summary["holdout"] = metrics_from_evidence(hout); summary["holdout_accessed"] = True
    dump(root / "summary.json", summary); (root / "DONE").touch()


def _state_parity(a: Path, b: Path) -> float:
    pa = torch.load(a, map_location="cpu", weights_only=False)["model_state_dict"]; pb = torch.load(b, map_location="cpu", weights_only=False)["model_state_dict"]
    if set(pa) != set(pb): raise RuntimeError("init state keys differ")
    worst = 0.0
    for k in pa:
        if torch.is_tensor(pa[k]) and pa[k].is_floating_point(): worst = max(worst, float((pa[k]-pb[k]).abs().max()))
        elif not torch.equal(pa[k], pb[k]): raise RuntimeError(f"nonfloat init differs: {k}")
    return worst


def _blend(control: Path, pareto: Path, beta: float, out: Path) -> None:
    c = torch.load(control, map_location="cpu", weights_only=False); p = torch.load(pareto, map_location="cpu", weights_only=False)
    cs, ps = c["model_state_dict"], p["model_state_dict"]; state = {}
    for k in cs:
        if k not in ps or cs[k].shape != ps[k].shape: raise RuntimeError(f"interpolation mismatch {k}")
        if cs[k].is_floating_point(): state[k] = (1.0-beta)*cs[k] + beta*ps[k]
        else:
            if not torch.equal(cs[k], ps[k]): raise RuntimeError(f"nonfloat state mismatch {k}")
            state[k] = cs[k]
    out_payload = dict(c); out_payload["model_state_dict"] = state; out_payload["interpolation_beta"] = beta; torch.save(out_payload, out)


def exp2(args) -> None:
    train_manifest = args.out_root / "train_dev_manifest.json"; hold = args.out_root / "holdout_eval_manifest.json"
    root = args.out_root / "exp2_pareto_tke_clean"; root.mkdir(parents=True, exist_ok=False)
    control, pareto = root / "control_22k", root / "pareto_22k"
    run(residual_cmd(args, train_manifest, args.clean_stage1, control, base_model="cno", tke=0.06, gradient="scalar"), root / "control.log")
    run(residual_cmd(args, train_manifest, args.clean_stage1, pareto, base_model="cno", tke=0.12, gradient="project_tke"), root / "pareto.log")
    init_diff = _state_parity(control / "model_init.pth", pareto / "model_init.pth")
    if init_diff != 0.0: raise RuntimeError(f"control/pareto init parity failed: {init_diff}")
    if sha256(control / "window_audit.jsonl") != sha256(pareto / "window_audit.jsonl"): raise RuntimeError("control/pareto sampling audit differs")
    control_eval, pareto_eval = root / "seen_control", root / "seen_pareto"
    run(eval_cmd(args, train_manifest, args.clean_stage1, control / "model_best.pth", control_eval, "cno"), root / "seen_control.log")
    run(eval_cmd(args, train_manifest, args.clean_stage1, pareto / "model_best.pth", pareto_eval, "cno"), root / "seen_pareto.log")
    baseline = metrics_from_evidence(control_eval); rows = []; inter_dir = root / "interpolation"; inter_dir.mkdir()
    for beta in BETAS:
        ck = inter_dir / f"beta_{beta:.2f}.pth"; _blend(control / "model_best.pth", pareto / "model_best.pth", beta, ck)
        evdir = inter_dir / f"beta_{beta:.2f}_seen"; run(eval_cmd(args, train_manifest, args.clean_stage1, ck, evdir, "cno"), root / f"beta_{beta:.2f}.log")
        m = metrics_from_evidence(evdir); gate = relative_gate(m, baseline, kind="pareto"); rows.append({"beta": beta, **m, "point_score": point_score(m), "gate": gate})
    passing = [r for r in rows if r["gate"]["status"] == "GO"]; selected = max(passing, key=lambda r: r["point_score"]) if passing else None
    summary = {"status": "REVIEW_REQUIRED", "init_max_abs_diff": init_diff, "sampling_audit_equal": True, "control": baseline,
               "pareto_uninterpolated": metrics_from_evidence(pareto_eval), "interpolation": rows, "selected": selected,
               "holdout_accessed": False, "codabench_accessed": False}
    if selected is not None:
        beta = float(selected["beta"]); ck = inter_dir / f"beta_{beta:.2f}.pth"; hout = root / "holdout_aoa10_selected"
        run(eval_cmd(args, hold, args.clean_stage1, ck, hout, "cno"), root / "holdout_eval.log"); summary["holdout"] = metrics_from_evidence(hout)
        summary["holdout_accessed"] = True; summary["selected_checkpoint"] = str(ck); summary["selected_checkpoint_sha256"] = sha256(ck)
    dump(root / "summary.json", summary); (root / "DONE").touch()


def exp3(args) -> None:
    root = args.out_root / "exp3_residual_aware_sps"
    run([sys.executable, "-u", "-B", str(TOOLS / "train_clean_residual_aware_sps.py"), "--real-root", str(args.real_root),
         "--train-manifest", str(args.out_root / "train_dev_manifest.json"), "--holdout-manifest", str(args.out_root / "holdout_eval_manifest.json"),
         "--residual-checkpoint", str(args.clean_stage2), "--model-root", str(args.model_root), "--out-dir", str(root),
         "--workers", str(args.workers), "--require-cuda"], args.out_root / "exp3_residual_aware_sps.log")


def review(args) -> None:
    s1 = json.loads((args.out_root / "exp1_strong_backbone_clean/summary.json").read_text()); s2 = json.loads((args.out_root / "exp2_pareto_tke_clean/summary.json").read_text()); s3 = json.loads((args.out_root / "exp3_residual_aware_sps/summary.json").read_text())
    lines = ["# CLEAN_BASELINE_FINAL_CAMPAIGN - Execution Review", "", "Status: `REVIEW_REQUIRED`", "", "Codex produced evidence only. Sol must make the merge decision.", "",
             "## Experiment 1 - Strong Backbone Clean", "", f"Gate: **{s1['gate']['status']}**", f"Deltas: `{s1['gate']['delta_pct']}`", "",
             "## Experiment 2 - Pareto-TKE Residual Clean", "", f"Selected: `{s2.get('selected')}`", "", "## Experiment 3 - Residual-aware SPS", "",
             f"Gate: **{s3['gate']['status']}**", f"Dev SPS gain vs static: `{s3['gate']['dev_sps_gain_vs_static']:.4f}`", "",
             "## Safety", "", "- locked-final/private: NOT ACCESSED", "- Codabench: NOT ACCESSED", "- full-data refit: NOT STARTED", "- submission packaging: NOT STARTED", ""]
    (args.out_root / "CAMPAIGN_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def archive(args) -> None:
    run([sys.executable, "-u", "-B", str(COLLEAGUE / "archive_v2_campaign.py"), "--run-root", str(args.out_root), "--dest", str(args.archive_dest)], args.out_root / "archive.log")


def common(p):
    p.add_argument("--real-root", type=Path, required=True); p.add_argument("--sim-pretrain", type=Path, required=True); p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--kit-root", type=Path, required=True); p.add_argument("--clean-stage1", type=Path, required=True); p.add_argument("--clean-stage2", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True); p.add_argument("--workers", type=int, default=4)


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "exp1", "exp2", "exp3", "review"):
        p = sub.add_parser(name); common(p)
    p = sub.add_parser("archive"); common(p); p.add_argument("--archive-dest", type=Path, required=True)
    p = sub.add_parser("run-all"); common(p); p.add_argument("--archive-dest", type=Path, default=None)
    args = ap.parse_args()
    if args.cmd == "preflight": prepare(args); return
    if args.cmd == "run-all":
        if args.out_root.exists(): raise FileExistsError(args.out_root)
        prepare(args); exp1(args); exp2(args); exp3(args); review(args)
        if args.archive_dest is not None: archive(args)
        return
    if not args.out_root.exists(): raise FileNotFoundError("run preflight first")
    {"exp1": exp1, "exp2": exp2, "exp3": exp3, "review": review, "archive": archive}[args.cmd](args)


if __name__ == "__main__": main()
