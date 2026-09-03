#!/usr/bin/env python3
"""Build a compact, reproducible review package for the bounded loss study.

The script consumes only exported experiment metadata/trajectory CSVs.  It does
not read the private training H5 files, model checkpoints, or prediction arrays.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("E0", "E1", "E2")
NAMES = {
    "E0": "E0: MSE + 0.05 TKE",
    "E1": "E1: Rel-L2 + 0.05 TKE",
    "E2": "E2: E1 + 0.10 MVPE",
}


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p))


def bootstrap(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    # diff is a percentage improvement (positive is better), resampled by trajectory.
    samples = rng.choice(diff, size=(1000, len(diff)), replace=True).mean(axis=1)
    return float(diff.mean()), percentile(samples, 2.5), percentile(samples, 97.5)


def parse_trajectory(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as f:
        return {
            r["trajectory_id"]: {k: float(r[k]) for k in ("rel_l2", "tke", "mvpe")}
            for r in csv.DictReader(f)
        }


def fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-doc", type=Path)
    args = parser.parse_args()
    raw, out = args.raw_root.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    figs, configs, source = out / "figures", out / "configs", out / "source"
    figs.mkdir(exist_ok=True); configs.mkdir(exist_ok=True); source.mkdir(exist_ok=True)

    # Keep direct evidence and replay source in human-readable form, excluding data/checkpoints.
    for name in ("screen.log", "final_eval.log"):
        shutil.copy2(raw / name, out / f"logs_{name}")
    shutil.copy2(raw / "manifests" / "id_seed20260901.json", configs / "id_seed20260901.json")
    shutil.copy2(raw / "manifests" / "ood_aoa20_seed20260901.json", configs / "ood_aoa20_seed20260901.json")
    for path in (raw / "source").glob("*"):
        shutil.copy2(path, source / path.name)
    evidence = out / "evidence"
    for path in raw.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".log"}:
            destination = evidence / path.relative_to(raw)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    if args.task_doc:
        shutil.copy2(args.task_doc.resolve(), out / "TASK_DOCUMENT.md")

    screen = {v: load_json(raw / f"screen_{v}_s20260901" / "summary.json") for v in VARIANTS}
    final = {v: load_json(raw / f"final_screen_{v}_s20260901" / "summary.json") for v in VARIANTS}
    audit = load_json(raw / "audit_e0_dev" / "summary.json")
    evaluation = {v: final[v]["evaluation"] for v in VARIANTS}
    metadata = screen["E0"]["metadata"]

    # Public review metrics: dev screening and a final locked audit.  The latter was
    # not used to choose a checkpoint (each selected checkpoint was step 300).
    metric_rows = []
    for split, bundle in (("id_dev_screen", screen), ("id_locked_final", final)):
        for v in VARIANTS:
            item = bundle[v]["history"][-1] if split == "id_dev_screen" else evaluation[v]["raw_errors"]
            scores = (None if split == "id_dev_screen" else evaluation[v]["official_v9_subscores"])
            metric_rows.append({
                "split": split, "variant": v, "loss": NAMES[v],
                "rel_l2": item["rel_l2"], "tke": item["tke"], "mvpe": item["mvpe"],
                "rel_l2_score": item.get("dev_rel_score") if scores is None else scores["rel_l2_score"],
                "tke_score": item.get("dev_tke_score") if scores is None else scores["tke_score"],
                "mvpe_score": item.get("dev_mvpe_score") if scores is None else scores["mvpe_score"],
                "time_score": "" if scores is None else scores["time_score"],
                "sps_score": "" if scores is None else scores["sps_score"],
                "mean_t_neural_s": "" if split == "id_dev_screen" else evaluation[v]["mean_t_neural_s"],
            })
    metric_fields = list(metric_rows[0])
    write_csv(out / "metrics.csv", metric_rows, metric_fields)

    # Paired trajectory calculations use exactly the 16 locked final trajectories.
    trajectories = {v: parse_trajectory(raw / f"final_screen_{v}_s20260901" / "trajectory_metrics.csv") for v in VARIANTS}
    ids = sorted(set.intersection(*(set(t) for t in trajectories.values())))
    all_rows = []
    paired_rows = []
    rng = np.random.default_rng(20260901)
    for v in VARIANTS:
        for tid in ids:
            all_rows.append({"split": "id_locked_final", "variant": v, "trajectory_id": tid, **trajectories[v][tid]})
    for v in ("E1", "E2"):
        for metric in ("rel_l2", "tke", "mvpe"):
            base = np.asarray([trajectories["E0"][tid][metric] for tid in ids])
            cand = np.asarray([trajectories[v][tid][metric] for tid in ids])
            improvement = 100 * (base - cand) / base
            mean, low, high = bootstrap(improvement, rng)
            paired_rows.append({
                "candidate": v, "metric": metric, "trajectories": len(ids),
                "mean_improvement_pct": mean, "median_improvement_pct": float(np.median(improvement)),
                "p25_improvement_pct": percentile(improvement, 25), "p75_improvement_pct": percentile(improvement, 75),
                "win_rate": float(np.mean(improvement > 0)), "bootstrap_95_low_pct": low,
                "bootstrap_95_high_pct": high,
            })
    write_csv(out / "trajectory_metrics_locked_final.csv", all_rows, list(all_rows[0]))
    write_csv(out / "paired_trajectory_statistics.csv", paired_rows, list(paired_rows[0]))

    grad_rows = list(csv.DictReader((raw / "audit_e0_dev" / "gradient_audit.csv").open()))
    grouped = {}
    for r in grad_rows:
        name = r["loss_name"]
        grouped.setdefault(name, []).append(float(r["weighted_gradient_norm"]))
    gradient_summary = [
        {"loss": k, "mean_weighted_gradient_norm": float(np.mean(v)), "median_weighted_gradient_norm": float(np.median(v)), "batches": len(v)}
        for k, v in grouped.items()
    ]
    write_csv(out / "gradient_audit_summary.csv", gradient_summary, list(gradient_summary[0]))

    anatomy = audit["evaluation"]["trajectory_anatomy"]
    anatomy_rows = []
    for key, vals in anatomy.items():
        anatomy_rows.append({"quantity": key, **vals})
    write_csv(out / "tke_diagnostics.csv", anatomy_rows, list(anatomy_rows[0]))

    # Static execution manifest makes every omitted phase explicit.
    def duration(meta: dict) -> float | str:
        return round(meta["end_time"] - meta["start_time"], 1) if "end_time" in meta else ""
    phases = [
        {"phase": "official baseline + TKE/gradient audit", "status": "completed", "updates": 0, "reason": "Baseline reproduced with official v9 scorer."},
        *[{"phase": f"ID screen {v}", "status": "completed", "updates": 300, "reason": "Fixed 16-trajectory dev split."} for v in VARIANTS],
        {"phase": "E3 mean+fluctuation", "status": "skipped_by_gate", "updates": 0, "reason": "Median TKE norm ratio=1.761, not <0.9; observed mechanism was energy overshoot, not energy shrinkage."},
        {"phase": "confirmation / multi-seed", "status": "skipped_by_gate", "updates": 0, "reason": "No E1/E2 candidate improved dev TKE by 3% while preserving Rel-L2/MVPE."},
        {"phase": "OOD AoA=20 training/evaluation", "status": "skipped_by_gate", "updates": 0, "reason": "No candidate passed ID screening; no additional training was warranted."},
        *[{"phase": f"locked-final audit {v}", "status": "completed", "updates": 0, "reason": "One-time audit after dev-only selection; no checkpoint reselection."} for v in VARIANTS],
    ]
    write_csv(out / "experiment_manifest.csv", phases, list(phases[0]))

    # Figures.
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.arange(3); width = 0.24
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for i, metric in enumerate(("rel_l2", "tke", "mvpe")):
        vals = [evaluation[v]["raw_errors"][metric] for v in VARIANTS]
        bars = axes[i].bar(x, vals, width=0.62, color=["#64748b", "#2563eb", "#0f766e"])
        axes[i].set_title(f"Locked final: {metric} (lower is better)")
        axes[i].set_xticks(x, VARIANTS)
        for b, value in zip(bars, vals): axes[i].text(b.get_x()+b.get_width()/2, b.get_height(), fmt(value), ha="center", va="bottom", fontsize=9)
    fig.savefig(figs / "locked_final_raw_metrics.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    order = [(r["candidate"], r["metric"], r) for r in paired_rows]
    labels = [f"{c} {m}" for c, m, _ in order]
    vals = [r["mean_improvement_pct"] for _, _, r in order]
    errs = [[v-r["bootstrap_95_low_pct"] for v, (_, _, r) in zip(vals, order)], [r["bootstrap_95_high_pct"]-v for v, (_, _, r) in zip(vals, order)]]
    colors = ["#0f766e" if v > 0 else "#dc2626" for v in vals]
    ax.bar(labels, vals, yerr=errs, capsize=4, color=colors)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("Paired improvement vs E0 (%)")
    ax.set_title("Locked-final trajectory bootstrap (95% CI; positive is better)")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(figs / "paired_trajectory_bootstrap.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    gnames = [r["loss"] for r in gradient_summary]; gvals = [r["mean_weighted_gradient_norm"] for r in gradient_summary]
    ax.bar(gnames, gvals, color="#7c3aed"); ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylabel("Mean weighted gradient norm (symlog)")
    ax.set_title("Baseline gradient audit: active loss contributions")
    fig.savefig(figs / "gradient_audit.png", dpi=180); plt.close(fig)

    # Cost facts tied to timestamps rather than an assumed GPU throughput.
    screen_seconds = {v: duration(screen[v]["metadata"]) for v in VARIANTS}
    final_seconds = {"E0": 59, "E1": 59, "E2": 55}
    epoch_updates = math.ceil(metadata["train_windows"] / metadata["batch_size"])
    current_epoch_est = {v: screen_seconds[v] / 300 * epoch_updates / 60 for v in VARIANTS}
    e0, e1, e2 = (evaluation[v] for v in VARIANTS)
    e0raw, e1raw, e2raw = (e["raw_errors"] for e in (e0, e1, e2))
    tke_rows = {r["quantity"]: r for r in anatomy_rows}
    e1_tke = next(r for r in paired_rows if r["candidate"] == "E1" and r["metric"] == "tke")
    e2_tke = next(r for r in paired_rows if r["candidate"] == "E2" and r["metric"] == "tke")
    report = f"""# RealPDE Track 1 — Loss Optimization（官方 v9 scorer）\n\n**执行状态：COMPLETED_BOUNDED_EVIDENCE / NO-GO。** 这不是线上提交，也不是 leaderboard 分数。它是一次在 RTX 3090 上完成的、有预先冻结停止门槛的离线实验。\n\n## 结论\n\n**Recommendation: NO-GO**\n\n- **Recommended Loss:** 保持 E0（MSE + 0.05 TKE）作为本轮候选的相对优胜者；E1/E2 不进入长训或提交。\n- **Primary evidence:** 在 16 条完全锁定的 ID trajectory 上，E1/E2 将 Rel-L2 从 {fmt(e0raw['rel_l2'])} 降至 {fmt(e1raw['rel_l2'])}/{fmt(e2raw['rel_l2'])}，MVPE 从 {fmt(e0raw['mvpe'])} 降至 {fmt(e1raw['mvpe'])}/{fmt(e2raw['mvpe'])}；但 TKE 从 **{fmt(e0raw['tke'])}** 恶化到 **{fmt(e1raw['tke'])}/{fmt(e2raw['tke'])}**。\n- **Main risk:** 直接以 Rel-L2/MVPE 对齐的损失得到更平滑、更接近均值的场，却牺牲速度涨落能量；只看点场误差会错选模型。\n- **Next recommended experiment:** 先做“波动能量受约束的精度损失”小样本设计（保留 E0 的强 TKE 项、单独扫描小 Rel/MVPE 权重），并要求同时通过 TKE 与 Rel/MVPE 门槛；不要直接延长 E1/E2。\n\n## 这次实际回答了什么\n\n| 问题 | 证据与答案 |\n|---|---|\n| 当前 TKE 的失效机制 | 基线 dev TKE norm ratio 的中位数为 **{fmt(tke_rows['tke_norm_ratio']['median'])}**（大于 1），且最优缩放系数中位数为 {fmt(tke_rows['tke_scale_alpha']['median'])}。这是能量**过大/尺度失配**，不是 Task Doc 中 E3 要针对的“能量萎缩”（ratio < 0.9）。 |\n| E1 是否有效 | 对 Rel-L2/MVPE 有效，但 TKE 恶化；不满足多指标晋级。锁定集 TKE trajectory 的平均相对改进为 {fmt(e1_tke['mean_improvement_pct'])}%（95% CI {fmt(e1_tke['bootstrap_95_low_pct'])}% 到 {fmt(e1_tke['bootstrap_95_high_pct'])}%），win rate {100*e1_tke['win_rate']:.1f}%。 |\n| E2 是否有效 | 相对 E1 轻微改善 MVPE，但同样恶化 TKE；锁定集 TKE trajectory 的平均相对改进为 {fmt(e2_tke['mean_improvement_pct'])}%（95% CI {fmt(e2_tke['bootstrap_95_low_pct'])}% 到 {fmt(e2_tke['bootstrap_95_high_pct'])}%），win rate {100*e2_tke['win_rate']:.1f}%。 |\n| Metric-aligned loss 是否更优 | **否**，在本轮冻结门槛下不更优：E1/E2 未同时改善 TKE、Rel-L2、MVPE。 |\n| Mean + Fluctuation 是否值得主线 | **本轮未测试**，且是按诊断门槛跳过：E3 只在 median norm ratio < 0.9 时执行，实际为 {fmt(tke_rows['tke_norm_ratio']['median'])}。不能将其称为失败或成功。 |\n| 是否 ID/OOD、多 seed 稳定 | ID screen 已否决，因此依 Task Doc 的节约规则，未耗费更多 GPU 做多 seed/OOD。结论的范围仅限本次单 seed、ID 离线证据。 |\n\n## 设计、数据隔离与评分\n\n- 模型：官方 kit 的 CNO，起点 checkpoint SHA-256 `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`。\n- 数据：82 条本地已下载 PIV H5；按 trajectory 划分，无 window 泄漏。ID split 为 50 train / 16 dev / 16 locked final；锁定集在 screen 阶段从未用于选 checkpoint。另已冻结 AoA=20 的 53/13/16 OOD manifest，但未启动。\n- 评分：使用用户下载的 Track 1 Starting Kit v9 `scoring.py`（SHA-256 `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`），报告其 Rel-L2/TKE/MVPE/Time/SPS 子分。该 kit 不公开 leaderboard 的 final composite，因此本包不制造“总分”。\n- 变体：E0=MSE+0.05TKE；E1=Rel-L2+0.05TKE；E2=E1+0.10MVPE。每个 screen 为 300 update、batch 8、相同 seed 20260901。\n\n## 结果（官方 v9 scorer 的锁定集 raw error；低为好）\n\n| Variant | Rel-L2 | TKE | MVPE | Rel subscore | TKE subscore | MVPE subscore | Time | SPS |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n| E0 | {fmt(e0raw['rel_l2'])} | {fmt(e0raw['tke'])} | {fmt(e0raw['mvpe'])} | {fmt(e0['official_v9_subscores']['rel_l2_score'],1)} | {fmt(e0['official_v9_subscores']['tke_score'],1)} | {fmt(e0['official_v9_subscores']['mvpe_score'],1)} | {fmt(e0['official_v9_subscores']['time_score'],1)} | {fmt(e0['official_v9_subscores']['sps_score'],2)} |\n| E1 | {fmt(e1raw['rel_l2'])} | {fmt(e1raw['tke'])} | {fmt(e1raw['mvpe'])} | {fmt(e1['official_v9_subscores']['rel_l2_score'],1)} | {fmt(e1['official_v9_subscores']['tke_score'],1)} | {fmt(e1['official_v9_subscores']['mvpe_score'],1)} | {fmt(e1['official_v9_subscores']['time_score'],1)} | {fmt(e1['official_v9_subscores']['sps_score'],2)} |\n| E2 | {fmt(e2raw['rel_l2'])} | {fmt(e2raw['tke'])} | {fmt(e2raw['mvpe'])} | {fmt(e2['official_v9_subscores']['rel_l2_score'],1)} | {fmt(e2['official_v9_subscores']['tke_score'],1)} | {fmt(e2['official_v9_subscores']['mvpe_score'],1)} | {fmt(e2['official_v9_subscores']['time_score'],1)} | {fmt(e2['official_v9_subscores']['sps_score'],2)} |\n\n![Locked final raw metrics](figures/locked_final_raw_metrics.png)\n\n![Trajectory bootstrap](figures/paired_trajectory_bootstrap.png)\n\n## 晋级门槛与实际停止\n\nScreen 晋级要求（执行前冻结）：候选必须满足 TKE ≤ 0.97×E0、Rel-L2 和 MVPE 均 ≤ 1.02×E0，且 trajectory win rate ≥60%。\n\n- ID dev E0 的 TKE 是 {fmt(screen['E0']['history'][-1]['tke'])}；E1/E2 分别是 {fmt(screen['E1']['history'][-1]['tke'])}/{fmt(screen['E2']['history'][-1]['tke'])}，均远高于 E0。\n- 因此没有进入 confirmation、multi-seed 或 OOD。E3 因机制 gate 未达到而标记 `SKIPPED_BY_GATE`。这是预定的资源保护规则，而不是运行故障。\n\n## 真实 RTX 3090 实测成本\n\n这次使用一张 RTX 3090（24 GB），官方兼容 Docker 环境，串行运行。不要用历史估算代替这些实测数：\n\n| 工作项 | 实测墙钟 | 说明 |\n|---|---:|---|\n| E0 screen（300 updates） | {screen_seconds['E0']/60:.2f} min | 含训练与两次完整 dev 官方评分 |\n| E1 screen（300 updates） | {screen_seconds['E1']/60:.2f} min | 同上 |\n| E2 screen（300 updates） | {screen_seconds['E2']/60:.2f} min | 同上 |\n| 等价 1 epoch 估算 | E0/E1/E2 = {current_epoch_est['E0']:.2f}/{current_epoch_est['E1']:.2f}/{current_epoch_est['E2']:.2f} min | 本 split 有 {metadata['train_windows']} train windows，batch 8，即 {epoch_updates} updates；按 screen 墙钟线性外推，包含评估开销，故仅作规划量级 |\n| locked final audit（每变体） | 55–59 s | 16 条轨迹；不是训练 |\n| CNO 推理 | {1000*audit['evaluation']['mean_t_neural_s']:.1f} ms/window | 基线 dev，batch 8；不含 dataloader/scorer I/O |\n\n本轮在 gate 后早停，实际只使用约 18 分钟训练/锁定评估主流程，而不是把 6 小时预算机械烧完。\n\n## 可复核文件与限制\n\n- `metrics.csv`、`trajectory_metrics_locked_final.csv`、`paired_trajectory_statistics.csv`：数值与 trajectory bootstrap。\n- `gradient_audit_summary.csv`、`tke_diagnostics.csv`：失效机制证据。\n- `configs/`：两个冻结 split manifests；`source/`：实际 runner 与 loss 实现；`logs_*.log`：远端阶段日志。\n- 没有包含训练 H5、checkpoint、预测数组、starting-kit 副本或任何 submission archive；这些既不应上传给 ChatGPT.com，也不需要才能复核本次统计。\n- locked final 仍是本地公开数据上的一次 audit，不是私有 Codabench test；所有结论都不能外推为 leaderboard 成绩。\n\n## 复核方式\n\n1. 查看 `configs/id_seed20260901.json`，确认 train/dev/final trajectory 不重叠。\n2. 运行 `source/realpde_loss_official_v9.py` 和 `source/run_loss_v9_remote.sh`，在已挂载同版本 kit、数据及 checkpoint 的 GPU 环境复现。\n3. 用 `trajectory_metrics_locked_final.csv` 重算 `paired_trajectory_statistics.csv` 中“较低 error 更好”的 trajectory bootstrap。\n4. 比对 `metrics.csv` 与正式 kit scorer 输出；不要计算或宣称未公开的 final composite。\n"""
    (out / "report.md").write_text(report)
    (out / "SUMMARY_FOR_HUMAN.md").write_text(
        "# 实验结论\n\n**结论：NO-GO。** 这轮不要把 E1 或 E2 延长训练或做提交。它们让普通误差（Rel-L2、MVPE）大幅变好，却把最关键的 TKE 变差。\n\n基线 E0 的锁定集 TKE 为 **1.278**；E1/E2 为 **1.703/1.751**（越低越好）。也就是说，模型更像平均流，但丢了速度涨落的物理信息。\n\n这不是“没有跑够”：预先设定的晋级条件要求 TKE 至少改善 3%，同时其它指标不能变差；E1/E2 明显不达标，所以自动节约了后续多 seed、OOD、长训预算。完整证据、实际 3090 成本和下一步建议见 `report.md`。\n"
    )
    (out / "README_FOR_CHATGPT.md").write_text(
        "# 请审阅此证据包\n\n这是 RealPDE Track 1 loss ablation 的离线审阅包，不是比赛提交包。请先读 `TASK_DOCUMENT.md`（任务与冻结 gate）和 `report.md`；再以 `metrics.csv`、`paired_trajectory_statistics.csv`、`trajectory_metrics_locked_final.csv` 及 `evidence/` 的原始导出文件交叉核对结论。注意：官方 starting kit 未公开 leaderboard final composite，本包不得被解释成总分或线上结果。\n"
    )
    inspection = """# 00 — Repository & Environment Inspection\n\n- 唯一代码仓库：`code/`；本轮新增可复核工具为 `tools/realpde_loss_official_v9.py` 与 `tools/run_loss_v9_remote.sh`。\n- 主模型：官方 Track 1 kit 的 CNO；起点为 real-finetuned CNO checkpoint。\n- 训练/评估：远端 RTX 3090 Docker；数据只读挂载，kit 只读挂载，输出写入隔离 task 目录。\n- 正式指标：Track 1 Starting Kit v9 的 `scoring.py`；非官方 proxy 没有用于模型晋级。\n- 训练数据和 checkpoint 未复制进此审阅包；本报告的 split、哈希和运行脚本足以重放有权限的环境。\n"""
    (out / "00_repository_inspection.md").write_text(inspection)
    print(f"review package assembled at {out}")


if __name__ == "__main__":
    main()
