# NEXT_ACTION

## Goal

把已选 SPS bounds `abs=0.0075, rel=0.02` 固定应用到 full@`43260` primary 与 full@`40560` backup，完成 submission package + clean-room smoke；本阶段不提交 Codabench。

## Tasks

1. 确认工作区无未知未提交改动，执行：
   - `git fetch origin`
   - `git pull --rebase origin main`
   - 确认 `HEAD == origin/main`
   - 确认 commit `3a90342fd1fd3c95ee9c13b0309b5f9ba139a4fb` 已包含在当前 HEAD。
2. 运行：
   `pytest -q tests/test_p0a_submission_package.py tests/test_p0a_submission_smoke.py tests/test_sota_sps_screen.py tests/test_runtime_context.py`
3. 从昨晚 artifact `/home/chyfuture/realpde_runs/sota_full_night_20260904_retry2/` 的 manifest/实际文件中定位：
   - primary：exact update `43260` checkpoint；
   - backup：exact update `40560` checkpoint；
   - 两者必须是 `P0-A` + frozen four-key N2，不得用相邻 update 替代。
4. 解析真实 `KIT_ROOT`，使用一个新的 `OUT_ROOT`。分别构建：

```bash
python tools/build_p0a_n2_submission.py \
  --checkpoint "$PRIMARY_CHECKPOINT" \
  --required-iteration 43260 \
  --kit-root "$KIT_ROOT" \
  --out-dir "$OUT_ROOT/full43260_abs0075_rel002" \
  --experiment-id T1-COMP-P0A-N2-FULL43260-SPS-A0075-R002 \
  --git-commit "$(git rev-parse HEAD)" \
  --bound-abs 0.0075 \
  --bound-rel 0.02

python tools/build_p0a_n2_submission.py \
  --checkpoint "$BACKUP_CHECKPOINT" \
  --required-iteration 40560 \
  --kit-root "$KIT_ROOT" \
  --out-dir "$OUT_ROOT/full40560_abs0075_rel002" \
  --experiment-id T1-COMP-P0A-N2-FULL40560-SPS-A0075-R002 \
  --git-commit "$(git rev-parse HEAD)" \
  --bound-abs 0.0075 \
  --bound-rel 0.02
```

5. 对两个 ZIP 分别运行 clean-room smoke：

```bash
python tools/realpde_p0a_submission_smoke.py \
  --zip "$OUT_ROOT/full43260_abs0075_rel002/submission.zip" \
  --required-iteration 43260 \
  --bound-abs 0.0075 \
  --bound-rel 0.02 \
  --out "$OUT_ROOT/full43260_abs0075_rel002/smoke.json"

python tools/realpde_p0a_submission_smoke.py \
  --zip "$OUT_ROOT/full40560_abs0075_rel002/submission.zip" \
  --required-iteration 40560 \
  --bound-abs 0.0075 \
  --bound-rel 0.02 \
  --out "$OUT_ROOT/full40560_abs0075_rel002/smoke.json"
```

6. 汇报两套 package 的：checkpoint path/SHA256、ZIP path/SHA256/size、smoke status、shape/dtype/finite/pressure-zero/deterministic/bounds-match。

## Constraints

- primary 固定 `43260`，backup 固定 `40560`。
- bounds 固定 `abs=0.0075, rel=0.02`，不得继续调参或扩 SPS 网格。
- 不训练模型，不修改 checkpoint，不访问 locked-final/private-test。
- 不提交 Codabench；不根据 smoke 自行选择线上 winner。
- 任一 exact checkpoint 不存在、iteration/config 不匹配、测试或 smoke 失败，立即停止并报告。

## Deliverables

新的 `OUT_ROOT` 下保留：
- `full43260_abs0075_rel002/submission.zip`
- `full43260_abs0075_rel002/build_report.json`
- `full43260_abs0075_rel002/smoke.json`
- `full40560_abs0075_rel002/submission.zip`
- `full40560_abs0075_rel002/build_report.json`
- `full40560_abs0075_rel002/smoke.json`

## Stop

两个 package 均完成 clean-room smoke 后立即停止，状态交给 ChatGPT/Sol review；不自动提交 Codabench。
