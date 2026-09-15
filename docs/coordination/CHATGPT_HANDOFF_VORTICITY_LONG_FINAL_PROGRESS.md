# Vorticity Supervision Long Final — Progress Handoff

> 阶段性审核报告，不是终局结论。V1-LONG、final replay、mechanical gate 和最终文档关闭尚未完成。

## 1. Execution status

- `C0-LONG`: completed through absolute update `15000`.
- `V1-LONG`: not started; waiting for the target RTX 3090 to become free.
- Current external GPU occupancy: unrelated `residual_multi.py` processes, approximately `4.85 GiB` GPU memory.
- Final gate: not computed.
- Scientific conclusion: not issued.

## 2. Protocol / provenance

The completed C0 run uses the frozen long-final protocol:

- 50 train / 16 dev frozen manifest;
- P0-A, N2, stride `20`;
- AdamW, `lr=1e-5`;
- physical batch `8`, no gradient accumulation;
- seed `20260901`, workers `2`;
- starting checkpoint: R2 C0@3000;
- absolute milestones: `3000 / 4500 / 6000 / 9000 / 12000 / 15000`;
- locked-final, full-data, SPS, and Codabench access: false.

Preflight passed for both C0 and V1 starting checkpoints: model/optimizer restoration, absolute update `3000`, batch protocol, DataLoader ordering, pressure zero, finite prediction, and checkpoint round-trip. V1 lambda is fixed at `15.5385751724`.

## 3. C0-LONG raw metrics

| Update | Rel-L2 | TKE | MVPE |
|---:|---:|---:|---:|
| 3000 | 0.1613427252 | 0.5579713583 | 0.1314960271 |
| 4500 | 0.1643437743 | 0.5606963038 | 0.1399780810 |
| 6000 | 0.1434570402 | 0.5299122929 | 0.1180881038 |
| 9000 | 0.1361329108 | 0.5304250717 | 0.1029610783 |
| 12000 | 0.1329368651 | 0.5083196759 | 0.1041182801 |
| 15000 | 0.1233130097 | 0.5104776621 | 0.0903621092 |

All six C0 milestone checkpoints and evaluations exist. `eval_15000/predictions.npz` is present.

## 4. Remaining work

1. Run V1-LONG from R2 V1@3000 to update `15000` with the fixed vorticity loss.
2. Generate matched metrics, trajectory stability, late-checkpoint gate, and combined runtime/metadata files.
3. Run the independent official v9 final replay for C0@15000 and V1@15000.
4. Write the final handoff with `FINAL_GATE = MERGE_CANDIDATE` or `PARK`.
5. Update the experiment registry and modeling overview with the mechanical final result and close the exploration.

## 5. Artifacts

- Remote root: `/home/chyfuture/realpde_runs/vorticity_long_final_20260914/`
- C0 run: `/home/chyfuture/realpde_runs/vorticity_long_final_20260914/C0-LONG/`
- Preflight: `/home/chyfuture/realpde_runs/vorticity_long_final_20260914/preflight_C0_fixed/` and `/home/chyfuture/realpde_runs/vorticity_long_final_20260914/preflight_V1_fixed/`
- Runner: `tools/realpde_vorticity_long_final.py`
- Summary: `tools/realpde_vorticity_long_final_summary.py`
- Replay: `tools/realpde_vorticity_long_final_replay.py`

## 6. Tests

- Focused tests: `3 passed`.
- New scripts: `py_compile` passed locally and in the official remote container.
- `git diff --check`: passed.

This report intentionally does not assign a final scientific gate.
