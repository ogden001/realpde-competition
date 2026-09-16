# ChatGPT Handoff — SOTA-V2 Full-Data Refit

Status: `COMPLETED`

Execution commit: `b025e44e2202ef40a1cba1a05907650431c4eb3f`
Remote run root: `/home/chyfuture/realpde_runs/sota_v2_full_20260916/run`

Warm-start SHA: `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`
Scorer SHA: `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`

Full trajectories: `82`
Canonical windows: `3383`
Dense windows: `66755`

Stage A end: `49461`
Final update: `53582`

Final checkpoint: `/home/chyfuture/realpde_runs/sota_v2_full_20260916/run/checkpoints/model_update_53582.pth`
Final checkpoint SHA: `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce8`

Runtime: `35208.44 s` (about 9h47m)
Peak GPU: `9.17 GiB` allocated / `9.76 GiB` reserved

Tests: full/integrated/real-fixture focused tests pass; 3 known local Dense-All ABI failures.
Preflight: `PASS` with micro-batch `4`, accumulation `2`, effective batch `8`.

Frozen recipe deviation: `NO`
