# ChatGPT Handoff — SOTA-V2 Integrated 50/16

Status: `COMPLETED`

- Execution commit: `89fe59dc23d31617d6212039a02d3706e11de40f`
- Remote run root: `/home/chyfuture/realpde_runs/sota_v2_integrated_50_16_20260916/run`
- Warm-start checkpoint: `/home/chyfuture/RealPDE_data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth`
- Checkpoint SHA: `82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61`
- Tests: SOTA-V2 and real fixture tests passed. Local Dense-All tests retain the known PyTorch/NumPy ABI incompatibility only.
- CUDA preflight: `PASS`

| Checkpoint | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| 30k | 0.105989 | 0.474373 | 0.090337 |
| 35k | 0.099978 | 0.471441 | 0.075749 |
| Historical Dense-All | 0.110092 | 0.479399 | 0.080831 |

- h19+h20 fraction: 30k `0.251052`; 35k `0.242244`
- Runtime: `23117.20 s` (about 6h25m)
- Peak GPU: allocated `9.29 GiB`; reserved `14.19 GiB`
- Frozen recipe deviation: `NO`
