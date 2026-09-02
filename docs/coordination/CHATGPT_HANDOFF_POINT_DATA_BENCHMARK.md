# Point-V0 data-pipeline benchmark handoff

Date: 2026-09-02
Reference: `T1-ID-POINT-DATA-BENCH-20260902`
Scope: engineering benchmark only; no Point-V0 retraining, Point-V1 training, dev/final target access, scorer, or Codabench.

## Executive result

The smoke benchmark already identifies the order-of-magnitude bottleneck: per-window HDF5 access dominates the current Point-V0 path. A trajectory-level RAM cache (B3) reached about **998 windows/s** on the training path versus **37 windows/s** for the current loader, approximately **27x** faster in this small smoke and with data-wait ratio reduced from **97.5% to 27.5%**. Data-only throughput was **3,929 vs 35.6 windows/s** (about **110x**).

The long formal sweep was intentionally paused at the user's request after completing B0. It is not a complete candidate ranking and must not be presented as one.

## Frozen protocol

- Train split only: 50 trajectories from manifest seed `20260901`.
- Batch size 8 complete windows; `T_in=20`, `T_out=20`, stride 20, subsample 2, float32, pressure zero-filled as in the current loader.
- One fixed repeated source-index order is shared by all candidates.
- Profiles: A (`DataLoader/HDF5 -> CPU batch`) and B (Point-V0 MLP forward + existing loss + backward).
- Formal timing: warmup 100, measured 1000, repeats 3.
- Exact DATA_EQUIVALENCE check: fixed windows, x/y max absolute difference must equal 0.

## Results

Formal B0 (24,000 measured windows per profile):

| profile | windows/s | mean latency (ms) | data wait (ms) | data-wait ratio | RSS (MB) |
|---|---:|---:|---:|---:|---:|
| A data-only | 33.64 | 237.80 | 237.80 | 1.000 | 526.8 |
| B training-path | 33.25 | 240.57 | 230.72 | 0.959 | 978.0 |

Smoke (5 measured batches, one repeat) — all B0/B2/B3 exact-equivalence checks passed with x/y max absolute difference 0:

| profile | candidate | windows/s | mean latency (ms) | data-wait ratio | RSS (MB) | cache |
|---|---|---:|---:|---:|---:|---|
| A | B0_CURRENT | 35.59 | 224.81 | 1.000 | 519.9 | HDF5 per window |
| B | B0_CURRENT | 37.23 | 214.86 | 0.975 | 968.9 | HDF5 per window |
| A | B2_HANDLE_W2 | 29.67 | 269.64 | 1.000 | 987.1 | worker-local H5 handle |
| B | B2_HANDLE_W2 | 38.07 | 210.15 | 0.976 | 987.4 | worker-local H5 handle |
| A | B3_RAM_W0 | 3929.11 | 2.04 | 1.000 | 2316.6 | ~1.04 GB cache |
| B | B3_RAM_W0 | 998.02 | 8.02 | 0.275 | 2316.6 | ~1.04 GB cache |

## Run state and evidence

- Formal run was paused cleanly; remote container exited after user request.
- Remote run: `/home/chyfuture/realpde_runs/point_data_benchmark_s20260901`
- Remote status: `PAUSED`, completed candidates: `B0_CURRENT`.
- Remote partial artifacts: `metrics.csv`, `equivalence.json`, `provenance.json`, `benchmark_indices.json`, `status.json`, `run.log`.
- Local copies of those artifacts are under `artifacts/point_data_benchmark_s20260901_review/` and are intentionally not committed as a large data bundle.

## Engineering conclusion

The dominant issue is not MLP/GPU arithmetic. The current loader reopens an HDF5 file for every window; with two workers this leaves the GPU waiting for CPU/I/O. B2 worker-local handles are not a reliable large gain in the smoke. B3 trajectory RAM caching is the first optimization candidate, subject to validating host-memory headroom and startup cost. A contiguous cache/memmap (B4) is only needed if the approximately 1 GB cache is unacceptable.

The benchmark does not alter the frozen Point-V0 accuracy result (`STOP_PURE_POINT`) and does not authorize Point-V1.

## Reproduction

Implementation: [`tools/realpde_point_data_benchmark.py`](../../tools/realpde_point_data_benchmark.py)
Formal command (remote container):

```bash
python realpde_point_data_benchmark.py \
  --data-root /data/p0ab_real_h5_20260830 \
  --manifest /runs/point_data_benchmark_s20260901/manifest.json \
  --out-dir /out/artifacts --seed 20260901 --batch-size 8 \
  --warmup 100 --measured 1000 --repeats 3 --eq-windows 100 \
  --device cuda --profiles A,B
```
