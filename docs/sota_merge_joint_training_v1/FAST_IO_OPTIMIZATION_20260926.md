# Full82 Fast-I/O Optimization V1

Status: `READY_FOR_EXECUTION / REVIEW_REQUIRED`

## Goal

Remove the runtime bottlenecks in the Full82 Corrector / Joint training path without changing the scientific training recipe.

## Frozen scientific semantics

- Full82: 82 released H5 trajectories.
- Dense-All: every legal stride-1 temporal start.
- Input / output: 20 / 20.
- P00 spatial sampling for the Corrector.
- Batch size: 8.
- Precision: FP32, unchanged.
- Corrector architecture, objective, loss weights, optimizer family, LR, weight decay and cosine schedule: unchanged.
- Corrector final target: 50,000 updates.
- Backbone: Full Stage-A @90,000, SHA256 `94d2d2d37110d76fd5104c7e6498b380b022e20f1f1cb621e17ba67165040e6c`.
- No Dev, AoA10, locked-final, Codabench, Joint or SPS access from the fast Corrector runner.

## Runtime-only changes

1. `H5WindowDataset(preload_to_ram=True)` reads each trajectory once and stores the sub-sampled three-channel tensor in host RAM.
2. Cached `__getitem__` performs only temporal slicing; it does not open HDF5.
3. DataLoader uses pinned memory, persistent workers and configurable prefetch.
4. Dense-All sampler supports `start_index` for exact mid-epoch continuation.
5. Corrector logging synchronizes GPU only at the configured log interval instead of every optimizer update.
6. Spatial-phase RAM preload caches Re/AoA so the Joint Stage-A hot path also performs no HDF5 open.

## Exact Corrector @20k continuation

For 66,755 Dense-All windows and batch 8 with `drop_last=True`:

- usable windows / epoch = 66,752
- optimizer updates / epoch = 8,344
- completed updates = 20,000
- resume epoch = 2
- completed batches inside epoch 2 = 3,312
- resume sample offset = 26,496

The fast runner derives this state from the checkpoint update count and refuses to continue unless:

- the checkpoint is bound to the exact @90k backbone;
- optimizer state exists;
- scheduler state exists;
- scheduler `T_max == 50,000`;
- scheduler `last_epoch == checkpoint updates`.

## Code

- `tools/realpde_p0_data.py`
- `tools/realpde_sota_v2_integrated.py`
- `tools/realpde_sota_merge_spatial.py`
- `tools/train_full82_corrector_fast.py`
- `tools/train_sota_merge_joint_full.py`
- `tests/test_fast_io_cache.py`

## Execution gate

Before launching the continuation:

1. stop the old slow Corrector process after confirming the 20k checkpoint is readable;
2. run focused tests and `py_compile`;
3. run `train_full82_corrector_fast.py --preflight-only`;
4. verify the preflight reports resume update 20,000, epoch 2, offset 26,496, cache enabled for 82 trajectories, and no optimizer step;
5. start the continuation in a fresh output directory.

After at least 500 real optimizer updates, record:

- updates/s;
- mean GPU utilization over a 60-second sample;
- GPU memory;
- host RAM.

If the fast runner is still below 5 updates/s and mean GPU utilization remains below 50%, stop at the next recovery checkpoint and return evidence for another runtime pass. Do not change batch size, precision, loss, LR or schedule autonomously.
