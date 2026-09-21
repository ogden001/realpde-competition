# NEXT_ACTION — COLLEAGUE 80PT INCREMENTAL SCREEN

## Goal

Implement, smoke-test, and launch the matched R0/R1/R2/R3 residual screen plus
the matched H0/H1 uncertainty-head input screen on the cloud GPU.

## Tasks

- Verify the archived 80-point assets and released-real data on the cloud GPU.
- Use the archived residual checkpoint as the common start unless a stronger
  checkpoint is independently present and SHA-verified.  Do not assume the
  historical longer-trained checkpoint exists.
- Run each residual arm for 5,000 updates, evaluating every 1,000 updates.
- Train H0 and H1 with the same frozen cache, seed, budget, architecture, loss,
  and calibration grid; H1 adds only `delta_u` and `delta_v`.
- Preserve overall, trajectory-level, and by-horizon evidence for the final
  update of each residual arm.

## Constraints

- Same initialization, optimizer-reset policy, LR schedule, batch, seed,
  global shuffle, all81/Dev16 split manifest, Dev16 windows, and scorer across
  residual arms.
- R3 changes only temporal phase selection.
- H1 changes only uncertainty-head input channels.
- Official-warm-start all81/Dev16 protocol with direct train/dev overlap; do
  not call the overall result clean or unbiased.
- No private external data, Codabench, OSS upload, simulator download,
  automatic long training, or destructive server cleanup.

## Deliverables

- Tested implementation and launch scripts in Git.
- Execution commit, run ID, data/checkpoint/code SHA provenance.
- Remote PID, log path, result path, and user-facing monitoring commands.
- Lightweight results committed after the run is explicitly collected.

## Stop

After the detached screening runner is confirmed running, return its provenance
and monitoring commands.  Do not continuously poll or start long training.
