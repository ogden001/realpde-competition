# Track 1 Dataset Profile

Status: `PENDING_INITIAL_PROFILE`

This file is the persistent dataset-distribution baseline for Track 1 experiments.

Current frozen split:

- 50 Train trajectories
- 16 Dev trajectories
- 16 Locked-final trajectories
- Manifest SHA-256: `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347`

Locked-final is intentionally excluded from development profiling and bad-case analysis.

The initial profile must be generated according to `docs/data/SKILL.md` and should include:

1. basic inventory and window protocol;
2. Train / Dev input-side descriptor distributions;
3. Dev coverage / nearest-neighbor / OOD-like assessment;
4. per-trajectory descriptor table;
5. target-side descriptors clearly marked `analysis-only`;
6. concise stable conclusions useful for later experiments.

Do not fill this document from historical guesses. Generate it from the frozen dataset and record the analysis code commit and manifest used.
