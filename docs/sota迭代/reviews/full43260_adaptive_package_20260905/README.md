# full@43260 + v5 base adaptive package

Status: `READY_FOR_SUBMISSION_REVIEW`

This is the single `PACKAGE_ONLY` candidate built from the frozen full@43260
backbone and the already-trained v5 `base_head_state_dict@1400`. No training,
recalibration, corrector, locked-final/private-test access, runtime benchmark,
or Codabench submission was performed.

Execution source commit: `dcae3bc` (`Add bounded full43260 adaptive package builder`).

Evidence files:

- `build_report.json` and `build_summary.md`
- `smoke_summary.md`
- `SHA256_PROVENANCE.md`
- `build.review.log`, `smoke.review.log`, `smoke_retry.review.log` (complete raw logs)

The generated ZIP is intentionally kept outside Git at
`artifacts/full43260_adaptive_package_20260905/submission.zip` and was also
left on the GPU run root.
