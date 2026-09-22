# Archived screening evidence

This directory contains lightweight, reviewable outputs from the completed cloud-GPU screening run. It deliberately excludes model checkpoints, generated caches, raw H5 data, and training logs.

- `RESULTS.md`: human-readable result table and gate analysis.
- `screen_report.json`: runner-generated final metrics and gate decisions.
- `run_manifest.json`: code/data/checkpoint/environment provenance.
- `commands.jsonl`: exact remote commands used for the seven stages.
- `R*/` and `H*/`: final summaries and primary metrics.
- `colleague_dev16_manifest.json`: the colleague-aligned deterministic split.
- `data_manifest.tsv`: hashes and sizes for the released real-data files used remotely.

The experiment is competition-oriented and uses all81/Dev16 overlap by explicit design; it must not be presented as clean offline generalization.
