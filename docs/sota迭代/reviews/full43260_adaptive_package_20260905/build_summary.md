# Build summary

The package was built once into the new GPU OUT_ROOT
`/runs/full43260_adaptive_package_20260905_v2`.

- Backbone: full@43260 P0-A `model_update_43260.pth`.
- Uncertainty: v5 validation `base_head_state_dict@1400`, `hidden=32`, `blocks=2`.
- Bounds: UV `0.0025 + 1.0 * sigma`; pressure half-width `0`.
- Package payload contains only the backbone state and base uncertainty-head state.
- ZIP inventory: 63 files; ZIP size: 30,177,999 bytes.
- No corrector module, corrector weights, or corrected-head weights were copied.

The complete build stdout/stderr is preserved in `build.review.log`.
