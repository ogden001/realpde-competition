# Clean-room smoke summary

The ZIP was extracted into an independent temporary directory inside the
official GPU container and imported there. A fixed random input with seed
`20260905` and shape `(1,20,32,64,3)` was used.

- Original full@43260 direct-backbone prediction max absolute difference: `0.0`.
- Output shape: `(1,20,32,64,3)`; dtype: `float32`; all values finite.
- Pressure channel: exactly zero.
- Repeated calls: byte-identical prediction, lower, and upper arrays.
- Sigma: finite and strictly positive; observed min `0.02895493060350418`, max `1.0`.
- Bounds: UV formula matched `0.0025 + sigma`; pressure half-width was zero.
- Maximum float32 arithmetic comparison error: `2.384185791015625e-07`.
- ZIP inventory contained 63 files and no `ResidualCorrector3D` source/path.

The first smoke log is retained as `smoke.review.log`; its only failure was an
overly strict zero-tolerance comparison across PyTorch/NumPy float32 rounding.
The corrected bounded smoke is retained as `smoke_retry.review.log` and passed.
