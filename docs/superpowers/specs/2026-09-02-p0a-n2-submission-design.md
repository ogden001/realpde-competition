# P0-A + N2 Track 1 submission package design

## Goal

Produce one offline, manually uploadable Track 1 `submission.zip` from the
explicit full-data P0-A + N2 checkpoint at iteration 15,300. The package must
be verified locally/inside the official-compatible container but must never
log in to, access, or submit to Codabench.

## Inputs and frozen provenance

- The packager requires `--checkpoint`; it must point to the explicitly
  selected 15,300-update checkpoint. It must not discover or select another
  checkpoint.
- The source checkpoint is the P0-A + N2 full-data continuation checkpoint
  generated at iteration 15,300. Its SHA256, experiment ID and Git commit are
  recorded in the manifest before packaging.
- The official Track 1 v9 starting-kit directory is passed through
  `--kit-root`. Its template and local smoke/evaluator are the interface
  authority.
- No locked-final, private test, online resource, HDF5 file, metadata field,
  Re, AoA, coordinate, mask or training data is read by packaging or inference.

## Package layout and inference behavior

The ZIP root contains `submission.py`, the frozen model checkpoint, and only
the vendored CNO source files required to instantiate the official CNO model.
It contains no optimizer state, data, HDF5 files, logs, training scripts or
unrelated artifacts, and its total size must be below 256 MiB.

`submission.py` exposes:

```python
def predict(input_array, metadata=None) -> numpy.ndarray:
    ...
```

It accepts exactly an array shaped `(N, 20, 32, 64, 3)` and returns a finite
`float32` array of the same shape and channel order `[u, v, p]`. It uses only
the supplied history to recreate the pure-torch P0-A features, sends the
20-channel tensor through the frozen CNO in `eval()` and `torch.inference_mode`,
and sets unavailable pressure to zero. `metadata` is optional and ignored.

Module-level lazy initialization locates every resource relative to
`Path(__file__).resolve().parent`; it loads the model and checkpoint once per
process. There is no networking, download, runtime installation, HDF5 import,
or dependency on a path outside the extracted package.

## Verification pipeline

1. Validate the selected checkpoint iteration, P0-A feature configuration,
   N2 weights, SHA256 and Git commit.
2. Generate deterministic synthetic N=1 and N>1 float32 inputs. Compare a
   direct training-code model load with the submission implementation on the
   same inputs. Record max absolute and relative errors; require exact float32
   equality unless the official module boundary demonstrably introduces a
   machine-precision-only difference.
3. Build the ZIP deterministically and enforce the 256 MiB limit and an
   allowlisted file inventory.
4. In a fresh temporary directory, extract the ZIP and import only its
   `submission.py`. Test N=1 and N>1 output shape, float32 dtype, finite values,
   pressure-zero rule, repeated-call determinism, inference duration and peak
   GPU allocated memory.
5. Invoke the official starting-kit smoke test or local evaluator when it is
   compatible with the package interface. Record the exact command and result.
6. Re-extract and repeat the clean-room prediction comparison after packaging.

Any failed API, dependency, file-size, prediction-consistency, determinism or
official-kit validation check makes the manifest `NOT_READY`; a package may not
be represented as upload-ready in that case.

## Artifacts and report

All generated artifacts live under an explicit directory beneath `artifacts/`:

- `submission.zip`
- `submission.zip.sha256`
- `SUBMISSION_MANIFEST.md`
- machine-readable verification report(s), if needed

The Markdown manifest records the experiment ID, source Git commit, checkpoint
path and SHA256, ZIP SHA256, ZIP inventory and sizes, input/output protocol,
commands/results for all checks, direct-versus-package prediction errors,
inference timing, GPU peak memory and final `READY_TO_SUBMIT` or `NOT_READY`.

## Non-goals

- No Codabench login, page access, submission, score retrieval or quota use.
- No new model training, checkpoint selection, bounds tuning or loss change.
- No fallback to the 10,300 checkpoint: it is retained only as a user-controlled
  recovery artifact and is not part of this package.
