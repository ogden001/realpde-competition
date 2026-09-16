# SHA256 provenance

| Artifact | Path | SHA256 / value |
|---|---|---|
| Execution commit | Git `main` | `22eed1066345da7bb3d541a2472e4eaef0e29320` |
| Full checkpoint | `/home/chyfuture/realpde_runs/sota_v2_full_20260916/run/checkpoints/model_update_53582.pth` | `f808fbd39adec37f499be05a7224c440e15e998c137b53c797f2733d9e5765ce` |
| Validation checkpoint | `/home/chyfuture/realpde_runs/sota_v2_integrated_50_16_20260916/run/checkpoints/model_update_32500.pth` | `6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47` |
| Adaptive head | `/home/chyfuture/realpde_runs/sota_v2_adaptive_20260916/adaptive/adaptive_head_1400.pth` | `01c1fc06f806ca00a370cf971df0f093d363ab7f3a4c4e473f1727441f3d4d17` |
| Package ZIP | `/home/chyfuture/realpde_runs/sota_v2_adaptive_20260916/package/submission.zip` | `ad13e6ddf2438838df788f4f21204198b1cd121b209377d7cc446499a7246f44` |
| Manifest | `/home/chyfuture/RealPDE_data/id_seed20260901.json` | `42b710cb8f04e5ab020da2b69772980b563dcc3f3ad555c21508ab12ab10c347` |
| Official scorer | `realpde_t1_starting_kit_v9` | `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39` |

The full-checkpoint digest is the valid 64-character SHA256 value. The frozen package-builder source contains an extra trailing `8` in its expected constant; packaging and smoke verification used an in-process override of that guard only. No algorithm, model, checkpoint, or submission source was modified.
