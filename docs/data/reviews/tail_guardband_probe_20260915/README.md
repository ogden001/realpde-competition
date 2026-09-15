# Future40 Guard-Band Inference Probe

结论标签：`RIGHT_BOUNDARY_HYPOTHESIS_NOT_SUPPORTED`；状态：`REVIEW_REQUIRED`

1. checkpoint / commit / split

checkpoint: `p0a_n2_simreal_validation_20260903/continuation_10300_to30900/run/model_last.pth@30900` (iteration `30900`; SHA-256 `e3d5faaf1a71e121b09077dd7dd7d0456a617e2916b8a671986f412fb54f6388`)

execution commit: `472adbc3253dd130fb9aeceaf3fa74f7f9cbf1fd`

split: fixed 50 train / 16 dev, 659 windows; no locked-final access.

2. T=40 是否直接支持

支持。未修改网络参数、模型结构或 checkpoint；同一 P0-A wrapper 直接完成 `[B,40,32,64,3] -> [B,40,32,64,3]`。

3. h1-10 / h11-18 / h19-20 结果

Rel-L2 mean h1-h10: A `0.09632175`, B `0.36875457`, improvement `-283.332%`.

Rel-L2 mean h11-h18: A `0.11487378`, B `0.47857633`, improvement `-315.843%`.

Rel-L2 mean h19-h20: A `0.16424564`, B `0.50595391`, improvement `-209.420%`.

h1-h15 mean signed improvement `-290.288%`; mean absolute change `290.288%`.

4. h19/h20 improvement

h19: A `0.15308228` -> B `0.50458825`, improvement `-229.619%`.

h20: A `0.17540900` -> B `0.50731951`, improvement `-189.221%`.

5. 是否支持 right-boundary hypothesis

`RIGHT_BOUNDARY_HYPOTHESIS_NOT_SUPPORTED`. Gate uses both h19/h20 improvements >=10% and h1-h15 mean absolute Rel-L2 change <=3%; no parameter tuning was performed.

6. 下一步建议

Review the paired frame table and complete output40 statistics. Output40 tail comparison is in `output40_tail_comparison.json`; h39/h40 simple mean/std/speed statistics are in `output40_stats.csv` (no Future21-40 ground truth was used).
