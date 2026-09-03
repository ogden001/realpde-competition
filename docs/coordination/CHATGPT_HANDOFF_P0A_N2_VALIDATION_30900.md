# P0-A + N2 50/16 Validation Continuation to 30,900

Reference: `T1-ID-P0A-N2-VALIDATION-30900-S20260903`

## Result

This document recovers the completed P0-A + N2 validation continuation from the prior Codex session. It is a retrospective experiment record: no training was rerun, no checkpoint was changed, and no Codabench submission or locked-final/private-test access occurred while creating this record.

The run uses the fixed 50-train / 16-dev Track 1 validation protocol and official starting-kit v9 scorer raw errors, where lower is better. The method is the same P0-A + N2 competition-oriented validation family used before the all-82-trajectory refit. Because its initialization is the official `sim_real_ft` warm-start, this is `OFFICIAL_WARM_START` evidence rather than clean causal evidence.

The full validation history now spans update 820 through 30,900. Training continues to improve after 10,300 updates. Around 22k updates the run enters a broad plateau / oscillation regime rather than a clear overfitting collapse. Rel-L2 and MVPE retain meaningful late-stage gains; TKE improves more slowly and is visibly noisier across checkpoints.

## Stage Definition

- A: initial fixed budget, 0 → 4,100
- B: continuation, 4,100 → 6,800
- C: continuation, 6,800 → 10,300
- D: continuation, 10,300 → 30,900

Recovered remote result locations from the prior session:

- A: `.../p0a_n2_simreal_validation_20260902/full/`
- B: `.../continuation_4100_to6800/`
- C: `.../continuation_6800_to10300/`
- D: `.../p0a_n2_simreal_validation_20260903/continuation_10300_to30900/run/`

The omitted path prefixes were not preserved in the recovered chat record and are intentionally not guessed here.

## Full Dev Metrics

| Update | Stage | Rel-L2 | TKE | MVPE |
|---:|:---:|---:|---:|---:|
| 820 | A | 0.208258 | 0.699158 | 0.191440 |
| 1,640 | A | 0.165046 | 0.600755 | 0.135805 |
| 2,460 | A | 0.155019 | 0.581221 | 0.128227 |
| 3,280 | A | 0.148534 | 0.548691 | 0.121120 |
| 4,100 | A | 0.141550 | 0.546986 | 0.120323 |
| 4,920 | B | 0.142579 | 0.537596 | 0.116350 |
| 5,740 | B | 0.140899 | 0.534357 | 0.112906 |
| 6,560 | B | 0.136514 | 0.531413 | 0.120715 |
| 6,800 | B | 0.134755 | 0.521813 | 0.108504 |
| 7,380 | C | 0.133431 | 0.510232 | 0.104979 |
| 8,200 | C | 0.130229 | 0.542222 | 0.101615 |
| 9,020 | C | 0.130865 | 0.517756 | 0.106612 |
| 9,840 | C | 0.125992 | 0.507496 | 0.105560 |
| 10,300 | C | 0.123734 | 0.515453 | 0.096836 |
| 10,660 | D | 0.130016 | 0.513366 | 0.103922 |
| 11,480 | D | 0.122212 | 0.514249 | 0.092857 |
| 12,300 | D | 0.120904 | 0.503454 | 0.091011 |
| 13,120 | D | 0.121792 | 0.518687 | 0.096522 |
| 13,940 | D | 0.118646 | 0.497346 | 0.090178 |
| 14,760 | D | 0.121486 | 0.516206 | 0.097503 |
| 15,580 | D | 0.121839 | 0.501015 | 0.103353 |
| 16,400 | D | 0.118596 | 0.501343 | 0.090992 |
| 17,220 | D | 0.118607 | 0.499528 | 0.086383 |
| 18,040 | D | 0.117943 | 0.507150 | 0.099345 |
| 18,860 | D | 0.115782 | 0.504105 | 0.087179 |
| 19,680 | D | 0.119656 | 0.504932 | 0.096329 |
| 20,500 | D | 0.119039 | 0.501616 | 0.098645 |
| 21,320 | D | 0.116700 | 0.506270 | 0.091210 |
| 22,140 | D | 0.116076 | 0.496212 | 0.090992 |
| 22,960 | D | 0.114906 | 0.496516 | 0.087271 |
| 23,780 | D | 0.115843 | 0.500814 | 0.090979 |
| 24,600 | D | 0.113167 | 0.500590 | 0.087900 |
| 25,420 | D | 0.113589 | 0.504909 | 0.088321 |
| 26,240 | D | 0.112925 | 0.494840 | 0.084671 |
| 27,060 | D | 0.117110 | 0.504609 | 0.092968 |
| 27,880 | D | 0.112398 | 0.496896 | 0.088587 |
| 28,700 | D | 0.113669 | 0.496979 | 0.086820 |
| 29,520 | D | 0.113854 | 0.509516 | 0.090787 |
| 30,340 | D | 0.112939 | 0.492848 | 0.088154 |
| 30,900 | D | 0.112845 | 0.500104 | 0.087283 |

## Best Checkpoints

| Criterion | Update | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|---:|
| Best Rel-L2 | 27,880 | **0.112398** | 0.496896 | 0.088587 |
| Best TKE | 30,340 | 0.112939 | **0.492848** | 0.088154 |
| Best MVPE / balanced candidate | 26,240 | 0.112925 | 0.494840 | **0.084671** |

Relative to update 10,300:

- best Rel-L2 error improves by about **9.16%**;
- best TKE error improves by about **4.39%**;
- best MVPE error improves by about **12.56%**.

Update **26,240** is the strongest balanced dev checkpoint candidate in this recovered sweep: its Rel-L2 is only about 0.47% above the best observed Rel-L2, its TKE is only about 0.40% above the best observed TKE, and its MVPE is the best observed value.

This label is deliberately `BALANCED_DEV_CHECKPOINT_CANDIDATE`, not “best submission checkpoint”. Codabench does not publish the final-score combination, and prior submission history shows that stronger local physical metrics do not automatically imply a higher final score or SPS.

## Interpretation

1. The earlier 10,300-update point was not the end of useful optimization. P0-A + N2 continued to improve substantially on the fixed dev protocol through the mid/late-20k range.
2. There is no clear long-horizon overfitting collapse up to 30,900. Instead, after roughly 22k the three metrics fluctuate around a slowly improving plateau.
3. TKE is the least smooth of the three curves. Checkpoint selection should therefore preserve all three official raw errors rather than select solely on Rel-L2 or MVPE.
4. The next useful analysis is not an automatic continuation to 40k/50k. Existing late checkpoints, especially 26,240 / 27,880 / 30,340, should be treated as the meaningful selection set for any later submission/SPS study.
5. No custom final-score proxy is used to promote a checkpoint.

`NEXT_ACTION = REVIEW_REQUIRED`
