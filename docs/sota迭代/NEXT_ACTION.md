# NEXT_ACTION

目标：产出下一版可提交的 Track 1 SOTA candidate。

1. 先完成 P0-A + N2 late-stage LR 小规模 A/B，判断平台期是否需要 `LR × 0.5`。
2. 根据 A/B 结果，从现有 full-data 15,300 checkpoint 继续训练，目标到约 37,850 updates；保留 31,100 / 36,500 / 37,850 checkpoints。
3. 不修改 P0-A feature、N2 loss、CNO 结构、seed、effective batch。
4. 训练完成后单独进行 SPS / bounds 优化。
5. official scorer、runtime smoke、clean-package smoke 通过后再准备 Codabench submission。
6. 不自动加入 H1、FF-01 或新 Loss。
7. 不访问 locked-final/private-test，不自动提交 Codabench。

完成后更新本目录 README 的当日记录，并等待 ChatGPT/Sol review。
