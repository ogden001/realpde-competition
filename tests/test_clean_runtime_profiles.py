from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(TOOLS))

from benchmark_clean_runtime_profiles import (  # noqa: E402
    BENCHMARK_UPDATES,
    BENCHMARK_WARMUP,
    build_command,
)


def test_cno_runtime_profile_is_disposable_and_cached(tmp_path: Path) -> None:
    cmd = build_command(
        stage="cno",
        python=sys.executable,
        real_root=Path("/data/real"),
        split_manifest=Path("/tmp/split.json"),
        checkpoint=Path("/tmp/sim_real_cno.pth"),
        model_root=Path("/repo/model"),
        out_dir=tmp_path / "cno_b8",
        workers=4,
        batch_size=8,
    )
    assert "--benchmark-mode" in cmd
    assert "--preload-to-ram" in cmd
    assert cmd[cmd.index("--updates") + 1] == str(BENCHMARK_UPDATES)
    assert cmd[cmd.index("--benchmark-warmup") + 1] == str(BENCHMARK_WARMUP)
    assert cmd[cmd.index("--batch-size") + 1] == "8"


def test_residual_runtime_profile_preserves_clean_protocol(tmp_path: Path) -> None:
    cmd = build_command(
        stage="residual",
        python=sys.executable,
        real_root=Path("/data/real"),
        split_manifest=Path("/tmp/split.json"),
        checkpoint=Path("/tmp/cno_best.pth"),
        model_root=Path("/repo/model"),
        out_dir=tmp_path / "res_b16",
        workers=4,
        batch_size=16,
    )
    assert "--benchmark-mode" in cmd
    assert "--train-on-all" not in cmd
    assert "--allow-train-dev-overlap" not in cmd
    assert cmd[cmd.index("--stride") + 1] == "1"
    assert cmd[cmd.index("--eval-stride") + 1] == "20"
    assert cmd[cmd.index("--eval-alphas") + 1] == "1.0"
    assert cmd[cmd.index("--selection-metric") + 1] == "point_score"
    assert cmd[cmd.index("--batch-size") + 1] == "16"
