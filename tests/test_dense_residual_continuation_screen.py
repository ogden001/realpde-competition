from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(SCRIPT_DIR))

from run_dense_residual_continuation_screen import (  # noqa: E402
    CURRENT80,
    HISTORICAL_CONTROL,
    HISTORICAL_SPARSE_5K,
    UPDATES,
    EVAL_INTERVAL,
    build_matched_comparison,
    build_progress,
    main,
)


def _write_eval(path: Path, rel: float, tke: float, mvpe: float) -> None:
    path.write_text(
        json.dumps([
            {
                "alpha": 1.0,
                "rel_l2_raw": rel,
                "tke_raw": tke,
                "mvpe_raw": mvpe,
                "best_final_est": 80.0,
            }
        ]),
        encoding="utf-8",
    )


def test_dense_continuation_semantics_are_frozen() -> None:
    assert UPDATES == 5000
    assert EVAL_INTERVAL == 1000
    assert HISTORICAL_CONTROL["stride"] == 20
    assert HISTORICAL_CONTROL["execution_commit"] == "ef9c54f621efcb82703cb2de40ce979c40df3c6a"
    assert HISTORICAL_CONTROL["updates"] == 5000
    assert HISTORICAL_CONTROL["lr"] == 0.0002
    assert HISTORICAL_CONTROL["seed"] == 41
    assert HISTORICAL_SPARSE_5K["rel_l2_raw"] == 0.0802410691976547
    assert CURRENT80["rel_l2_raw"] == 0.0804204195737838


def test_progress_and_comparison_use_alpha_one_eval(tmp_path: Path) -> None:
    for step in range(1000, 5001, 1000):
        _write_eval(
            tmp_path / f"eval_step_{step:05d}.json",
            0.08 - step * 1e-8,
            0.44 + step * 1e-8,
            0.07 - step * 1e-9,
        )

    progress = build_progress(tmp_path)
    comparison = build_matched_comparison(tmp_path)

    assert len(progress) == 6
    assert progress[0]["source"] == "frozen_current80"
    assert progress[-1]["source"] == "dense_stride1_continuation"
    assert comparison[1]["role"] == "matched_control_stride20"
    assert comparison[2]["role"] == "candidate_stride1"
    assert "delta_rel_l2_raw_pct_vs_sparse5k" in comparison[2]



def test_dense_runner_preserves_historical_eval_stride() -> None:
    source = inspect.getsource(main)
    assert '"--stride", "1"' in source
    assert '"--eval-stride", "20"' in source
