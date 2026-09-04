from tools import build_training_review_log as review


def test_small_log_is_kept_in_full():
    lines = [f"TRAIN update={i}\n" for i in range(10)]
    selected, meta = review.build_review(
        lines,
        full_copy_max_lines=1000,
        full_copy_max_bytes=200 * 1024,
        head_lines=20,
        tail_lines=20,
        sample_lines=40,
    )
    assert selected == lines
    assert meta["selection_mode"] == "full"


def test_large_log_keeps_events_head_tail_and_deterministic_train_samples():
    lines = [f"TRAIN update={i} loss={1 / (i + 1):.6f}\n" for i in range(200)]
    lines[80] = "EVAL update=80 rel=0.1\n"
    lines[120] = "WARNING gradient spike\n"

    kwargs = dict(
        full_copy_max_lines=10,
        full_copy_max_bytes=100,
        head_lines=3,
        tail_lines=3,
        sample_lines=7,
    )
    first, meta1 = review.build_review(lines, **kwargs)
    second, meta2 = review.build_review(lines, **kwargs)

    assert first == second
    assert meta1 == meta2
    assert lines[80] in first
    assert lines[120] in first
    assert lines[0] in first and lines[1] in first and lines[2] in first
    assert lines[-1] in first and lines[-2] in first and lines[-3] in first
    assert meta1["selection_mode"] == "compressed_train"
    assert meta1["event_line_count"] == 2
    assert meta1["sampled_line_count"] == 7


def test_large_log_without_train_lines_samples_ordinary_lines():
    lines = [f"step {i}\n" for i in range(100)]
    lines[50] = "ERROR bad thing\n"
    selected, meta = review.build_review(
        lines,
        full_copy_max_lines=10,
        full_copy_max_bytes=100,
        head_lines=2,
        tail_lines=2,
        sample_lines=5,
    )
    assert lines[50] in selected
    assert meta["selection_mode"] == "compressed_ordinary"
    assert meta["event_line_count"] == 1
    assert meta["sampled_line_count"] == 5
