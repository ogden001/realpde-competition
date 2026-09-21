#!/usr/bin/env python3
"""Reference implementation of the RealPDE Track 1 v9 subscores (numpy).

This mirrors the official ``scoring.py`` that Codabench runs for the season
(rel_l2 / TKE / MVPE / time / SPS) so we can evaluate candidates offline with
the exact same numbers instead of a self-written proxy.

Key facts encoded here (see docs/scoring_notes.md):
  * score_error(err) = 100/(1 + 0.5*max(err,0)) for rel_l2/tke/mvpe.
  * time score uses the frozen numerical runtime T_NUMERICAL_SEC.
  * SPS = 0.5*sps_dm + 0.3*sps_tke + 0.2*sps_mvpe, where each branch averages
        (1 - accuracy) * exp(-(upper-lower)/sigma_global) * inside
    over the scored (target != 0) elements, with per-sample accuracy
    acc = error/(0.5+error). ``inside`` means the target falls in [lower, upper].
  * lower/upper may be 3-channel (width on p is ignored after slicing to the
    measured channels) but must be finite, per-element and lower<=upper.

Usage:
    python realpde_sps_scoring.py score  --pred pred.npz --ref targets.npz [--limit N]
    python realpde_sps_scoring.py grid   --pred pred.npz --ref targets.npz [--limit N]
    python realpde_sps_scoring.py oracle --pred pred.npz --ref targets.npz [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

T_NUMERICAL_SEC = 0.72896
SIGMA_GLOBAL = 0.0563870259


# --------------------------------------------------------------------------- #
# per-sample metrics (mirror the official reference implementation)
# --------------------------------------------------------------------------- #
def rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, c: int) -> np.ndarray:
    p = pred[..., :c].reshape(pred.shape[0], -1)
    t = target[..., :c].reshape(target.shape[0], -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=1) / denom


def kinetic_energy(x: np.ndarray) -> np.ndarray:
    u = x[..., 0]
    v = x[..., 1]
    u_prime = np.mean((u - np.mean(u, axis=1, keepdims=True)) ** 2, axis=1)
    v_prime = np.mean((v - np.mean(v, axis=1, keepdims=True)) ** 2, axis=1)
    return 0.5 * (u_prime + v_prime)


def tke_rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, c: int) -> np.ndarray:
    if c < 2:
        return np.zeros((pred.shape[0],), dtype=np.float32)
    pred_ke = kinetic_energy(pred[..., :c])
    target_ke = kinetic_energy(target[..., :c])
    p = pred_ke.reshape(pred.shape[0], -1)
    t = target_ke.reshape(target.shape[0], -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=1) / denom


def mvpe_rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, sub_s_real: int = 2) -> np.ndarray:
    d = 16
    center_x = 10
    center_y = 32
    n_probe = 9
    N, _, h, w, _ = pred.shape
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(h / (n_probe + 1)))
    probe_y = [probe_center_y + interval_y * j for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)]
    probe_y = [y for y in probe_y if 0 <= y < h]
    if not probe_y:
        return np.zeros((N,), dtype=np.float32)
    errors = []
    for i in range(4):
        if int((2 * d + center_x) / sub_s_real) < w:
            probe_x = int(((i + 1) * d + center_x) / sub_s_real)
        else:
            probe_x = int((0.5 * (i + 2) * d + center_x) / sub_s_real)
        if not 0 <= probe_x < w:
            continue
        pp = pred[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(N, -1)
        tt = target[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(N, -1)
        denom = np.linalg.norm(tt, axis=1).clip(min=1e-8)
        errors.append(np.linalg.norm(pp - tt, axis=1) / denom)
    if not errors:
        return np.zeros((N,), dtype=np.float32)
    return np.mean(np.stack(errors, axis=0), axis=0)


def mvpe_rel_l2(pred: np.ndarray, target: np.ndarray, sub_s_real: int = 2) -> float:
    return float(np.mean(mvpe_rel_l2_per_sample(pred, target, sub_s_real)))


# --------------------------------------------------------------------------- #
# subscores
# --------------------------------------------------------------------------- #
def score_error(err: float, scale: float = 0.5) -> float:
    if not np.isfinite(err):
        return 0.0
    return float(100.0 / (1.0 + abs(scale) * max(float(err), 0.0)))


def score_time(t_neural: float) -> float:
    if not np.isfinite(t_neural) or t_neural <= 0.0:
        return 0.0
    r = float(t_neural) / T_NUMERICAL_SEC
    return float(100.0 / (1.0 + r ** 0.5))


def _acc_factor(err: np.ndarray, normalize_factor: float = 0.5) -> np.ndarray:
    # acc = err/(0.5+err), in [0,1); reward factor (1-acc) = 0.5/(0.5+err)
    return 0.5 / (normalize_factor + np.asarray(err, dtype=np.float64))


def aggregate_sps(
    pred: np.ndarray,
    target: np.ndarray,
    c: int,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    weight_dm: float = 0.5,
    weight_tke: float = 0.3,
    weight_mvpe: float = 0.2,
) -> tuple[float, float]:
    """Return (sps, coverage) exactly like the official aggregate_sps."""
    p = pred[..., :c].astype(np.float32, copy=False)
    t = target[..., :c].astype(np.float32, copy=False)
    if lower is None and upper is None:
        interval = 0.1 * np.abs(p)
        lo = p - interval / 2.0
        hi = p + interval / 2.0
    else:
        if lower is None or upper is None:
            raise ValueError("SPS bounds require both lower and upper, or neither.")
        lo = np.asarray(lower, dtype=np.float32)[..., :c]
        hi = np.asarray(upper, dtype=np.float32)[..., :c]
        if lo.shape != p.shape or hi.shape != p.shape:
            raise ValueError(f"bounds shape {lo.shape} != measured prediction shape {p.shape}")
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise ValueError("SPS bounds must be finite.")
        if np.any(lo > hi):
            raise ValueError("SPS bounds require lower <= upper for every element.")

    dm = rel_l2_per_sample(pred, target, c)
    tke = tke_rel_l2_per_sample(pred, target, c)
    mvpe = mvpe_rel_l2_per_sample(pred, target)

    reward_dm = _acc_factor(dm)      # (1 - dm/(0.5+dm))
    reward_tke = _acc_factor(tke)
    reward_mvpe = _acc_factor(mvpe)

    scored = t != 0.0
    n_scored = int(np.count_nonzero(scored))
    if n_scored == 0:
        raise ValueError("Reference has no measured non-zero element.")

    # width in units of sigma; per-element on measured channels
    nil = (hi - lo) / SIGMA_GLOBAL
    inside = (t >= lo) & (t <= hi)
    exp_pen = np.exp(-nil)

    # accuracy factors are constant per sample -> do the masked mean per sample
    # and divide by the global number of scored elements.
    def branch(reward: np.ndarray) -> float:
        total = 0.0
        for n in range(p.shape[0]):
            r = reward[n]
            if not np.isfinite(r):
                continue
            elem = float(r) * exp_pen[n] * inside[n]
            total += float(np.sum(elem, where=scored[n], dtype=np.float64))
        return total / n_scored

    sps = (
        weight_dm * branch(reward_dm)
        + weight_tke * branch(reward_tke)
        + weight_mvpe * branch(reward_mvpe)
    )
    coverage = float(np.count_nonzero(inside & scored) / n_scored)
    return float(sps), coverage


def compute_subscores(
    pred: np.ndarray,
    target: np.ndarray,
    mean_t_neural_s: float = 0.0,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, float]:
    """Return the five 0-100 subscores plus raw errors and coverage."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} != target {target.shape}")
    c = int(measured_channels(target))
    rel = float(np.mean(rel_l2_per_sample(pred, target, c)))
    tke = float(np.mean(tke_rel_l2_per_sample(pred, target, c)))
    mvpe = float(np.mean(mvpe_rel_l2_per_sample(pred, target)))
    sps, coverage = aggregate_sps(pred, target, c, lower=lower, upper=upper)
    out = {
        "rel_l2": rel,
        "rel_l2_score": score_error(rel),
        "tke": tke,
        "tke_score": score_error(tke),
        "mvpe": mvpe,
        "mvpe_score": score_error(mvpe),
        "mean_t_neural_s": float(mean_t_neural_s),
        "time_score": score_time(float(mean_t_neural_s)),
        "sps": sps,
        "sps_score": 100.0 * min(max(sps, 0.0), 1.0),
        "coverage": coverage,
    }
    return out


def measured_channels(target: np.ndarray) -> int:
    active = [not np.allclose(target[..., idx], 0.0) for idx in range(target.shape[-1])]
    return max(1, int(sum(active)))


# --------------------------------------------------------------------------- #
# bound builders (same convention as our submission.py: half width on u/v only,
# p channel zero; lower/upper are 3-channel, official scoring slices to c).
# --------------------------------------------------------------------------- #
def constant_bounds(pred: np.ndarray, abs_w: float, rel_w: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32)
    half = (float(abs_w) + float(rel_w) * np.abs(pred)).astype(np.float32)
    if half.shape[-1] >= 3:
        half[..., 2] = 0.0
    return pred - half, pred + half


def sigma_bounds(
    pred: np.ndarray,
    sigma_uv: np.ndarray,
    floor_w: float = 0.0,
    mult: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive bounds from a per-element sigma field on the u/v channels.

    sigma_uv: [N,T,H,W,2] or [N,T,H,W] (broadcast to both channels).
    half_width on u/v = floor_w + mult * sigma_uv.
    """
    pred = np.asarray(pred, dtype=np.float32)
    sig = np.asarray(sigma_uv, dtype=np.float32)
    if sig.ndim == pred.ndim - 1:
        sig = np.stack([sig, sig], axis=-1)
    half = np.zeros_like(pred, dtype=np.float32)
    half[..., :2] = float(floor_w) + float(mult) * sig[..., :2]
    return pred - half, pred + half


def oracle_bounds(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perfect per-element interval: h = |error| on measured u/v elements.

    This is the theoretical ceiling for adaptive intervals on a fixed point
    forecast (coverage 100%, tightest possible). Used to size the prize.
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    err = np.abs(target - pred)
    err[..., 2:] = 0.0
    return pred - err, pred + err


# --------------------------------------------------------------------------- #
# study helpers
# --------------------------------------------------------------------------- #
def _limit_arrays(pred, target, lower, upper, limit):
    if limit is not None and int(limit) < pred.shape[0]:
        pred = pred[: int(limit)]
        target = target[: int(limit)]
        if lower is not None:
            lower = lower[: int(limit)]
        if upper is not None:
            upper = upper[: int(limit)]
    return pred, target, lower, upper


def grid_search_constant_band(
    pred: np.ndarray,
    target: np.ndarray,
    abs_list: list[float],
    rel_list: list[float],
    limit: int | None = None,
) -> dict[str, object]:
    """Scan constant half-width bands abs + rel*|pred| maximizing real SPS."""
    pred, target, _, _ = _limit_arrays(pred, target, None, None, limit)
    best = {"abs": abs_list[0], "rel": rel_list[0], "sps_score": -1.0, "coverage": 0.0}
    rows = []
    for a in abs_list:
        for r in rel_list:
            lo, hi = constant_bounds(pred, a, r)
            sps, cov = aggregate_sps(pred, target, measured_channels(target), lo, hi)
            row = {"abs": a, "rel": r, "sps": sps, "sps_score": 100.0 * sps, "coverage": cov}
            rows.append(row)
            if row["sps_score"] > best["sps_score"]:
                best = {**row}
    rows.sort(key=lambda row: -row["sps_score"])
    return {"best": best, "rows": rows[:40], "n_scanned": len(rows)}


def oracle_ceiling(pred: np.ndarray, target: np.ndarray, limit: int | None = None) -> dict[str, float]:
    """SPS if we could set h=|error| per element (ceiling for this forecast)."""
    pred, target, _, _ = _limit_arrays(pred, target, None, None, limit)
    lo, hi = oracle_bounds(pred, target)
    sps, cov = aggregate_sps(pred, target, measured_channels(target), lo, hi)
    return {"sps": sps, "sps_score": 100.0 * sps, "coverage": cov}


# --------------------------------------------------------------------------- #
# npz io + cli
# --------------------------------------------------------------------------- #
def load_pred_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def find_target(path: Path) -> Path:
    for cand in [path / "targets.npz", path / "ref" / "targets.npz"]:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"targets.npz not found near {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "grid", "oracle"):
        p = sub.add_parser(name)
        p.add_argument("--pred", type=Path, required=True, help="predictions.npz (keys prediction[/lower/upper/mean_t_neural_s])")
        p.add_argument("--ref", type=Path, default=None, help="targets.npz")
        p.add_argument("--limit", type=int, default=None, help="use first N windows only")
    args = ap.parse_args()

    data = load_pred_npz(args.pred)
    pred = np.asarray(data["prediction"], dtype=np.float32)
    ref_path = Path(args.ref) if args.ref is not None else find_target(args.pred.parent)
    with np.load(ref_path, allow_pickle=False) as zz:
        target = np.asarray(zz["target"], dtype=np.float32)
    if pred.shape != target.shape:
        raise SystemExit(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    lower = data.get("lower")
    upper = data.get("upper")
    t_neural = float(np.asarray(data.get("mean_t_neural_s", [0.0])).reshape(-1)[0])

    if args.cmd == "score":
        pred, target, lower, upper = _limit_arrays(pred, target, lower, upper, args.limit)
        out = compute_subscores(pred, target, t_neural, lower, upper)
    elif args.cmd == "grid":
        out = grid_search_constant_band(pred, target, [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03], [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03], args.limit)
        out["fixed_submission_sps"] = compute_subscores(pred[: args.limit] if args.limit else pred, target[: args.limit] if args.limit else target, t_neural)["sps_score"]
    else:
        out = oracle_ceiling(pred, target, args.limit)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
