"""Frozen definitions and validation for the colleague incremental screen."""

from __future__ import annotations


REGISTERED_ARMS: dict[str, dict[str, object]] = {
    "R0": {"tke": 0.06, "window_mode": "fixed"},
    "R1": {"tke": 0.09, "window_mode": "fixed"},
    "R2": {"tke": 0.12, "window_mode": "fixed"},
    "R3": {"tke": 0.06, "window_mode": "random_phase"},
}


def validate_registered_arms(arms: dict[str, dict[str, object]]) -> None:
    if set(arms) != {"R0", "R1", "R2", "R3"}:
        raise ValueError("screen must contain exactly R0-R3")
    for name, config in arms.items():
        if set(config) != {"tke", "window_mode"}:
            raise ValueError(f"{name} has an unregistered variable")
        if config["window_mode"] not in {"fixed", "random_phase"}:
            raise ValueError(f"{name} has an invalid window mode")


validate_registered_arms(REGISTERED_ARMS)


def _relative_change(control: float, candidate: float) -> float:
    if control <= 0:
        raise ValueError("control raw error must be positive")
    return (candidate - control) / control


def tke_gate(control: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    tke_change = _relative_change(control["tke_raw"], candidate["tke_raw"])
    rel_change = _relative_change(control["rel_l2_raw"], candidate["rel_l2_raw"])
    mvpe_change = _relative_change(control["mvpe_raw"], candidate["mvpe_raw"])
    checks = {
        "tke_reduction_at_least_3pct": tke_change <= -0.03,
        "rel_degradation_at_most_1pct": rel_change <= 0.01,
        "mvpe_degradation_at_most_1pct": mvpe_change <= 0.01,
        "fixed_time_composite_not_lower": candidate["final_est"] >= control["final_est"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "relative_changes": {"rel_l2": rel_change, "tke": tke_change, "mvpe": mvpe_change},
        "composite_delta": candidate["final_est"] - control["final_est"],
    }


def random_phase_gate(control: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    changes = {
        name: _relative_change(control[key], candidate[key])
        for name, key in (("rel_l2", "rel_l2_raw"), ("tke", "tke_raw"), ("mvpe", "mvpe_raw"))
    }
    checks = {
        "at_least_two_metrics_improve": sum(value < 0 for value in changes.values()) >= 2,
        "no_metric_degrades_over_1pct": max(changes.values()) <= 0.01,
        "fixed_time_composite_gain_at_least_0_15": candidate["final_est"] - control["final_est"] >= 0.15 - 1e-12,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "relative_changes": changes,
        "composite_delta": candidate["final_est"] - control["final_est"],
    }
