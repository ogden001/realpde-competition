#!/usr/bin/env python3
"""Frozen batch-profile definitions and selection rules for colleague80 V2."""

from __future__ import annotations

from dataclasses import dataclass, asdict

MIN_B16_THROUGHPUT_GAIN = 1.20
MAX_B16_ALLOCATED_FRACTION = 0.92
MAX_B16_RESERVED_FRACTION = 0.96


@dataclass(frozen=True)
class TrainProfile:
    name: str
    batch_size: int
    lr: float
    updates: int
    eval_interval: int
    matched_control_step: int | None = None

    @property
    def samples(self) -> int:
        return self.batch_size * self.updates

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["samples"] = self.samples
        return value


ARM_PROFILES = {
    "A_pareto_tke": {
        "b8": TrainProfile("b8", 8, 2e-4, 12_000, 2_000),
        "b16": TrainProfile("b16", 16, 2.8e-4, 6_000, 1_000),
    },
    "B_strong_backbone": {
        "b8": TrainProfile("b8", 8, 2e-4, 38_400, 4_800),
        "b16": TrainProfile("b16", 16, 2.8e-4, 19_200, 2_400),
    },
    "C_aoa_meanfield": {
        "b8": TrainProfile("b8", 8, 2e-4, 20_000, 2_500, 5_000),
        "b16": TrainProfile("b16", 16, 2.8e-4, 10_000, 1_250, 2_500),
    },
}


def validate_profiles() -> None:
    for arm, profiles in ARM_PROFILES.items():
        if set(profiles) != {"b8", "b16"}:
            raise ValueError(f"{arm}: expected exactly b8/b16")
        if profiles["b8"].samples != profiles["b16"].samples:
            raise ValueError(f"{arm}: sample exposure differs across profiles")
        if profiles["b8"].batch_size != 8 or profiles["b16"].batch_size != 16:
            raise ValueError(f"{arm}: batch profile drift")


def select_profile(
    b8: dict[str, object],
    b16: dict[str, object],
) -> dict[str, object]:
    """Select b16 only for a clear, safe throughput win.

    Scientific metrics are intentionally absent from this decision.
    """
    b8_ok = bool(b8.get("success"))
    b16_ok = bool(b16.get("success"))
    if not b8_ok:
        raise RuntimeError("b8 reference benchmark failed; cannot select profile")
    if not b16_ok:
        return {
            "selected": "b8",
            "reason": "b16_failed",
            "throughput_gain": None,
        }

    s8 = float(b8["samples_per_second"])
    s16 = float(b16["samples_per_second"])
    gain = s16 / max(s8, 1e-12)
    alloc = float(b16["peak_allocated_fraction"])
    reserved = float(b16["peak_reserved_fraction"])

    checks = {
        "throughput_gain_ge_1p20": gain >= MIN_B16_THROUGHPUT_GAIN,
        "peak_allocated_le_0p92": alloc <= MAX_B16_ALLOCATED_FRACTION,
        "peak_reserved_le_0p96": reserved <= MAX_B16_RESERVED_FRACTION,
    }
    selected = "b16" if all(checks.values()) else "b8"
    return {
        "selected": selected,
        "reason": "all_b16_checks_pass" if selected == "b16" else "b16_checks_failed",
        "throughput_gain": gain,
        "checks": checks,
    }


validate_profiles()
