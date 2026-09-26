#!/usr/bin/env python3
"""Minimal inference-only runtime for the reviewed Clean Joint matched SPS head.

The feature semantics intentionally match tools/colleague_80pt/residual_multi.py
used by train_sota_merge_sps.py, but this module contains no training, split,
scoring, calibration, or model-selection code.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


def _central_diff(arr: torch.Tensor, axis: int) -> torch.Tensor:
    axis %= arr.ndim
    out = torch.empty_like(arr)
    n = arr.shape[axis]
    if n <= 1:
        out.zero_(); return out
    middle = [slice(None)] * arr.ndim
    before = [slice(None)] * arr.ndim
    after = [slice(None)] * arr.ndim
    middle[axis], before[axis], after[axis] = slice(1, -1), slice(None, -2), slice(2, None)
    out[tuple(middle)] = 0.5 * (arr[tuple(after)] - arr[tuple(before)])
    first = [slice(None)] * arr.ndim; second = [slice(None)] * arr.ndim
    first[axis], second[axis] = 0, 1
    out[tuple(first)] = arr[tuple(second)] - arr[tuple(first)]
    last = [slice(None)] * arr.ndim; prev = [slice(None)] * arr.ndim
    last[axis], prev[axis] = -1, -2
    out[tuple(last)] = arr[tuple(last)] - arr[tuple(prev)]
    return out


def _backward_diff(arr: torch.Tensor, axis: int) -> torch.Tensor:
    axis %= arr.ndim
    out = torch.zeros_like(arr)
    if arr.shape[axis] <= 1:
        return out
    cur = [slice(None)] * arr.ndim; prev = [slice(None)] * arr.ndim
    cur[axis], prev[axis] = slice(1, None), slice(None, -1)
    out[tuple(cur)] = arr[tuple(cur)] - arr[tuple(prev)]
    return out


def _coordinates(u: torch.Tensor):
    shape = tuple(u.shape)
    t, h, w = shape[-3:]
    leading = (1,) * (len(shape) - 3)
    x = torch.linspace(-1.0, 1.0, w, device=u.device, dtype=u.dtype).reshape(leading + (1, 1, w))
    y = torch.linspace(-1.0, 1.0, h, device=u.device, dtype=u.dtype).reshape(leading + (1, h, 1))
    tm = torch.linspace(-1.0, 1.0, t, device=u.device, dtype=u.dtype).reshape(leading + (t, 1, 1))
    return x.expand(shape), y.expand(shape), tm.expand(shape)


def ensure_three_channels(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] >= 3:
        return x[..., :3]
    return torch.cat([x[..., :2], torch.zeros_like(x[..., :1])], dim=-1)


def zero_pressure(x: torch.Tensor) -> torch.Tensor:
    x = ensure_three_channels(x).clone()
    x[..., 2] = 0.0
    return x


def augment_torch(x: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    x = ensure_three_channels(x)
    if x.dtype != torch.float32:
        x = x.float()
    u, v, p = x[..., 0], x[..., 1], x[..., 2]
    du_dt, dv_dt = _backward_diff(u, -3), _backward_diff(v, -3)
    du_dy, du_dx = _central_diff(u, -2), _central_diff(u, -1)
    dv_dy, dv_dx = _central_diff(v, -2), _central_diff(v, -1)
    speed = torch.sqrt(u * u + v * v + eps)
    kinetic = 0.5 * (u * u + v * v)
    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy
    strain = torch.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2 + eps)
    x_coord, y_coord, t_coord = _coordinates(u)
    return torch.stack([
        u, v, p, speed, kinetic, du_dt, dv_dt, vorticity,
        divergence, strain, x_coord, y_coord, t_coord,
    ], dim=-1)


def future_linear_extrapolation(x: torch.Tensor, out_steps: int) -> torch.Tensor:
    raw = ensure_three_channels(x)
    last = raw[:, -1:]
    trend = raw[:, -1:] - raw[:, -2:-1] if raw.shape[1] > 1 else torch.zeros_like(last)
    steps = torch.linspace(
        1.0 / float(out_steps), 1.0, out_steps, device=x.device, dtype=x.dtype
    ).view(1, out_steps, 1, 1, 1)
    return zero_pressure(last + steps * trend)


def build_future_features(x: torch.Tensor, base_pred: torch.Tensor) -> torch.Tensor:
    base = zero_pressure(base_pred)
    out_steps = int(base.shape[1])
    last_raw = zero_pressure(ensure_three_channels(x[:, -1:])).expand(-1, out_steps, -1, -1, -1)
    linear = future_linear_extrapolation(x, out_steps)
    base_features = augment_torch(base)
    past_features = augment_torch(ensure_three_channels(x))
    last_features = past_features[:, -1:].expand(-1, out_steps, -1, -1, -1)
    return torch.cat([base_features, last_features, linear, base - last_raw, base - linear], dim=-1)


def norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(channels), channels), nn.SiLU(), nn.Dropout3d(float(dropout)),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(channels), channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


@dataclass(frozen=True)
class HeadConfig:
    hidden: int = 64
    blocks: int = 2
    dropout: float = 0.0
    include_pressure: bool = True
    history_context: bool = False
    sigma0: float = 0.02
    min_sigma: float = 1e-4
    max_sigma: float = 1.0
    include_delta: bool = False

    def validate(self) -> None:
        expected = {
            "hidden": 64, "blocks": 2, "dropout": 0.0,
            "include_pressure": True, "history_context": False,
            "sigma0": 0.02, "min_sigma": 1e-4, "max_sigma": 1.0,
            "include_delta": False,
        }
        for key, value in expected.items():
            if getattr(self, key) != value:
                raise ValueError(f"frozen SPS config mismatch: {key}")


class Head3D(nn.Module):
    IN_CHANNELS = 35

    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.input_norm = nn.LayerNorm(self.IN_CHANNELS)
        layers: list[nn.Module] = [
            nn.Conv3d(self.IN_CHANNELS, cfg.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(cfg.hidden), cfg.hidden), nn.SiLU(),
        ]
        for _ in range(cfg.blocks):
            layers.append(ResidualBlock3D(cfg.hidden, cfg.dropout))
        layers.append(nn.Conv3d(cfg.hidden, 2, kernel_size=1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(math.log(cfg.sigma0)))

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        features = build_future_features(x, base)
        if features.shape[-1] != self.IN_CHANNELS:
            raise RuntimeError(f"SPS feature count drift: {features.shape[-1]}")
        z = self.input_norm(features).permute(0, 4, 1, 2, 3).contiguous()
        out = self.net(z).permute(0, 2, 3, 4, 1).contiguous()
        return out.clamp(min=float(math.log(self.config.min_sigma)), max=float(math.log(self.config.max_sigma)))


def adaptive_half_width(prediction: torch.Tensor, sigma: torch.Tensor, *,
                        floor: float, mult_u: float, mult_v: float, rel: float) -> torch.Tensor:
    if prediction.shape[-1] != 3 or sigma.shape != prediction[..., :2].shape:
        raise ValueError("expected prediction [...,3] and sigma matching prediction[..., :2]")
    uv = torch.empty_like(sigma)
    uv[..., 0] = float(floor) + float(mult_u) * sigma[..., 0] + float(rel) * prediction[..., 0].abs()
    uv[..., 1] = float(floor) + float(mult_v) * sigma[..., 1] + float(rel) * prediction[..., 1].abs()
    return torch.cat([uv, torch.zeros_like(uv[..., :1])], dim=-1)