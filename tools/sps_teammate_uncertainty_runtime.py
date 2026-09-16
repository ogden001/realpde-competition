#!/usr/bin/env python3
"""Teammate-style 35-channel uncertainty features and head for Track 1 SPS.

The inference architecture and feature recipe are reproduced from
``submission_all81_probe_fp16_20260915.zip``.  The head consumes 35
future-aligned channels: 13 augmented prediction channels, 13 augmented last
observation channels, linear extrapolation (3), prediction-last (3), and
prediction-linear (3).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

MIN_SIGMA = 1e-4
MAX_SIGMA = 1.0


def _ensure_three_channels(x: Tensor) -> Tensor:
    if x.shape[-1] >= 3:
        return x[..., :3]
    return torch.cat([x[..., :2], torch.zeros_like(x[..., :1])], dim=-1)


def _zero_pressure(x: Tensor) -> Tensor:
    if x.shape[-1] < 3:
        return x
    y = x.clone()
    y[..., 2] = 0.0
    return y


def _central_diff(value: Tensor, axis: int) -> Tensor:
    axis %= value.ndim
    out = torch.empty_like(value)
    if value.shape[axis] <= 1:
        out.zero_()
        return out
    middle = [slice(None)] * value.ndim
    before = [slice(None)] * value.ndim
    after = [slice(None)] * value.ndim
    middle[axis], before[axis], after[axis] = slice(1, -1), slice(None, -2), slice(2, None)
    out[tuple(middle)] = 0.5 * (value[tuple(after)] - value[tuple(before)])
    first = [slice(None)] * value.ndim
    second = [slice(None)] * value.ndim
    first[axis], second[axis] = 0, 1
    out[tuple(first)] = value[tuple(second)] - value[tuple(first)]
    last = [slice(None)] * value.ndim
    previous = [slice(None)] * value.ndim
    last[axis], previous[axis] = -1, -2
    out[tuple(last)] = value[tuple(last)] - value[tuple(previous)]
    return out


def _backward_diff(value: Tensor, axis: int) -> Tensor:
    axis %= value.ndim
    out = torch.zeros_like(value)
    if value.shape[axis] <= 1:
        return out
    current = [slice(None)] * value.ndim
    previous = [slice(None)] * value.ndim
    current[axis], previous[axis] = slice(1, None), slice(None, -1)
    out[tuple(current)] = value[tuple(current)] - value[tuple(previous)]
    return out


def _coordinate_features(u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if u.ndim != 4:
        raise ValueError("u must be [B,T,H,W]")
    _, steps, height, width = u.shape
    x = torch.linspace(-1.0, 1.0, width, device=u.device, dtype=u.dtype).view(1, 1, 1, width).expand_as(u)
    y = torch.linspace(-1.0, 1.0, height, device=u.device, dtype=u.dtype).view(1, 1, height, 1).expand_as(u)
    t = torch.linspace(-1.0, 1.0, steps, device=u.device, dtype=u.dtype).view(1, steps, 1, 1).expand_as(u)
    return x, y, t


def augment_torch(x: Tensor, *, include_pressure: bool = True, eps: float = 1e-6) -> Tensor:
    """Exact 13-channel augmentation used by the teammate package."""
    if x.ndim != 5 or x.shape[-1] < 2:
        raise ValueError("x must be [B,T,H,W,C>=2]")
    if x.dtype != torch.float32:
        x = x.float()
    u, v = x[..., 0], x[..., 1]
    p = x[..., 2] if x.shape[-1] >= 3 else torch.zeros_like(u)
    du_dt, dv_dt = _backward_diff(u, -3), _backward_diff(v, -3)
    du_dy, du_dx = _central_diff(u, -2), _central_diff(u, -1)
    dv_dy, dv_dx = _central_diff(v, -2), _central_diff(v, -1)
    speed = torch.sqrt(u * u + v * v + eps)
    kinetic = 0.5 * (u * u + v * v)
    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy
    strain = torch.sqrt((du_dx - dv_dy).square() + (du_dy + dv_dx).square() + eps)
    x_coord, y_coord, t_coord = _coordinate_features(u)
    channels = [u, v]
    if include_pressure:
        channels.append(p)
    channels.extend([speed, kinetic, du_dt, dv_dt, vorticity, divergence, strain, x_coord, y_coord, t_coord])
    return torch.stack(channels, dim=-1)


def future_feature_count(*, include_pressure: bool = True) -> int:
    augmented = 13 if include_pressure else 12
    return 2 * augmented + 9


def future_linear_extrapolation(past: Tensor, out_steps: int) -> Tensor:
    raw = _ensure_three_channels(past)
    last = raw[:, -1:]
    trend = raw[:, -1:] - raw[:, -2:-1] if raw.shape[1] > 1 else torch.zeros_like(last)
    steps = torch.linspace(
        1.0 / float(out_steps), 1.0, out_steps, device=past.device, dtype=past.dtype
    ).view(1, out_steps, 1, 1, 1)
    return _zero_pressure(last + steps * trend)


def teammate_future_features(
    past: Tensor,
    base_prediction: Tensor,
    *,
    include_pressure: bool = True,
) -> Tensor:
    """Build the teammate's future-aligned 35-channel uncertainty features."""
    if past.ndim != 5 or base_prediction.ndim != 5:
        raise ValueError("past/base_prediction must be rank-5")
    if past.shape[0] != base_prediction.shape[0] or past.shape[2:4] != base_prediction.shape[2:4]:
        raise ValueError("past/base_prediction geometry mismatch")
    base = _zero_pressure(_ensure_three_channels(base_prediction))
    out_steps = int(base.shape[1])
    last_raw = _zero_pressure(_ensure_three_channels(past[:, -1:]).expand(-1, out_steps, -1, -1, -1))
    linear = future_linear_extrapolation(past, out_steps)
    base_features = augment_torch(base, include_pressure=include_pressure)
    past_features = augment_torch(_ensure_three_channels(past), include_pressure=include_pressure)
    last_features = past_features[:, -1:].expand(-1, out_steps, -1, -1, -1)
    features = torch.cat([base_features, last_features, linear, base - last_raw, base - linear], dim=-1)
    expected = future_feature_count(include_pressure=include_pressure)
    if features.shape[-1] != expected:
        raise RuntimeError(f"teammate feature count {features.shape[-1]} != {expected}")
    return features


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.SiLU(),
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(_norm_groups(channels), channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class TeammateUncertaintyHead(nn.Module):
    """Exact inference architecture extracted from the teammate submission."""

    def __init__(
        self,
        hidden: int = 32,
        blocks: int = 2,
        dropout: float = 0.0,
        include_pressure: bool = True,
    ):
        super().__init__()
        if hidden < 1 or blocks < 1:
            raise ValueError("hidden/blocks must be positive")
        self.include_pressure = bool(include_pressure)
        in_channels = future_feature_count(include_pressure=self.include_pressure)
        self.input_norm = nn.LayerNorm(in_channels)
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(_norm_groups(hidden), hidden),
            nn.SiLU(),
        ]
        for _ in range(blocks):
            layers.append(_ResidualBlock3D(hidden, dropout=dropout))
        layers.append(nn.Conv3d(hidden, 2, 1))
        self.net = nn.Sequential(*layers)

    def forward_features(self, features: Tensor) -> Tensor:
        expected = future_feature_count(include_pressure=self.include_pressure)
        if features.ndim != 5 or features.shape[-1] != expected:
            raise ValueError(f"features must be [B,T,H,W,{expected}]")
        normalized = self.input_norm(features)
        z = normalized.permute(0, 4, 1, 2, 3).contiguous()
        return self.net(z).permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, past: Tensor, base_prediction: Tensor) -> Tensor:
        features = teammate_future_features(
            past, base_prediction, include_pressure=self.include_pressure
        )
        return self.forward_features(features)


def clamp_log_std(
    log_std: Tensor,
    min_sigma: float = MIN_SIGMA,
    max_sigma: float = MAX_SIGMA,
) -> Tensor:
    if min_sigma <= 0 or max_sigma < min_sigma:
        raise ValueError("invalid sigma bounds")
    return log_std.clamp(min=math.log(min_sigma), max=math.log(max_sigma))


def sigma_from_log_std(
    log_std: Tensor,
    min_sigma: float = MIN_SIGMA,
    max_sigma: float = MAX_SIGMA,
) -> Tensor:
    return torch.exp(clamp_log_std(log_std, min_sigma=min_sigma, max_sigma=max_sigma))


def gaussian_nll_from_log_std(
    target_uv: Tensor,
    prediction_uv: Tensor,
    log_std_uv: Tensor,
) -> Tensor:
    if target_uv.shape != prediction_uv.shape or log_std_uv.shape != target_uv.shape:
        raise ValueError("target/prediction/log_std shapes are incompatible")
    bounded = clamp_log_std(log_std_uv)
    inv_sigma = torch.exp(-bounded)
    return (bounded + 0.5 * ((target_uv - prediction_uv) * inv_sigma).square()).mean()
