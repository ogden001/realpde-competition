#!/usr/bin/env python3
"""Inference primitives for SOTA-V2 MF-CNO + adaptive uncertainty."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class _AdaptiveBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.net(value)


class AdaptiveUncertaintyHead(nn.Module):
    """Frozen v5 uncertainty architecture, freshly trained for SOTA-V2."""

    def __init__(self, in_channels: int = 15, hidden: int = 32, blocks: int = 2):
        super().__init__()
        if in_channels != 15 or hidden < 1 or blocks < 1:
            raise ValueError("invalid adaptive-head configuration")
        self.input = nn.Conv3d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*(_AdaptiveBlock3D(hidden) for _ in range(blocks)))
        self.output = nn.Conv3d(hidden, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, math.log(0.02))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 5 or features.shape[1] != 15:
            raise ValueError("uncertainty input must be [B,15,T,H,W]")
        raw = self.output(self.blocks(torch.nn.functional.gelu(self.input(features))))
        return torch.exp(raw).clamp(1e-4, 1.0)


def _diff(value: Tensor, dim: int) -> Tensor:
    result = torch.empty_like(value)
    first, second = [slice(None)] * value.ndim, [slice(None)] * value.ndim
    first[dim], second[dim] = 0, 1
    result[tuple(first)] = value[tuple(second)] - value[tuple(first)]
    last, before = [slice(None)] * value.ndim, [slice(None)] * value.ndim
    last[dim], before[dim] = -1, -2
    result[tuple(last)] = value[tuple(last)] - value[tuple(before)]
    if value.shape[dim] > 2:
        middle, right, left = [slice(None)] * value.ndim, [slice(None)] * value.ndim, [slice(None)] * value.ndim
        middle[dim], right[dim], left[dim] = slice(1, -1), slice(2, None), slice(None, -2)
        result[tuple(middle)] = (value[tuple(right)] - value[tuple(left)]) * 0.5
    return result


def flow_features(flow: Tensor) -> Tensor:
    """Frozen 12-channel flow features from the previous online-valid v5 head."""
    if flow.ndim != 5 or flow.shape[-1] < 2 or min(flow.shape[-3:-1]) < 2:
        raise ValueError("flow must be [B,T,H,W,C>=2] with spatial dims >=2")
    u, v = flow[..., 0], flow[..., 1]
    du_dt, dv_dt = torch.zeros_like(u), torch.zeros_like(v)
    du_dt[:, 1:], dv_dt[:, 1:] = u[:, 1:] - u[:, :-1], v[:, 1:] - v[:, :-1]
    du_dy, du_dx = _diff(u, -2), _diff(u, -1)
    dv_dy, dv_dx = _diff(v, -2), _diff(v, -1)
    vorticity, divergence = dv_dx - du_dy, du_dx + dv_dy
    strain = torch.sqrt(du_dx.square() + dv_dy.square() + 0.5 * (du_dy + dv_dx).square())
    _, steps, height, width = u.shape
    x = torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype).view(1, 1, 1, width).expand_as(u)
    y = torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype).view(1, 1, height, 1).expand_as(u)
    time_index = torch.linspace(-1.0, 1.0, steps, device=flow.device, dtype=flow.dtype).view(1, steps, 1, 1).expand_as(u)
    extras = [torch.sqrt(u.square() + v.square()), 0.5 * (u.square() + v.square()), du_dt, dv_dt,
              vorticity, divergence, strain, x, y, time_index]
    return torch.cat([flow[..., :2], *(item.unsqueeze(-1) for item in extras)], dim=-1)


def mf_reconstruct(raw: Tensor) -> Tensor:
    """Frozen SOTA-V2 MF head semantics; returned pressure is exactly zero."""
    if raw.ndim != 5 or raw.shape[-1] != 5:
        raise ValueError("MF raw output must be [B,T,H,W,5]")
    mean_raw, fluct_raw = raw[..., :2], raw[..., 2:4]
    mean_field = mean_raw.mean(dim=1, keepdim=True).expand_as(mean_raw)
    fluct = fluct_raw - fluct_raw.mean(dim=1, keepdim=True)
    uv = mean_field + fluct
    return torch.cat([uv, torch.zeros_like(raw[..., 4:5])], dim=-1)


def mf_forward(cno: nn.Module, builder: nn.Module, past: Tensor) -> Tensor:
    if past.ndim != 5 or past.shape[-1] != 3:
        raise ValueError("past must be [B,T,H,W,3]")
    features = builder(past)
    raw = cno(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
    return mf_reconstruct(raw)


def adaptive_bounds(prediction: Tensor, sigma_uv: Tensor, *, floor: float, mult: float) -> tuple[Tensor, Tensor]:
    if prediction.ndim != 5 or prediction.shape[-1] != 3 or sigma_uv.shape != prediction.shape[:-1] + (2,):
        raise ValueError("prediction/sigma shapes are incompatible")
    if floor < 0 or mult < 0:
        raise ValueError("bound parameters must be non-negative")
    half_uv = floor + mult * sigma_uv
    half = torch.cat([half_uv, torch.zeros_like(half_uv[..., :1])], dim=-1)
    return prediction - half, prediction + half
