#!/usr/bin/env python3
"""Causal P0-A/P0-B feature construction for RealPDE Track 1.

The builder is deliberately pure PyTorch so exactly the same implementation can
run during training and inside an offline Codabench submission.  It accepts the
official Track 1 layout ``[B, T, H, W, C]`` and preserves its first three
channels (``u, v, p``).  P0-A uses only the supplied historical window; P0-B
requires explicit per-sample metadata and a registered coordinate grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


P0_A_NAMES: tuple[str, ...] = (
    "velocity_magnitude",
    "du_dx",
    "du_dy",
    "dv_dx",
    "dv_dy",
    "vorticity",
    "abs_vorticity",
    "strain_magnitude",
    "delta_u",
    "delta_v",
    "u_history_mean",
    "v_history_mean",
    "u_history_std",
    "v_history_std",
    "u_fluctuation",
    "v_fluctuation",
    "history_tke_proxy",
)

P0_B_NAMES: tuple[str, ...] = (
    "x_norm",
    "y_norm",
    "re_norm",
    "sin_aoa",
    "cos_aoa",
)


@dataclass(frozen=True)
class P0FeatureConfig:
    """Frozen feature semantics shared by train, validation, and inference."""

    include_p0_a: bool = True
    include_p0_b: bool = True
    dx: float = 1.0
    dy: float = 1.0
    dt: float | None = None
    re_center: float = 0.0
    re_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.dx == 0.0 or self.dy == 0.0:
            raise ValueError("dx and dy must be non-zero")
        if self.dt is not None and self.dt <= 0.0:
            raise ValueError("dt must be positive when supplied")
        if self.re_scale <= 0.0:
            raise ValueError("re_scale must be positive")


def _first_derivative(x: Tensor, *, dim: int, spacing: float) -> Tensor:
    """Second-order central / first-order one-sided finite difference."""

    if x.shape[dim] < 2:
        raise ValueError(f"need at least two cells along dimension {dim}")
    out = torch.empty_like(x)
    all_idx = [slice(None)] * x.ndim
    first, second = all_idx.copy(), all_idx.copy()
    first[dim], second[dim] = 0, 1
    out[tuple(first)] = (x[tuple(second)] - x[tuple(first)]) / spacing
    last, penultimate = all_idx.copy(), all_idx.copy()
    last[dim], penultimate[dim] = -1, -2
    out[tuple(last)] = (x[tuple(last)] - x[tuple(penultimate)]) / spacing
    if x.shape[dim] > 2:
        middle, right, left = all_idx.copy(), all_idx.copy(), all_idx.copy()
        middle[dim], right[dim], left[dim] = slice(1, -1), slice(2, None), slice(None, -2)
        out[tuple(middle)] = (x[tuple(right)] - x[tuple(left)]) / (2.0 * spacing)
    return out


class P0FeatureBuilder(nn.Module):
    """Append deterministic P0-A/P0-B channels to an official model input."""

    def __init__(self, config: P0FeatureConfig, x_grid: Tensor | None = None, y_grid: Tensor | None = None):
        super().__init__()
        self.config = config
        if (x_grid is None) != (y_grid is None):
            raise ValueError("x_grid and y_grid must either both be set or both be omitted")
        if config.include_p0_b and x_grid is None:
            raise ValueError("P0-B requires registered x_grid and y_grid")
        if x_grid is None:
            self.register_buffer("x_norm", torch.empty(0), persistent=True)
            self.register_buffer("y_norm", torch.empty(0), persistent=True)
        else:
            if x_grid.ndim != 2 or y_grid is None or y_grid.shape != x_grid.shape:
                raise ValueError("coordinate grids must be same-shape [H, W] tensors")
            self.register_buffer("x_norm", self._normalize_coordinate(x_grid), persistent=True)
            self.register_buffer("y_norm", self._normalize_coordinate(y_grid), persistent=True)

    @staticmethod
    def _normalize_coordinate(grid: Tensor) -> Tensor:
        grid = grid.detach().to(dtype=torch.float32)
        if not torch.isfinite(grid).all():
            raise ValueError("coordinate grid contains non-finite values")
        lo, hi = grid.amin(), grid.amax()
        if bool((hi - lo) <= torch.finfo(grid.dtype).eps):
            raise ValueError("coordinate grid must vary spatially")
        return 2.0 * (grid - lo) / (hi - lo) - 1.0

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = ["u", "v", "p"]
        if self.config.include_p0_a:
            names.extend(P0_A_NAMES)
        if self.config.include_p0_b:
            names.extend(P0_B_NAMES)
        return tuple(names)

    @staticmethod
    def _metadata_value(metadata: Mapping[str, Tensor | float | int], name: str, batch: int, ref: Tensor) -> Tensor:
        if name not in metadata:
            raise ValueError(f"P0-B requires metadata['{name}'] for every inference sample")
        value = torch.as_tensor(metadata[name], dtype=ref.dtype, device=ref.device).reshape(-1)
        if value.numel() == 1:
            value = value.expand(batch)
        if value.numel() != batch:
            raise ValueError(f"metadata['{name}'] must have one value per batch item, got {value.numel()} for {batch}")
        if not torch.isfinite(value).all():
            raise ValueError(f"metadata['{name}'] contains non-finite values")
        return value

    def forward(self, input_window: Tensor, metadata: Mapping[str, Tensor | float | int] | None = None) -> Tensor:
        if input_window.ndim != 5 or input_window.shape[-1] < 3:
            raise ValueError("expected input_window shaped [B, T, H, W, C>=3]")
        if not torch.is_floating_point(input_window):
            raise ValueError("input_window must be floating point")
        if not torch.isfinite(input_window[..., :3]).all():
            raise ValueError("input velocity channels contain non-finite values")

        batch, steps, height, width, _ = input_window.shape
        u, v = input_window[..., 0], input_window[..., 1]
        channels: list[Tensor] = [input_window[..., :3]]

        if self.config.include_p0_a:
            du_dx = _first_derivative(u, dim=-1, spacing=self.config.dx)
            du_dy = _first_derivative(u, dim=-2, spacing=self.config.dy)
            dv_dx = _first_derivative(v, dim=-1, spacing=self.config.dx)
            dv_dy = _first_derivative(v, dim=-2, spacing=self.config.dy)
            vorticity = dv_dx - du_dy
            strain_magnitude = torch.sqrt(du_dx.square() + dv_dy.square() + 0.5 * (du_dy + dv_dx).square())

            delta_u = torch.zeros_like(u)
            delta_v = torch.zeros_like(v)
            delta_u[:, 1:] = u[:, 1:] - u[:, :-1]
            delta_v[:, 1:] = v[:, 1:] - v[:, :-1]
            if self.config.dt is not None:
                delta_u = delta_u / self.config.dt
                delta_v = delta_v / self.config.dt

            u_mean, v_mean = u.mean(dim=1, keepdim=True), v.mean(dim=1, keepdim=True)
            u_var, v_var = u.var(dim=1, keepdim=True, unbiased=False), v.var(dim=1, keepdim=True, unbiased=False)
            u_std, v_std = torch.sqrt(u_var), torch.sqrt(v_var)
            history_tke = 0.5 * (u_var + v_var)
            broadcast = lambda z: z.expand(batch, steps, height, width)
            channels.extend([
                torch.sqrt(u.square() + v.square()),
                du_dx, du_dy, dv_dx, dv_dy,
                vorticity, vorticity.abs(), strain_magnitude,
                delta_u, delta_v,
                broadcast(u_mean), broadcast(v_mean),
                broadcast(u_std), broadcast(v_std),
                u - broadcast(u_mean), v - broadcast(v_mean),
                broadcast(history_tke),
            ])

        if self.config.include_p0_b:
            if metadata is None:
                raise ValueError("P0-B is enabled but metadata was not provided")
            if self.x_norm.shape != (height, width) or self.y_norm.shape != (height, width):
                raise ValueError(
                    f"registered grid {tuple(self.x_norm.shape)} does not match input spatial shape {(height, width)}"
                )
            re = self._metadata_value(metadata, "re", batch, input_window)
            aoa = self._metadata_value(metadata, "aoa", batch, input_window)
            static = lambda z: z.view(1, 1, height, width).expand(batch, steps, height, width)
            condition = lambda z: z.view(batch, 1, 1, 1).expand(batch, steps, height, width)
            radians = torch.deg2rad(aoa)
            channels.extend([
                static(self.x_norm), static(self.y_norm),
                condition((re - self.config.re_center) / self.config.re_scale),
                condition(torch.sin(radians)), condition(torch.cos(radians)),
            ])

        return torch.cat([c.unsqueeze(-1) if c.ndim == 4 else c for c in channels], dim=-1)


def make_p0_builder(config: P0FeatureConfig, x_grid: Tensor, y_grid: Tensor) -> P0FeatureBuilder:
    """Small explicit factory used by training and submission wrappers."""

    return P0FeatureBuilder(config=config, x_grid=x_grid, y_grid=y_grid)
