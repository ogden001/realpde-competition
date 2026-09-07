from __future__ import annotations

import torch
from torch import Tensor


def normalized_tail_weights(
    tail_factor: float,
    *,
    horizons: int = 20,
    tail_count: int = 2,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if horizons < 1:
        raise ValueError("horizons must be positive")
    if not 1 <= tail_count <= horizons:
        raise ValueError("tail_count must be within [1, horizons]")
    if tail_factor <= 0:
        raise ValueError("tail_factor must be positive")
    weights = torch.ones(horizons, device=device, dtype=dtype or torch.float32)
    weights[-tail_count:] = float(tail_factor)
    return weights / weights.mean()


def weighted_mse(pred: Tensor, target: Tensor, *, tail_factor: float, tail_count: int = 2) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim < 2:
        raise ValueError("prediction must include batch and horizon dimensions")
    weights = normalized_tail_weights(
        tail_factor,
        horizons=pred.shape[1],
        tail_count=tail_count,
        device=pred.device,
        dtype=pred.dtype,
    )
    view_shape = [1, pred.shape[1]] + [1] * (pred.ndim - 2)
    return ((pred - target).square() * weights.view(*view_shape)).mean()


def blend_tail(cno: Tensor, persist: Tensor, *, alpha19: float, alpha20: float) -> Tensor:
    if cno.shape != persist.shape:
        raise ValueError(f"CNO/PERSIST shape mismatch: {tuple(cno.shape)} vs {tuple(persist.shape)}")
    if cno.ndim < 2 or cno.shape[1] != 20:
        raise ValueError(f"expected Future20 tensors, got shape {tuple(cno.shape)}")
    if not 0.0 <= alpha19 <= 1.0 or not 0.0 <= alpha20 <= 1.0:
        raise ValueError("alpha19/alpha20 must lie in [0, 1]")
    out = cno.clone()
    out[:, 18] = float(alpha19) * cno[:, 18] + (1.0 - float(alpha19)) * persist[:, 18]
    out[:, 19] = float(alpha20) * cno[:, 19] + (1.0 - float(alpha20)) * persist[:, 19]
    return out
