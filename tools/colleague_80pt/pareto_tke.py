#!/usr/bin/env python3
"""Conflict-aware TKE objective for the colleague-80 residual corrector.

The primary objective is exactly the existing residual objective with the TKE
term removed. The energy objective is the weighted TKE term. When the two
gradients conflict (negative dot product), only the TKE gradient is projected
onto the plane orthogonal to the primary gradient. The primary gradient is
never altered.

This module contains no data loading, checkpoint selection, or evaluation
logic. It is intentionally small so the experiment semantics are auditable.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from realpde_h5_feature_adapter_train import (
    kinetic_energy_torch,
    rel_l2_loss_torch,
)


def split_residual_objective(
    pred: Tensor,
    target: Tensor,
    base: Tensor,
    delta: Tensor,
    *,
    weights: dict[str, float],
    residual_mse_weight: float,
    delta_penalty_weight: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Return primary loss, weighted TKE loss, and tensor-valued terms.

    primary_loss contains every frozen colleague residual term except TKE.
    energy_loss is weights["tke"] * TKE. Therefore
    primary_loss + energy_loss is numerically the original scalar objective.
    """
    pred_uv = pred[..., :2]
    target_uv = target[..., :2]
    point = rel_l2_loss_torch(pred_uv, target_uv)
    mse = torch.mean((pred_uv - target_uv) ** 2)

    pred_ke = kinetic_energy_torch(pred_uv)
    target_ke = kinetic_energy_torch(target_uv)
    tke = rel_l2_loss_torch(pred_ke, target_ke)

    temporal = rel_l2_loss_torch(
        pred_uv[:, 1:] - pred_uv[:, :-1],
        target_uv[:, 1:] - target_uv[:, :-1],
    )
    pred_dx = pred_uv[:, :, :, 1:] - pred_uv[:, :, :, :-1]
    target_dx = target_uv[:, :, :, 1:] - target_uv[:, :, :, :-1]
    pred_dy = pred_uv[:, :, 1:, :] - pred_uv[:, :, :-1, :]
    target_dy = target_uv[:, :, 1:, :] - target_uv[:, :, :-1, :]
    grad = 0.5 * rel_l2_loss_torch(pred_dx, target_dx) + 0.5 * rel_l2_loss_torch(
        pred_dy, target_dy
    )
    p_zero = (
        torch.mean(pred[..., 2] ** 2)
        if pred.shape[-1] > 2
        else pred.new_tensor(0.0)
    )

    residual_target = target[..., :2] - base[..., :2]
    residual_mse = torch.mean((delta[..., :2] - residual_target) ** 2)
    delta_penalty = torch.mean(delta[..., :2] ** 2)

    primary_loss = (
        weights["point"] * point
        + weights["mse"] * mse
        + weights["temporal"] * temporal
        + weights["grad"] * grad
        + weights["p_zero"] * p_zero
        + float(residual_mse_weight) * residual_mse
        + float(delta_penalty_weight) * delta_penalty
    )
    energy_loss = weights["tke"] * tke
    terms = {
        "point_rel": point,
        "mse": mse,
        "tke_rel": tke,
        "temporal_rel": temporal,
        "grad_rel": grad,
        "p_zero": p_zero,
        "residual_mse": residual_mse,
        "delta_penalty": delta_penalty,
    }
    return primary_loss, energy_loss, terms


def _dot(
    left: tuple[Tensor | None, ...],
    right: tuple[Tensor | None, ...],
) -> Tensor:
    values = [
        (a * b).sum()
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None
    ]
    if not values:
        raise RuntimeError("no shared gradients for Pareto-TKE projection")
    return torch.stack(values).sum()


def _sq_norm(grads: tuple[Tensor | None, ...], reference: Tensor) -> Tensor:
    values = [g.square().sum() for g in grads if g is not None]
    if not values:
        return reference.new_tensor(0.0)
    return torch.stack(values).sum()


def project_tke_backward(
    primary_loss: Tensor,
    energy_loss: Tensor,
    parameters: Iterable[nn.Parameter],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Populate parameter gradients with primary + conflict-projected TKE.

    If dot(g_primary, g_tke) < 0:
        g_tke <- g_tke - dot(g_tke,g_primary)/||g_primary||^2 * g_primary

    Otherwise the ordinary summed gradient is used. The returned diagnostics
    describe the pre-projection conflict and the post-projection orthogonality.
    """
    params = tuple(p for p in parameters if p.requires_grad)
    if not params:
        raise ValueError("Pareto-TKE requires trainable parameters")

    primary_grads = torch.autograd.grad(
        primary_loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    energy_grads = torch.autograd.grad(
        energy_loss,
        params,
        retain_graph=False,
        allow_unused=True,
    )

    dot_before = _dot(primary_grads, energy_grads)
    primary_norm_sq = _sq_norm(primary_grads, dot_before)
    energy_norm_sq = _sq_norm(energy_grads, dot_before)
    conflict = bool((dot_before < 0).detach().cpu()) and bool(
        (primary_norm_sq > eps).detach().cpu()
    )

    coefficient = (
        dot_before / primary_norm_sq.clamp_min(eps)
        if conflict
        else dot_before.new_tensor(0.0)
    )
    projected_energy: list[Tensor | None] = []
    for gp, ge in zip(primary_grads, energy_grads, strict=True):
        if ge is None:
            projected_energy.append(None)
        elif conflict and gp is not None:
            projected_energy.append(ge - coefficient * gp)
        else:
            projected_energy.append(ge)

    projected_tuple = tuple(projected_energy)
    dot_after = _dot(primary_grads, projected_tuple)
    projected_norm_sq = _sq_norm(projected_tuple, dot_before)

    for parameter, gp, ge in zip(
        params, primary_grads, projected_tuple, strict=True
    ):
        if gp is None and ge is None:
            parameter.grad = None
        elif gp is None:
            parameter.grad = ge.detach().clone()
        elif ge is None:
            parameter.grad = gp.detach().clone()
        else:
            parameter.grad = (gp + ge).detach().clone()

    cosine = dot_before / (
        primary_norm_sq.sqrt() * energy_norm_sq.sqrt()
    ).clamp_min(eps)
    return {
        "pareto_conflict": 1.0 if conflict else 0.0,
        "pareto_grad_dot_before": float(dot_before.detach().cpu()),
        "pareto_grad_dot_after": float(dot_after.detach().cpu()),
        "pareto_grad_cosine_before": float(cosine.detach().cpu()),
        "pareto_primary_grad_norm": float(primary_norm_sq.sqrt().detach().cpu()),
        "pareto_tke_grad_norm": float(energy_norm_sq.sqrt().detach().cpu()),
        "pareto_projected_tke_grad_norm": float(
            projected_norm_sq.sqrt().detach().cpu()
        ),
    }
