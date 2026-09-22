#!/usr/bin/env python3
"""Adapter exposing the frozen SOTA-V2 P0-A/MF backbone through raw Past20."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class SOTAV2MFBackboneAdapter(nn.Module):
    """Raw 3-channel Past20 -> frozen SOTA-V2 MF prediction."""

    def __init__(
        self,
        checkpoint: Path,
        kit_root: Path,
        device: torch.device,
    ) -> None:
        super().__init__()
        import realpde_sota_v2_integrated as sota_v2
        from realpde_sota_v3_residual import _load_backbone

        payload, config, builder, model = _load_backbone(
            checkpoint, kit_root, device
        )
        self.builder = builder
        self.model = model
        self._forward_mf = sota_v2.forward_mf
        self.checkpoint_iteration = int(payload.get("iteration", -1))
        self.feature_set = str(payload.get("feature_set", ""))
        self.feature_config = vars(config)

        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[-1] != 3:
            raise ValueError("SOTA-V2 adapter expects [B,T,H,W,3]")
        return self._forward_mf(self.model, self.builder, x)


def load_sota_v2_mf_backbone(
    checkpoint: Path,
    kit_root: Path,
    device: torch.device,
) -> nn.Module:
    model = SOTAV2MFBackboneAdapter(checkpoint, kit_root, device).to(device)
    model.eval()
    return model
