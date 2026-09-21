import os
import traceback

import numpy as np


_MODEL = None
_MODEL_ERROR = None
_BOUND_ABS = 0.0075
_BOUND_REL = 0.0075
_CORRECTION_ALPHA = 1.0
_HIDDEN = 96
_BLOCKS = 2
_DROPOUT = 0.0
_INCLUDE_PRESSURE = True
_MAX_DELTA = 0.04
_UNC_FLOOR_U = 0.0025
_UNC_MULT_U = 1.25
_UNC_FLOOR_V = 0.0025
_UNC_MULT_V = 1.5
_UNC_REL = 0.005
_HEAD_HIDDEN = 64
_HEAD_BLOCKS = 2
_HEAD_DROPOUT = 0.0
_HEAD_INCLUDE_PRESSURE = True
_HEAD_MIN_SIGMA = 1e-4
_HEAD_MAX_SIGMA = 1.0
_HISTORY_CONTEXT = False


def _persistence(input_array):
    x = np.asarray(input_array)
    pred = np.repeat(x[:, -1:, :, :, :], 20, axis=1).astype(np.float32, copy=False)
    if pred.shape[-1] >= 3:
        pred[..., 2] = 0.0
    return pred


def _with_bounds(pred, sigma=None):
    pred = np.asarray(pred, dtype=np.float32)
    if sigma is None:
        half_width = (_BOUND_ABS + _BOUND_REL * np.abs(pred)).astype(np.float32)
    else:
        sigma = np.asarray(sigma, dtype=np.float32)
        half_width = np.zeros_like(pred, dtype=np.float32)
        half_width[..., 0] = _UNC_FLOOR_U + _UNC_MULT_U * sigma[..., 0] + _UNC_REL * np.abs(pred[..., 0])
        half_width[..., 1] = _UNC_FLOOR_V + _UNC_MULT_V * sigma[..., 1] + _UNC_REL * np.abs(pred[..., 1])
    if pred.shape[-1] >= 3:
        half_width[..., 2] = 0.0
    return {"prediction": pred, "lower": pred - half_width, "upper": pred + half_width}


def _load_state_dict_flexible(module, state):
    fixed = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        fixed[new_key] = value
    result = module.load_state_dict(fixed, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint did not match residual-corrected CNO; "
            f"missing_keys={missing[:20]}, unexpected_keys={unexpected[:20]}"
        )


def _load_model(device):
    import torch
    from torch import nn
    from rpde_baselines.cno import CNO3d
    from realpde_feature_engineering import (
        augment_torch,
        feature_names,
        future_context_feature_count,
        future_context_torch,
    )

    torch.backends.cudnn.benchmark = False

    def ensure_three_channels(x):
        if x.shape[-1] >= 3:
            return x[..., :3]
        return torch.cat([x[..., :2], torch.zeros_like(x[..., :1])], dim=-1)

    def zero_pressure(x):
        if x.shape[-1] < 3:
            return x
        y = x.clone()
        y[..., 2] = 0.0
        return y

    def future_linear_extrapolation(x, out_steps):
        raw = ensure_three_channels(x)
        last = raw[:, -1:]
        if raw.shape[1] > 1:
            trend = raw[:, -1:] - raw[:, -2:-1]
        else:
            trend = torch.zeros_like(last)
        steps = torch.linspace(
            1.0 / float(out_steps),
            1.0,
            out_steps,
            device=x.device,
            dtype=x.dtype,
        ).view(1, out_steps, 1, 1, 1)
        return zero_pressure(last + steps * trend)

    def future_feature_count(include_pressure, history_context):
        count = 2 * len(feature_names(include_pressure=include_pressure)) + 9
        if history_context:
            count += future_context_feature_count()
        return count

    def build_future_features(x, base_pred, include_pressure, history_context):
        base = zero_pressure(ensure_three_channels(base_pred))
        out_steps = int(base.shape[1])
        last_raw = ensure_three_channels(x[:, -1:]).expand(-1, out_steps, -1, -1, -1)
        last_raw = zero_pressure(last_raw)
        linear = future_linear_extrapolation(x, out_steps)
        base_features = augment_torch(base, include_pressure=include_pressure)
        past_features = augment_torch(ensure_three_channels(x), include_pressure=include_pressure)
        last_features = past_features[:, -1:].expand(-1, out_steps, -1, -1, -1)
        pieces = [base_features, last_features, linear, base - last_raw, base - linear]
        if history_context:
            pieces.append(future_context_torch(x, out_steps))
        return torch.cat(pieces, dim=-1)

    def norm_groups(channels):
        for groups in (8, 4, 2):
            if channels % groups == 0:
                return groups
        return 1

    class ResidualBlock3D(nn.Module):
        def __init__(self, channels, dropout=0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(norm_groups(channels), channels),
                nn.SiLU(),
                nn.Dropout3d(float(dropout)),
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(norm_groups(channels), channels),
            )
            self.act = nn.SiLU()

        def forward(self, x):
            return self.act(x + self.net(x))

    class ResidualCorrector3D(nn.Module):
        def __init__(self):
            super().__init__()
            in_channels = future_feature_count(
                include_pressure=_INCLUDE_PRESSURE,
                history_context=_HISTORY_CONTEXT,
            )
            self.input_norm = nn.LayerNorm(in_channels)
            layers = [
                nn.Conv3d(in_channels, _HIDDEN, kernel_size=3, padding=1),
                nn.GroupNorm(norm_groups(_HIDDEN), _HIDDEN),
                nn.SiLU(),
            ]
            for _ in range(_BLOCKS):
                layers.append(ResidualBlock3D(_HIDDEN, dropout=_DROPOUT))
            layers.append(nn.Conv3d(_HIDDEN, 3, kernel_size=1))
            self.net = nn.Sequential(*layers)

        def forward(self, x, base_pred):
            features = build_future_features(
                x,
                base_pred,
                include_pressure=_INCLUDE_PRESSURE,
                history_context=_HISTORY_CONTEXT,
            )
            features = self.input_norm(features)
            z = features.permute(0, 4, 1, 2, 3).contiguous()
            raw_delta = self.net(z).permute(0, 2, 3, 4, 1).contiguous()
            if _MAX_DELTA > 0:
                delta = _MAX_DELTA * torch.tanh(raw_delta / _MAX_DELTA)
            else:
                delta = raw_delta
            delta = delta.clone()
            delta[..., 2] = 0.0
            return delta

    class UncertaintyHead3D(nn.Module):
        def __init__(self, hidden, blocks, dropout, include_pressure):
            super().__init__()
            in_channels = future_feature_count(include_pressure=include_pressure, history_context=False)
            self.input_norm = nn.LayerNorm(in_channels)
            layers = [
                nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(norm_groups(hidden), hidden),
                nn.SiLU(),
            ]
            for _ in range(blocks):
                layers.append(ResidualBlock3D(hidden, dropout=dropout))
            layers.append(nn.Conv3d(hidden, 2, kernel_size=1))
            self.net = nn.Sequential(*layers)

        def forward(self, x, base_pred):
            features = build_future_features(x, base_pred, include_pressure=_HEAD_INCLUDE_PRESSURE, history_context=False)
            features = self.input_norm(features)
            z = features.permute(0, 4, 1, 2, 3).contiguous()
            log_std = self.net(z).permute(0, 2, 3, 4, 1).contiguous()
            log_std = log_std.clamp(min=float(np.log(max(_HEAD_MIN_SIGMA, 1e-6))),
                                    max=float(np.log(max(_HEAD_MAX_SIGMA, _HEAD_MIN_SIGMA))))
            return log_std

    class ResidualCorrectionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = CNO3d(
                in_dim=3,
                out_dim=3,
                out_dim_mult=1,
                in_size=64,
                N_layers=3,
                activation="LeakyReLU",
            )
            self.corrector = ResidualCorrector3D()

        def forward(self, x):
            x = ensure_three_channels(x)
            base = self.base_model(x)
            base = zero_pressure(ensure_three_channels(base))
            delta = self.corrector(x, base)
            pred = base + _CORRECTION_ALPHA * delta
            return zero_pressure(pred)

    class ProbeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rcm = ResidualCorrectionModel()
            self.head = UncertaintyHead3D(_HEAD_HIDDEN, _HEAD_BLOCKS, _HEAD_DROPOUT, _HEAD_INCLUDE_PRESSURE)

        def forward(self, x):
            x = ensure_three_channels(x)
            base = self.rcm.base_model(x.half()).float()
            base = zero_pressure(ensure_three_channels(base))
            delta = self.rcm.corrector(x, base)
            pred = zero_pressure(base + _CORRECTION_ALPHA * delta)
            log_std = self.head(x, base)
            return pred, log_std

    submission_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(submission_dir, "model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = ProbeModel().to(device)
    _load_state_dict_flexible(model.rcm, state)
    head_path = os.path.join(submission_dir, "uncertainty_head.pt")
    head_ckpt = torch.load(head_path, map_location=device)
    head_state = head_ckpt.get("head_state_dict", head_ckpt) if isinstance(head_ckpt, dict) else head_ckpt
    _load_state_dict_flexible(model.head, head_state)
    model.rcm.base_model = model.rcm.base_model.half()
    model.eval()
    return model


def predict(input_array, metadata=None):
    global _MODEL, _MODEL_ERROR
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if _MODEL is None and _MODEL_ERROR is None:
            try:
                _MODEL = _load_model(device)
            except Exception:
                _MODEL_ERROR = traceback.format_exc()
                print("residual-corrected CNO failed to load; falling back to persistence.")
                print(_MODEL_ERROR)

        if _MODEL is None:
            return _with_bounds(_persistence(input_array))

        x = np.asarray(input_array, dtype=np.float32)
        with torch.inference_mode():
            tensor = torch.from_numpy(x).to(device)
            pred, log_std = _MODEL(tensor)
            pred = pred.detach().cpu().numpy().astype(np.float32, copy=False)
            sigma = torch.exp(log_std).detach().cpu().numpy().astype(np.float32, copy=False)

        if pred.shape != (x.shape[0], 20, 32, 64, 3):
            print(f"probe model returned unexpected shape {pred.shape}; falling back to persistence.")
            return _with_bounds(_persistence(input_array))
        pred[..., 2] = 0.0
        if not np.isfinite(pred).all() or not np.isfinite(sigma).all():
            print("probe model returned non-finite values; falling back to persistence.")
            return _with_bounds(_persistence(input_array))
        return _with_bounds(pred, sigma)

    except Exception:
        print("predict() failed; falling back to persistence.")
        print(traceback.format_exc())
        return _with_bounds(_persistence(input_array))
