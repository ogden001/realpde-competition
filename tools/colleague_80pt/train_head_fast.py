#!/usr/bin/env python3
"""Train the per-element uncertainty head on cached (x, base) tensors.

Reads /runs/frozen_cache/frozen_{tr,va}_{x,b,y}.npy (see cache_frozen.py) so the
frozen champion never has to run again: each update is a small 3D-conv forward /
backward on the head only.  This makes it practical to sweep head size, feature
set, loss and training length instead of running one fixed configuration.

Losses:
  nll      0.5*((e/sigma)^2 + 2 log sigma)        (the historical baseline)
  pinball  quantile loss of |e| against sigma
  logmae   |log sigma - log |e||                  (multiplicative accuracy)
  mae      |sigma - |e||
  sps      -reward*exp(-2h/sigma_g)*softstep(h-|e|) with h=sigma
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/repo/tools")
import realpde_sps_scoring as SPS  # noqa: E402
from residual_multi import (  # noqa: E402
    ResidualBlock3D,
    build_future_features,
    future_feature_count,
    norm_groups,
)

CACHE = Path("/runs/frozen_cache")
SIGMA_G = SPS.SIGMA_GLOBAL

FLOORS = (0.0, 0.0025, 0.005, 0.0075)
MULTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class HeadConfig:
    hidden: int = 32
    blocks: int = 2
    dropout: float = 0.0
    include_pressure: bool = True
    history_context: bool = False
    sigma0: float = 0.02
    min_sigma: float = 1e-4
    max_sigma: float = 1.0
    include_delta: bool = False


def build_head_features(
    x: torch.Tensor,
    base: torch.Tensor,
    corrected: torch.Tensor | None,
    *,
    include_pressure: bool,
    include_delta: bool,
    history_context: bool = False,
) -> torch.Tensor:
    features = build_future_features(
        x,
        base,
        include_pressure=include_pressure,
        history_context=history_context,
    )
    if not include_delta:
        return features
    if corrected is None:
        raise ValueError("corrected prediction is required when include_delta=True")
    return torch.cat([features, corrected[..., :2] - base[..., :2]], dim=-1)


class Head3D(nn.Module):
    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        self.config = cfg
        in_ch = future_feature_count(
            include_pressure=cfg.include_pressure, history_context=cfg.history_context
        )
        if cfg.include_delta:
            in_ch += 2
        self.in_channels = in_ch
        self.input_norm = nn.LayerNorm(in_ch)
        layers: list[nn.Module] = [
            nn.Conv3d(in_ch, cfg.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(cfg.hidden), cfg.hidden),
            nn.SiLU(),
        ]
        for _ in range(int(cfg.blocks)):
            layers.append(ResidualBlock3D(cfg.hidden, dropout=cfg.dropout))
        layers.append(nn.Conv3d(cfg.hidden, 2, kernel_size=1))
        self.net = nn.Sequential(*layers)
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, float(np.log(max(cfg.sigma0, 1e-4))))

    def forward(
        self,
        x: torch.Tensor,
        base: torch.Tensor,
        corrected: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        f = build_head_features(
            x,
            base,
            corrected,
            include_pressure=cfg.include_pressure,
            include_delta=cfg.include_delta,
            history_context=cfg.history_context,
        )
        f = self.input_norm(f)
        z = f.permute(0, 4, 1, 2, 3).contiguous()
        out = self.net(z).permute(0, 2, 3, 4, 1).contiguous()
        return out.clamp(
            min=float(np.log(max(cfg.min_sigma, 1e-6))),
            max=float(np.log(max(cfg.max_sigma, cfg.min_sigma))),
        )


def initialize_coupled(head: Head3D, *, seed: int) -> None:
    """Initialize H0/H1 shared weights identically; delta-only input weights are zero."""
    base_channels = future_feature_count(
        include_pressure=head.config.include_pressure,
        history_context=head.config.history_context,
    )
    final_layer = head.net[-1]
    for name, module in head.named_modules():
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv3d):
            if module is final_layer:
                nn.init.zeros_(module.weight)
                nn.init.constant_(module.bias, float(np.log(max(head.config.sigma0, 1e-4))))
                continue
            name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
            generator = torch.Generator(device=module.weight.device)
            generator.manual_seed((int(seed) + name_seed) % (2**63 - 1))
            if name == "net.0":
                nn.init.zeros_(module.weight)
                nn.init.kaiming_uniform_(module.weight[:, :base_channels], a=np.sqrt(5), generator=generator)
                fan_in = base_channels * int(np.prod(module.kernel_size))
            else:
                nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5), generator=generator)
                fan_in = int(module.in_channels * np.prod(module.kernel_size))
            if module.bias is not None:
                bound = 1.0 / np.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound, generator=generator)


def load_split(tag: str, cache_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = (cache_dir or CACHE) / f"frozen_{tag}"
    x = np.load(str(p) + "_x.npy", mmap_mode="r")
    b = np.load(str(p) + "_b.npy", mmap_mode="r")
    pr = np.load(str(p) + "_p.npy", mmap_mode="r")
    y = np.load(str(p) + "_y.npy", mmap_mode="r")
    return x, b, pr, y


def to_torch(a: np.ndarray, idx: np.ndarray, device) -> torch.Tensor:
    out = torch.from_numpy(np.asarray(a[idx], dtype=np.float32))
    return out.to(device, non_blocking=True)


@torch.no_grad()
def predict_sigma(head, xs, bs, ps, device, batch=64):
    head.eval()
    sig = np.empty((xs.shape[0], 20, 32, 64, 2), dtype=np.float32)
    for s in range(0, xs.shape[0], batch):
        sl = np.arange(s, min(s + batch, xs.shape[0]))
        x = to_torch(xs, sl, device)
        b = to_torch(bs, sl, device)
        p = to_torch(ps, sl, device)
        sig[sl] = torch.exp(head(x, b, p)).float().cpu().numpy()
    head.train()
    return sig


def fast_sps(pred, target, c, h, rewards, chunk=64):
    t = np.asarray(target[..., :c], dtype=np.float32)
    lo = np.asarray(pred[..., :c] - h, dtype=np.float32)
    hi = np.asarray(pred[..., :c] + h, dtype=np.float32)
    scored = t != 0.0
    n = int(np.count_nonzero(scored))
    total = {k: 0.0 for k in rewards}
    inside_n = 0
    for s in range(0, pred.shape[0], chunk):
        sl = slice(s, s + chunk)
        nil = (hi[sl] - lo[sl]) / SIGMA_G
        ins = (t[sl] >= lo[sl]) & (t[sl] <= hi[sl])
        inside_n += int(np.count_nonzero(ins & scored[sl]))
        pen = np.exp(-nil, out=nil) * ins
        pen *= scored[sl]
        for k, r in rewards.items():
            total[k] += float(np.sum(r[sl][:, None, None, None, None] * pen))
    sps = (0.5 * total["dm"] + 0.3 * total["tke"] + 0.2 * total["mvpe"]) / n
    return float(sps) * 100.0, inside_n / n


def calibrate(pred, target, sig, floors=FLOORS, mults=MULTS):
    c = SPS.measured_channels(target)
    dm = SPS.rel_l2_per_sample(pred, target, c)
    tke = SPS.tke_rel_l2_per_sample(pred, target, c)
    mvpe = SPS.mvpe_rel_l2_per_sample(pred, target)
    rewards = {"dm": SPS._acc_factor(dm), "tke": SPS._acc_factor(tke), "mvpe": SPS._acc_factor(mvpe)}
    best = (-1.0, None, None, None)
    for f in floors:
        for m in mults:
            sps, cov = fast_sps(pred, target, c, f + m * sig, rewards)
            if sps > best[0]:
                best = (sps, f, m, cov)
    return {"sps": round(best[0], 4), "floor": best[1], "mult": best[2], "coverage": round(best[3], 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--history-context", action="store_true")
    ap.add_argument("--include-delta", action="store_true")
    ap.add_argument("--loss", default="nll", choices=["nll", "pinball", "logmae", "mae", "sps"])
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--smooth", type=float, default=0.002)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", type=str, default="/runs/head_fast_run")
    ap.add_argument("--init-from", type=str, default=None)
    ap.add_argument("--reward-weighted", action="store_true", help="weight the sps loss per window by the true reward")
    ap.add_argument("--cache", type=str, default=str(CACHE))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = Path(args.cache)
    xtr, btr, ptr, ytr = load_split("tr", cache_dir)
    xva, bva, pva, yva = load_split("va", cache_dir)
    n_tr, n_va = xtr.shape[0], xva.shape[0]
    print(f"[head] train {n_tr} val {n_va}", flush=True)

    cfg = HeadConfig(
        hidden=args.hidden,
        blocks=args.blocks,
        dropout=args.dropout,
        history_context=args.history_context,
        include_delta=args.include_delta,
    )
    head = Head3D(cfg).to(device)
    initialize_coupled(head, seed=args.seed)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        state = ck["head_state_dict"] if isinstance(ck, dict) and "head_state_dict" in ck else ck
        head.load_state_dict(state)
        print(f"[head] init from {args.init_from}", flush=True)
    params = sum(p.numel() for p in head.parameters())
    print(f"[head] params {params} in_channels {head.in_channels}", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = max(args.updates, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(args.warmup, 1)) * (0.5 * (1 + np.cos(np.pi * min(1.0, s / total)))),
    )

    pred_va = np.asarray(pva, dtype=np.float32).copy()
    pred_va[..., 2:] = 0.0
    tgt_va = np.asarray(yva, dtype=np.float32)

    # per-window SPS reward (only used by the reward-weighted surrogate loss)
    c_ = SPS.measured_channels(np.asarray(ytr, dtype=np.float32)[:64])
    ytr_np = np.asarray(ytr, dtype=np.float32)
    ptr_np = np.asarray(ptr, dtype=np.float32)
    ptr_np[..., 2:] = 0.0
    rew = (
        0.5 * SPS._acc_factor(SPS.rel_l2_per_sample(ptr_np, ytr_np, c_))
        + 0.3 * SPS._acc_factor(SPS.tke_rel_l2_per_sample(ptr_np, ytr_np, c_))
        + 0.2 * SPS._acc_factor(SPS.mvpe_rel_l2_per_sample(ptr_np, ytr_np))
    ).astype(np.float32)
    print(f"[head] reward mean {float(rew.mean()):.4f} min {float(rew.min()):.4f}", flush=True)

    log = []
    step = 0
    t0 = time.time()

    def evaluate(step_idx: int) -> None:
        sig = predict_sigma(head, xva, bva, pva, device)
        np.save(out / f"sigma_val_{step_idx}.npy", sig.astype(np.float32))
        ev = calibrate(pred_va, tgt_va, sig)
        ev["step"] = step_idx
        log.append(ev)
        print(f"[head] EVAL step={step_idx} " + json.dumps(ev), flush=True)
        (out / "eval_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    evaluate(0)
    while step < args.updates:
        idx = np.random.randint(0, n_tr, size=args.batch)
        x = to_torch(xtr, idx, device)
        b = to_torch(btr, idx, device)
        pf = to_torch(ptr, idx, device)
        y = to_torch(ytr, idx, device)
        log_std = head(x, b, pf)
        err = (pf[..., :2] - y[..., :2]).abs()
        mask = (y[..., :2] != 0.0).float()
        denom = mask.sum().clamp_min(1.0)
        if args.loss == "nll":
            sig = torch.exp(log_std)
            e = (y[..., :2] - pf[..., :2])
            per = 0.5 * ((e / sig) ** 2 + 2.0 * log_std)
        elif args.loss == "pinball":
            sig = torch.exp(log_std)
            d = err - sig
            per = torch.maximum(args.tau * d, (args.tau - 1.0) * d)
        elif args.loss == "logmae":
            per = (log_std - torch.log(err + 1e-6)).abs()
        elif args.loss == "mae":
            per = (torch.exp(log_std) - err).abs()
        else:  # sps surrogate: h = sigma, soft coverage indicator
            h = torch.exp(log_std)
            soft = torch.sigmoid((h - err) / args.smooth)
            per = -(torch.exp(-2.0 * h / SIGMA_G) * soft)
            if args.reward_weighted:
                rw = torch.from_numpy(rew[idx]).to(device)
                per = per * rw[:, None, None, None, None]
                denom = (mask * rw[:, None, None, None, None]).sum().clamp_min(1.0)
        loss = (per * mask).sum() / denom
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 250 == 0:
            print(f"[head] step {step} loss {float(loss.detach().cpu()):.5f} {time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.updates:
            evaluate(step)
            torch.save({"head_state_dict": head.state_dict(), "head_config": asdict(cfg), "step": step}, out / f"head_{step}.pth")
    best = max(log, key=lambda row: float(row["sps"]))
    (out / "summary.json").write_text(
        json.dumps({"best": best, "head_config": asdict(cfg), "updates": args.updates}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "DONE").touch()
    print("[head] DONE", flush=True)


if __name__ == "__main__":
    main()
