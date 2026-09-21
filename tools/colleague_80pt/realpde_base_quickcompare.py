#!/usr/bin/env python3
"""Same-protocol quick comparison of base backbones on the RealPDE H5 data.

Every candidate is trained with the *same* real-data protocol (train-only
trajectories, same windows, same multi-term loss, same step budget) so the only
difference is the base architecture / starting weights:

    cno   : official CNO3d,  init = official sim/sim-real CNO checkpoint
    fno   : official FNO3d,  init = official sim/sim-real FNO checkpoint
    unet  : official Unet3d, init = scratch (no official weights on server)

All models consume the channel-last field [B, T, H, W, C] (u, v, p) and output
[B, T, H, W, 3] with p zeroed, exactly like the CNO+residual pipeline.

Reported numbers are the official Track-1 errors/subscores (rel_l2, TKE, MVPE)
computed with realpde_sps_scoring on the same 640-window validation split.

Usage (server, inside a container with /repo, /data, /third_party mounted):
    python /repo/tools/realpde_base_quickcompare.py \
        --real-root /data/p0ab_real_h5_20260830 --realpdebench-root /third_party \
        --model cno --init /runs/.../official_cno.pth \
        --out-dir /runs/qc_cno --updates 600 --lr 1e-4

    # pure shape self-test, no data needed:
    python /repo/tools/realpde_base_quickcompare.py --model fno --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realpde_h5_feature_adapter_train import (  # noqa: E402
    BAD_TRAIN_FILES,
    H5WindowDataset,
    list_h5,
    load_cno_checkpoint,
    load_cno_class,
    split_paths,
)
import realpde_sps_scoring as SPS  # noqa: E402


def flexible_load(module: nn.Module, state: dict[str, Tensor], strict: bool = False) -> None:
    fixed = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        fixed[new_key] = value
    result = module.load_state_dict(fixed, strict=strict)
    print(
        f"[load] missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}",
        flush=True,
    )
    if result.missing_keys:
        print("  missing sample:", result.missing_keys[:8], flush=True)
    if result.unexpected_keys:
        print("  unexpected sample:", result.unexpected_keys[:8], flush=True)


def build_cno(realpdebench_root: Path, device: torch.device) -> nn.Module:
    CNO3d = load_cno_class(realpdebench_root)
    model = CNO3d(
        in_dim=3,
        out_dim=3,
        out_dim_mult=1,
        in_size=64,
        N_layers=3,
        activation="LeakyReLU",
    ).to(device)
    return model


def build_fno(device: torch.device) -> nn.Module:
    sys.path.insert(0, str(Path("/third_party")))
    from realpdebench.model.fno import FNO3d

    model = FNO3d(
        modes1=4,
        modes2=12,
        modes3=16,
        n_layers=4,
        width=64,
        shape_in=(20, 32, 64, 3),
        shape_out=(20, 32, 64, 3),
    ).to(device)
    return model


def build_unet(device: torch.device) -> nn.Module:
    sys.path.insert(0, str(Path("/third_party")))
    from realpdebench.model.unet import Unet3d

    model = Unet3d(
        dim=64,
        out_channels=3,
        dim_mults=(1, 2, 4),
        channels=3,
        in_time=20,
        out_time=20,
    ).to(device)
    return model


def build_transolver(init: Path | None, device: torch.device) -> nn.Module:
    """Official Transolver via the starting-kit vendored loader (works at 32x64)."""
    kit = Path("/runs/_kit")
    if str(kit) not in sys.path:
        sys.path.insert(0, str(kit))
    import load_baseline as LB  # noqa: E402

    if init is not None and Path(init).exists():
        model, meta = LB.load_baseline("transolver", str(init), device=device.type)
        print(f"[load] transolver missing={len(meta.get('missing_keys', []))} unexpected={len(meta.get('unexpected_keys', []))}", flush=True)
    else:
        model = LB.build_model("transolver", device=device.type)
    model.train()
    return model


def make_model(model_name: str, init: Path | None, realpdebench_root: Path, device: torch.device) -> nn.Module:
    if model_name == "transolver":
        return build_transolver(init, device)
    if model_name == "cno":
        model = build_cno(realpdebench_root, device)
    elif model_name == "fno":
        model = build_fno(device)
    elif model_name == "unet":
        model = build_unet(device)
    else:
        raise ValueError(model_name)
    if init is not None and init.exists():
        ckpt = torch.load(init, map_location=device)
        state = ckpt if isinstance(ckpt, dict) and "state_dict" not in ckpt else ckpt.get("state_dict", ckpt)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        flexible_load(model, state)
    else:
        print(f"[model] {model_name}: no init -> random init", flush=True)
    model.train()
    return model


def forward(model: nn.Module, model_name: str, x: Tensor) -> Tensor:
    """x: [B, T, H, W, C] -> pred [B, T, H, W, 3] (p zeroed)."""
    pred = model(x)
    if pred.shape[-1] < 3:
        pred = torch.cat([pred, torch.zeros_like(pred[..., :1])], dim=-1)
    if pred.ndim == 6:
        pred = pred[:, :, 0]  # safety for any extra dim
    pred = pred[..., :3]
    pred = pred.clone()
    pred[..., 2] = 0.0
    return pred


def combined_loss(pred: Tensor, target: Tensor) -> Tensor:
    p = pred[..., :2]
    t = target[..., :2]
    point = ((p - t) ** 2).mean()
    mse = ((pred - target) ** 2).mean()
    ke_p = 0.5 * ((p - p.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    ke_t = 0.5 * ((t - t.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    tke = ((ke_p - ke_t) ** 2).mean()
    p_zero = (pred[..., 2] ** 2).mean()
    return 1.0 * point + 0.05 * mse + 0.12 * tke + 0.01 * p_zero


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targs: list[np.ndarray] = []
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pr = forward(model, model_name, x).detach().cpu().numpy().astype(np.float32)
        preds.append(pr)
        targs.append(y.cpu().numpy().astype(np.float32))
    model.train()
    pred = np.concatenate(preds, 0)
    target = np.concatenate(targs, 0)
    c = SPS.measured_channels(target)
    rel = float(np.mean(SPS.rel_l2_per_sample(pred, target, c)))
    tke = float(np.mean(SPS.tke_rel_l2_per_sample(pred, target, c)))
    mvpe = float(np.mean(SPS.mvpe_rel_l2_per_sample(pred, target)))
    return {
        "n_windows": int(pred.shape[0]),
        "rel_l2": rel,
        "rel_l2_score": SPS.score_error(rel),
        "tke": tke,
        "tke_score": SPS.score_error(tke),
        "mvpe": mvpe,
        "mvpe_score": SPS.score_error(mvpe),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", type=Path, default=None)
    ap.add_argument("--realpdebench-root", type=Path, default=None)
    ap.add_argument("--model", choices=["cno", "fno", "unet", "transolver"], default="cno")
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--test-batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--in-steps", type=int, default=20)
    ap.add_argument("--out-steps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--sub-sample", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--max-eval-batches", type=int, default=None)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.selftest:
        model = make_model(args.model, args.init, args.realpdebench_root or Path("/third_party"), device)
        x = torch.randn(2, 20, 32, 64, 3, device=device)
        with torch.no_grad():
            y = forward(model, args.model, x)
        print(
            f"[selftest] {args.model}: in {tuple(x.shape)} -> out {tuple(y.shape)} "
            f"nan={bool(torch.isnan(y).any().item())}",
            flush=True,
        )
        return

    if args.real_root is None or args.out_dir is None or args.realpdebench_root is None:
        raise SystemExit("need --real-root --realpdebench-root --out-dir (unless --selftest)")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    paths = list_h5(args.real_root, BAD_TRAIN_FILES)
    train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    common = dict(
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        stride=args.stride,
        sub_sample=args.sub_sample,
        include_pressure=False,
    )
    train_ds = H5WindowDataset(train_paths, **common)
    val_ds = H5WindowDataset(val_paths, **common)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.test_batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")

    model = make_model(args.model, args.init, args.realpdebench_root, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run] model={args.model} params={n_params} train_windows={len(train_ds)}", flush=True)

    run_config = vars(args)
    run_config["params"] = int(n_params)
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    eval_log: list[dict[str, object]] = []
    it0 = evaluate(model, args.model, val_loader, device, args.max_eval_batches)
    it0["step"] = 0
    eval_log.append(it0)
    print(f"[eval] step=0 " + " ".join(f"{k}={v:.5g}" for k, v in it0.items() if isinstance(v, float)), flush=True)
    (args.out_dir / "eval_log.jsonl").open("a").write(json.dumps(it0, default=float) + "\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = dict(it0)
    best["best_step"] = 0
    step = 0
    it = iter(train_loader)
    while step < args.updates:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = forward(model, args.model, x)
        loss = combined_loss(pred, y)
        loss.backward()
        optimizer.step()
        step += 1
        if step % 50 == 0:
            print(f"[train] step {step}/{args.updates} loss={float(loss.detach().cpu()):.5f}", flush=True)
        if step % args.eval_interval == 0 or step == args.updates:
            ev = evaluate(model, args.model, val_loader, device, args.max_eval_batches)
            ev["step"] = step
            eval_log.append(ev)
            print(f"[eval] step={step} rel={ev['rel_l2']:.5g} tke={ev['tke']:.5g} mvpe={ev['mvpe']:.5g}", flush=True)
            (args.out_dir / "eval_log.jsonl").open("a").write(json.dumps(ev, default=float) + "\n")
            score = -float(ev["rel_l2"] + 0.5 * ev["tke"] + ev["mvpe"])
            if score < -float(best["rel_l2"] + 0.5 * best["tke"] + best["mvpe"]):
                best = dict(ev)
                best["best_step"] = step
                torch.save({"state_dict": model.state_dict(), "run_config": run_config},
                           args.out_dir / "model_best.pth")

    torch.save({"state_dict": model.state_dict(), "run_config": run_config}, args.out_dir / "model_final.pth")
    summary = {"model": args.model, "init": str(args.init), "eval_log": eval_log, "best": best}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"model": args.model, "best": best}, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()

