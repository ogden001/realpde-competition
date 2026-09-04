#!/usr/bin/env python3
"""REP-01: matched frozen-CNO representation probe.

The CNO backbone is never optimized.  A tiny per-pixel linear probe maps the
frozen 20x32x64x3 CNO representation to Future20 u/v.  A1 loads official
sim_pretrain weights; A0 uses a seeded random CNO with the identical shape.
Only the manifest train/dev trajectories are touched.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from realpde_p0_data import H5WindowDataset  # noqa: E402


class TinyLinearProbe(nn.Module):
    def __init__(self, in_dim: int = 60, out_dim: int = 40):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def coarse_target(target: np.ndarray) -> np.ndarray:
    """Temporal-mean target, retaining the spatial/velocity dimensions."""
    return np.asarray(target).mean(axis=1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_cno(kit_root: Path, checkpoint: Path | None, device: torch.device, seed: int):
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    seed_all(seed)
    model = CNO3d(in_dim=3, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


@torch.no_grad()
def representation(model, x: torch.Tensor) -> torch.Tensor:
    return model(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


def probe_features(rep: torch.Tensor) -> torch.Tensor:
    b, t, h, w, c = rep.shape
    return rep.permute(0, 2, 3, 1, 4).reshape(b, h * w, t * c)


def probe_output(raw: torch.Tensor, head: TinyLinearProbe) -> torch.Tensor:
    b, hw, _ = raw.shape
    return head(raw).reshape(b, 32, 64, 20, 2).permute(0, 3, 1, 2, 4)


def metrics(scoring, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p = np.zeros((*pred.shape[:-1], 3), dtype=np.float32); p[..., :2] = pred
    y = np.zeros_like(p); y[..., :2] = target
    channels = scoring.measured_channels(y)
    return {"rel_l2": float(np.mean(scoring.rel_l2_per_sample(p, y, channels))),
            "tke": float(np.mean(scoring.tke_rel_l2_per_sample(p, y, channels))),
            "mvpe": float(scoring.mvpe_rel_l2(p, y))}


def make_loader(paths, batch_size, shuffle, seed):
    ds = H5WindowDataset(paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, num_workers=0)


def run_arm(name, args, train_paths, dev_paths, scoring, device):
    checkpoint = args.checkpoint if name == "A1" else None
    model = load_cno(args.kit_root, checkpoint, device, args.seed)
    head = TinyLinearProbe().to(device)
    seed_all(args.seed + 1)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    train_ds, train_loader = make_loader(train_paths, args.batch_size, True, args.seed)
    started = time.monotonic(); head.train(); last_loss = None
    it = iter(train_loader)
    for step in range(1, args.updates + 1):
        try: x, y, _, _ = next(it)
        except StopIteration: it = iter(train_loader); x, y, _, _ = next(it)
        x, y = x.to(device), y[..., :2].to(device)
        with torch.no_grad(): z = probe_features(representation(model, x))
        pred = probe_output(z, head)
        loss = (pred - y).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); last_loss = float(loss.detach().cpu())
    dev_ds, dev_loader = make_loader(dev_paths, args.batch_size, False, args.seed)
    head.eval(); pred_all=[]; target_all=[]; elapsed=0.0
    with torch.no_grad():
        for x, y, _, _ in dev_loader:
            xdev=x.to(device); t0=time.perf_counter(); z=probe_features(representation(model,xdev)); pred=probe_output(z,head)
            if device.type == "cuda": torch.cuda.synchronize()
            elapsed += time.perf_counter()-t0
            pred_all.append(pred.cpu().numpy().astype(np.float32)); target_all.append(y[..., :2].numpy().astype(np.float32))
    pred=np.concatenate(pred_all); target=np.concatenate(target_all)
    return {"arm":name,"updates":args.updates,"train_windows":len(train_ds),"dev_windows":len(dev_ds),"last_train_mse":last_loss,"train_seconds":time.monotonic()-started,"mean_inference_seconds_per_window":elapsed/max(len(dev_ds),1),"metrics":metrics(scoring,pred,target),"representation_dim":60,"head_parameters":sum(p.numel() for p in head.parameters())}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--kit-root",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True); p.add_argument("--updates",type=int,default=500); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--seed",type=int,default=20260901); p.add_argument("--device",default="cuda")
    args=p.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True); seed_all(args.seed)
    manifest=json.loads(args.manifest.read_text()); train_paths=[args.data_root/r["file"] for r in manifest["train"]]; dev_paths=[args.data_root/r["file"] for r in manifest["dev"]]
    if any(not x.is_file() for x in train_paths+dev_paths): raise FileNotFoundError("manifest trajectory missing")
    sys.path.insert(0,str(args.kit_root)); import scoring
    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    results=[run_arm(n,args,train_paths,dev_paths,scoring,device) for n in ("A1","A0")]
    meta={"experiment_id":"T1-ID-SIM2REAL-REP01-S20260904","manifest_sha256":sha256(args.manifest),"checkpoint_sha256":sha256(args.checkpoint),"checkpoint":str(args.checkpoint),"kit_scorer_sha256":sha256(args.kit_root/"scoring.py"),"train_trajectories":len(train_paths),"dev_trajectories":len(dev_paths),"locked_final_accessed":False,"codabench":False,"protocol":"frozen CNO; 20->20; stride20; 32x64; u/v only; tiny per-pixel 60->40 linear probe; MSE; no backbone updates","device":str(device),"arms":results}
    (args.out_dir/"summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    with (args.out_dir/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["arm","rel_l2","tke","mvpe","train_seconds","last_train_mse"]); w.writeheader(); [w.writerow({"arm":r["arm"],**r["metrics"],"train_seconds":r["train_seconds"],"last_train_mse":r["last_train_mse"]}) for r in results]
    (args.out_dir/"report.md").write_text("# REP-01 Frozen CNO Representation Probe\n\n"+json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2))

if __name__ == "__main__": main()
