"""MF Energy Campaign 01: bounded E1/E2/E3 runners and offline oracle tools."""
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn

import realpde_loss_official_v9 as core
from realpde_mf01 import MF01CNO, build_features, factorized_reconstruct, init_mf_from_direct

BASE_N2 = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}
E1_N2 = BASE_N2 | {"tke": 0.10}
GAIN_NAMES = ("u_history_std", "v_history_std", "delta_u", "delta_v", "history_tke_proxy")


def reconstruct_with_gain(mean_raw: Tensor, fluct_raw: Tensor, pressure: Tensor, gain_raw: Tensor):
    mean_field = mean_raw.mean(dim=1, keepdim=True).expand_as(mean_raw)
    fluctuation = fluct_raw - fluct_raw.mean(dim=1, keepdim=True)
    alpha = 1.0 + 0.20 * torch.tanh(gain_raw)
    return torch.cat((mean_field + alpha.permute(0, 2, 3, 1).unsqueeze(1) * fluctuation, pressure), dim=-1), fluctuation, alpha


class GainModel(nn.Module):
    def __init__(self, base: MF01CNO, feature_names: tuple[str, ...], mode: str):
        super().__init__(); self.base = base; self.mode = mode
        indices = [feature_names.index(name) for name in GAIN_NAMES]
        self.register_buffer("gain_indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        if mode == "condgain": self.gain_head = nn.Linear(5, 1)
        elif mode == "spatialgain": self.gain_head = nn.Conv2d(5, 1, kernel_size=1)
        else: raise ValueError(mode)
        nn.init.zeros_(self.gain_head.weight); nn.init.zeros_(self.gain_head.bias)

    def forward(self, features: Tensor):
        raw = self.base.cno(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        mean_raw, fluct_raw, pressure = raw[..., :2], raw[..., 2:4], raw[..., 4:5]
        gain_features = features.index_select(-1, self.gain_indices)
        if self.mode == "condgain":
            gain_raw = self.gain_head(gain_features.mean(dim=(1, 2, 3))).reshape(-1, 1, 1, 1)
        else:
            gain_raw = self.gain_head(gain_features.mean(dim=1).permute(0, 3, 1, 2))
        return reconstruct_with_gain(mean_raw, fluct_raw, pressure, gain_raw)


def forward(model: nn.Module, builder, x: Tensor):
    features = builder(x)
    if isinstance(model, GainModel): return model(features)[0]
    raw = model.cno(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
    uv = factorized_reconstruct(raw[..., :2], raw[..., 2:4])[0]
    return torch.cat((uv, raw[..., 4:5]), -1)


@torch.no_grad()
def evaluate(model, builder, paths, args, device, kit_root, out):
    out.mkdir(parents=True, exist_ok=True); ds, loader = core.loader(paths, args, shuffle=False)
    preds, targets, elapsed = [], [], 0.0; model.eval()
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter(); pred = forward(model, builder, x)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - started; preds.append(pred.cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
    pred, target = np.concatenate(preds), np.concatenate(targets)
    result = core.score_bundle(kit_root, pred, target, elapsed / len(ds), out)
    rows, anatomy = core.trajectory_rows(ds, pred, target, kit_root); result.update({"windows": len(ds), "trajectories": len(rows), "trajectory_anatomy": anatomy})
    with (out / "trajectory_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / "predictions.npz", prediction=pred, target=target)
    return result


def save(path, model, optimizer, step, metadata):
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": step, "metadata": metadata}, path)


def run(args):
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    core.set_seed(args.seed); args.out_dir.mkdir(parents=True); _, train_paths = core.read_manifest(args.manifest, "train"); _, dev_paths = core.read_manifest(args.manifest, "dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); builder, config = build_features(train_paths, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu"); base = MF01CNO(args.kit_root, len(builder.feature_names), device); init_mf_from_direct(base, ckpt, len(builder.feature_names))
    model = base if args.mode == "e1" else GainModel(base, builder.feature_names, args.mode).to(device)
    weights = E1_N2 if args.mode == "e1" else BASE_N2
    metadata = {"experiment_id": args.experiment_id, "mode": args.mode, "seed": args.seed, "updates": args.updates, "milestones": args.milestones, "lr": args.lr, "batch_size": args.batch_size, "workers": args.workers, "loss_weights": weights, "gain_names": list(GAIN_NAMES) if args.mode != "e1" else [], "manifest_sha256": core.sha256(args.manifest), "checkpoint_sha256": core.sha256(args.checkpoint), "scorer_sha256": core.sha256(args.kit_root / "scoring.py"), "feature_config": vars(config), "locked_final_accessed": False, "codabench_accessed": False, "started_at": time.time()}
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    train_ds, loader = core.loader(train_paths, args, shuffle=True); opt = torch.optim.AdamW(model.parameters(), lr=args.lr); evaluate(model, builder, dev_paths, args, device, args.kit_root, args.out_dir / "eval_00000"); iterator = iter(loader); started = time.monotonic(); history=[]
    for step in range(1, args.updates + 1):
        try: x, y, _, _ = next(iterator)
        except StopIteration: iterator = iter(loader); x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); model.train(); pred = forward(model, builder, x); parts = core.loss_parts(pred, y); loss = sum(weights[k] * parts[k] for k in weights); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in args.milestones:
            ev=evaluate(model,builder,dev_paths,args,device,args.kit_root,args.out_dir/f"eval_{step:05d}"); row={"iteration":step,**ev["raw_errors"],"official_v9_subscores":ev["official_v9_subscores"],"elapsed_seconds":time.monotonic()-started}; history.append(row); save(args.out_dir/f"model_update_{step:05d}.pth",model,opt,step,metadata); print(json.dumps(row,sort_keys=True),flush=True)
    save(args.out_dir/"model_last.pth",model,opt,args.updates,metadata); (args.out_dir/"summary.json").write_text(json.dumps({"metadata":metadata|{"train_windows":len(train_ds),"elapsed_seconds":time.monotonic()-started},"history":history},indent=2,default=str),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("e1","condgain","spatialgain"),required=True); p.add_argument("--experiment-id",required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--kit-root",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True); p.add_argument("--updates",type=int,default=1500); p.add_argument("--milestones",type=int,nargs="+",default=[500,1000,1500]); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--workers",type=int,default=2); p.add_argument("--max-windows",type=int,default=None); p.add_argument("--lr",type=float,default=1e-5); p.add_argument("--seed",type=int,default=20260901); run(p.parse_args())
if __name__ == "__main__": main()
