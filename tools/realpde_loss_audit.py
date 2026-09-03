#!/usr/bin/env python3
"""Validation-only TKE anatomy and loss-gradient audit for Track 1.

This tool intentionally reuses the repository's current local metric proxy.
It does not call it an official scorer: use results only after running against
the released validation split and comparing the baseline to Codabench.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from realpdebench.data.fluid_dataset import Foil
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import set_seed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from realpde_calibrate_bounds import measured_channels, mvpe_rel_l2_per_sample, rel_l2_per_sample, tke_rel_l2_per_sample
from realpde_tke_finetune import (
    BAD_TRAIN_FILES, FilteredDataset, fluctuation_rel_loss_torch,
    kinetic_energy_torch, mean_rel_loss_torch, mvpe_rel_loss_torch,
    rel_l2_loss_torch, spatial_grad_rel_loss, temporal_rel_loss,
)
from realpde_calibrate_bounds import make_args


def _stats(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a): return {k: float("nan") for k in ("mean", "median", "std", "p10", "p25", "p75", "p90")}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std()), "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90))}


def _tke(x):
    f = x[..., :2] - x[..., :2].mean(axis=1, keepdims=True)
    return .5 * np.mean(f ** 2, axis=1).sum(axis=-1)


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float("nan")


def _losses(pred, target):
    uv, tuv = pred[..., :2], target[..., :2]
    return {
        "rel_l2": rel_l2_loss_torch(uv, tuv),
        "mse": torch.mean((uv - tuv) ** 2),
        "tke": rel_l2_loss_torch(kinetic_energy_torch(uv), kinetic_energy_torch(tuv)),
        "mvpe_proxy": mvpe_rel_loss_torch(uv, tuv),
        "temporal": temporal_rel_loss(uv, tuv),
        "spatial_gradient": spatial_grad_rel_loss(uv, tuv),
        "pressure_zero": torch.mean(pred[..., 2] ** 2) if pred.shape[-1] > 2 else pred.new_tensor(0.0),
        "mean": mean_rel_loss_torch(uv, tuv),
        "fluctuation": fluctuation_rel_loss_torch(uv, tuv),
    }


def _grad_vector(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([g.detach().reshape(-1).cpu() for g in gs if g is not None]).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=8, help="Bounded validation batches for this diagnostic.")
    p.add_argument("--gradient-batches", type=int, default=3)
    p.add_argument("--weights-json", default='{"rel_l2":1,"mse":0.05,"tke":0.1,"temporal":0.05,"spatial_gradient":0.03,"pressure_zero":0.01,"mvpe_proxy":0,"mean":0,"fluctuation":0}')
    args = p.parse_args()
    weights = json.loads(args.weights_json)
    out = Path(args.out_dir); fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = make_args(args.config, args.dataset_root, args.checkpoint, str(out))
    base = Foil(dataset_name=cfg.dataset_name, dataset_root=cfg.dataset_root, mode="val", dataset_type="real")
    loader = DataLoader(base, batch_size=args.batch_size, shuffle=False, num_workers=min(getattr(cfg, "num_workers", 0), 4))
    model = load_model(base, device=device, **vars(cfg)); meta = model.load_checkpoint(args.checkpoint, device) or {}
    model.eval(); params = [v for v in model.parameters() if v.requires_grad]
    rows, grad_rows, cosine_rows, maps_p, maps_t = [], [], [], [], []
    for bi, (x, y) in enumerate(loader):
        if bi >= args.max_batches: break
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(bi < args.gradient_batches): pred = model(x)
        pn, tn = pred.detach().cpu().numpy().astype(np.float32), y.detach().cpu().numpy().astype(np.float32)
        pn[..., 2] = 0.0
        kp, kt = _tke(pn), _tke(tn); maps_p.append(kp); maps_t.append(kt)
        p_fluc = pn[..., :2] - pn[..., :2].mean(axis=1, keepdims=True)
        t_fluc = tn[..., :2] - tn[..., :2].mean(axis=1, keepdims=True)
        ku_p, ku_t = .5 * np.mean(p_fluc[..., 0] ** 2, axis=1), .5 * np.mean(t_fluc[..., 0] ** 2, axis=1)
        kv_p, kv_t = .5 * np.mean(p_fluc[..., 1] ** 2, axis=1), .5 * np.mean(t_fluc[..., 1] ** 2, axis=1)
        c = measured_channels(tn); rel = rel_l2_per_sample(pn, tn, c); tke = tke_rel_l2_per_sample(pn, tn, c); mvpe = mvpe_rel_l2_per_sample(pn, tn)
        for j in range(len(pn)):
            gt, pr = kt[j].ravel(), kp[j].ravel(); denom = max(np.linalg.norm(gt), 1e-12)
            alpha = float(np.dot(pr, gt) / max(np.dot(pr, pr), 1e-12))
            corr = float(np.corrcoef(pr, gt)[0, 1]) if np.std(pr) and np.std(gt) else float("nan")
            rows.append({"sample": len(rows), "rel_l2": rel[j], "tke_rel_l2": tke[j], "mvpe": mvpe[j], "tke_norm_ratio": np.linalg.norm(pr)/denom, "tke_sum_ratio": pr.sum()/max(gt.sum(),1e-12), "tke_scale_alpha": alpha, "tke_scale_corrected_rel_l2": np.linalg.norm(alpha*pr-gt)/denom, "tke_correlation": corr, "tke_cosine": _cos(pr,gt), "u_tke_norm_ratio": np.linalg.norm(ku_p[j].ravel()) / max(np.linalg.norm(ku_t[j].ravel()), 1e-12), "v_tke_norm_ratio": np.linalg.norm(kv_p[j].ravel()) / max(np.linalg.norm(kv_t[j].ravel()), 1e-12), "u_tke_rel_l2": np.linalg.norm(ku_p[j].ravel()-ku_t[j].ravel()) / max(np.linalg.norm(ku_t[j].ravel()), 1e-12), "v_tke_rel_l2": np.linalg.norm(kv_p[j].ravel()-kv_t[j].ravel()) / max(np.linalg.norm(kv_t[j].ravel()), 1e-12)})
        if bi < args.gradient_batches:
            ls = _losses(pred, y); vec = {name: _grad_vector(value, params) for name, value in ls.items()}
            for name, v in vec.items(): grad_rows.append({"batch":bi,"loss":name,"raw_grad_norm":float(np.linalg.norm(v)),"weighted_grad_norm":float(abs(weights.get(name,0))*np.linalg.norm(v)),"loss_value":float(ls[name].detach().cpu())})
            for a, va in vec.items():
                for b, vb in vec.items(): cosine_rows.append({"batch":bi,"loss_a":a,"loss_b":b,"cosine":_cos(va,vb)})
    for name, data in (("metrics.csv", rows), ("gradient_norms.csv", grad_rows), ("gradient_cosines.csv", cosine_rows)):
        with (out/name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]) if data else ["empty"]); w.writeheader(); w.writerows(data)
    kp, kt = np.concatenate(maps_p), np.concatenate(maps_t)
    np.savez_compressed(out/"tke_maps.npz", pred_mean=kp.mean(0), target_mean=kt.mean(0))
    fig_, ax = plt.subplots(); ax.hist([r["tke_norm_ratio"] for r in rows], bins=20); ax.set(xlabel="||TKE_pred|| / ||TKE_gt||", ylabel="samples"); fig_.savefig(fig/"tke_energy_ratio_hist.png", dpi=160, bbox_inches="tight"); plt.close(fig_)
    fig_, ax = plt.subplots(); ax.scatter([r["tke_norm_ratio"] for r in rows], [r["tke_correlation"] for r in rows]); ax.set(xlabel="TKE norm ratio", ylabel="TKE Pearson correlation"); fig_.savefig(fig/"tke_ratio_vs_correlation.png", dpi=160, bbox_inches="tight"); plt.close(fig_)
    fig_, ax = plt.subplots(1,4,figsize=(14,3)); ims=[kp.mean(0),kt.mean(0),abs(kp.mean(0)-kt.mean(0)),abs(kp.mean(0)-kt.mean(0))/np.maximum(abs(kt.mean(0)),1e-8)]; titles=["Pred mean TKE","GT mean TKE","Absolute error","Relative error"]
    for a, im, title in zip(ax,ims,titles): a.imshow(im); a.set_title(title); a.axis("off")
    fig_.savefig(fig/"tke_maps.png", dpi=160, bbox_inches="tight"); plt.close(fig_)
    summary={"checkpoint":args.checkpoint,"checkpoint_meta":meta,"validation_samples":len(rows),"max_batches":args.max_batches,"metric_summary":{k:_stats([r[k] for r in rows]) for k in ("rel_l2","tke_rel_l2","mvpe","tke_norm_ratio","tke_sum_ratio","tke_scale_corrected_rel_l2","tke_correlation","tke_cosine","u_tke_norm_ratio","v_tke_norm_ratio","u_tke_rel_l2","v_tke_rel_l2")},"gradient_note":"Norms/cosines are per-batch, full-model autograd values; aggregate CSV before conclusion.","scorer_note":"Rel-L2/TKE/MVPE use the repository local proxy, not a verified official starting-kit scorer."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,indent=2,default=str))

if __name__ == "__main__": main()
