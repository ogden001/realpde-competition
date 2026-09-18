#!/usr/bin/env python3
"""Fast 50/16 joint fine-tune for SOTA-V3.

Start from validated SOTA-V2 Dev backbone @32500 and mature 30k residual
corrector, then jointly fine-tune both for 5k updates on frozen Train50 with
AoA augmentation. Dev16 is used only at fixed 1k milestones. There is no
full-data, SPS, package, locked-final/private, or Codabench path here.
"""
from __future__ import annotations

import argparse, csv, json, math, time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

import realpde_residual_corrector_projection as projection
import realpde_sota_v2_integrated as base
import realpde_sota_v3_backbone as v3_backbone
import realpde_teammate_residual_transfer as transfer
from realpde_adaptive_probe import ResidualCorrector3D, feature_config_from_checkpoint
from realpde_p0_features import P0FeatureBuilder

BACKBONE_SHA256 = "6926722895611c38dee79cf16ce43b87c7de3d4a1305e9db431dadd4c1a7bf47"
CORRECTOR_SHA256 = "1f72b329d9a160b206a6e9f4fce5e780022b3bac7a9ce4bdf5b8c9efa30b4506"
BACKBONE_REFERENCE_UPDATE = 32_500
CORRECTOR_REFERENCE_UPDATE = 30_000
UPDATES = 5_000
MILESTONES = (1_000, 2_000, 3_000, 4_000, 5_000)
RECOVERY_EVERY = 500
EFFECTIVE_BATCH = 8
BACKBONE_LR = 1e-6
CORRECTOR_LR = 1e-5
BACKBONE_WEIGHT_DECAY = 1e-2
CORRECTOR_WEIGHT_DECAY = 1e-5
SEED = base.SEED
HISTORICAL_FINAL = {
    "rel_l2": 0.0889103040099144,
    "tke": 0.4692927300930023,
    "mvpe": 0.07112862169742584,
}
HISTORICAL_BASE = {
    "rel_l2": 0.09993461519479752,
    "tke": 0.4692927300930023,
    "mvpe": 0.07577798515558243,
}
REPLAY_ATOL = 5e-6
MIN_AVG_IMPROVEMENT = 0.01
MAX_SINGLE_METRIC_DEGRADATION = 0.02
MIN_IMPROVED_METRICS = 2


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict]) -> None:
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in values for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(values)


def selection_objective(metrics: dict[str, float]) -> float:
    vals = []
    for key in ("rel_l2", "tke", "mvpe"):
        val = float(metrics[key])
        if not math.isfinite(val) or val < 0:
            return float("inf")
        vals.append(val / HISTORICAL_FINAL[key])
    return float(sum(vals))


def select_candidate(history: list[dict]) -> dict:
    candidates = [r for r in history if int(r["update"]) in MILESTONES]
    if not candidates:
        raise ValueError("no eligible joint milestone")
    best = min(candidates, key=lambda r: (selection_objective(r), int(r["update"])))
    return {**best, "selection_objective": selection_objective(best)}


def evaluate_gate(metrics: dict[str, float]) -> dict[str, object]:
    improvements = {
        k: 1.0 - float(metrics[k]) / HISTORICAL_FINAL[k]
        for k in ("rel_l2", "tke", "mvpe")
    }
    improved_count = sum(v > 0 for v in improvements.values())
    avg_improvement = 1.0 - selection_objective(metrics) / 3.0
    checks = {
        "avg_normalized_error_improvement_ge_1pct": avg_improvement >= MIN_AVG_IMPROVEMENT,
        "no_single_metric_degradation_gt_2pct": all(v >= -MAX_SINGLE_METRIC_DEGRADATION for v in improvements.values()),
        "at_least_two_metrics_improve": improved_count >= MIN_IMPROVED_METRICS,
    }
    return {
        "status": "FAST_JOINT_GO_FULL" if all(checks.values()) else "FAST_JOINT_NO_GO",
        "checks": checks,
        "improvements": improvements,
        "improved_metric_count": improved_count,
        "avg_normalized_error_improvement": avg_improvement,
        "historical_final": HISTORICAL_FINAL,
    }


def _load_backbone(path: Path, kit_root: Path, device: torch.device):
    sha = base.sha256(path)
    if sha != BACKBONE_SHA256:
        raise ValueError(f"backbone SHA mismatch: {sha}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("feature_set") != "P0-A" or int(payload.get("iteration", -1)) != BACKBONE_REFERENCE_UPDATE:
        raise ValueError("backbone must be frozen P0-A @32500")
    cfg = feature_config_from_checkpoint(payload)
    builder = P0FeatureBuilder(cfg).to(device)
    model = base.MF01CNO(kit_root, len(builder.feature_names), device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return cfg, builder, model


def _load_corrector(path: Path, device: torch.device):
    sha = base.sha256(path)
    if sha != CORRECTOR_SHA256:
        raise ValueError(f"corrector SHA mismatch: {sha}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("backbone_sha256") != BACKBONE_SHA256 or int(payload.get("updates", -1)) != CORRECTOR_REFERENCE_UPDATE:
        raise ValueError("corrector must be mature @30000 on frozen @32500 backbone")
    model = ResidualCorrector3D(in_channels=42, hidden=64, blocks=2, max_delta=0.04).to(device)
    model.load_state_dict(payload["corrector_state_dict"], strict=True)
    return model


def forward_final(backbone, builder, corrector, x):
    base_pred = base.forward_mf(backbone, builder, x)
    delta = transfer._delta_from_corrector(corrector, x, base_pred)
    corrected = transfer.apply_scaled_correction(base_pred, delta, 1.0)
    final = projection.spatial_tke_map_projection(base_pred, corrected)
    return base_pred, final


def stage_b_loss(pred, target):
    parts = base.core.loss_parts(pred, target)
    parts["vorticity"] = (
        base.vorticity(pred, dx=1.0, dy=1.0) - base.vorticity(target, dx=1.0, dy=1.0)
    ).square().mean()
    total = sum(base.N2[k] * parts[k] for k in base.N2)
    total = total + base.LAMBDA_VORT * parts["vorticity"] + base.EXTRA_REL * parts["rel"]
    return total, parts


def _save_resume(path, backbone, corrector, optimizer, scheduler, update, epoch, batches_in_epoch, history, cfg):
    payload = {
        "backbone_state_dict": backbone.state_dict(), "corrector_state_dict": corrector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "joint_update": update, "epoch": epoch, "batches_in_epoch": batches_in_epoch,
        "history": history, "feature_config": vars(cfg), "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(), "source_backbone_sha256": BACKBONE_SHA256,
        "source_corrector_sha256": CORRECTOR_SHA256,
    }
    if torch.cuda.is_available(): payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def _restore_resume(path, backbone, corrector, optimizer, scheduler):
    p = torch.load(path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(p["backbone_state_dict"], strict=True)
    corrector.load_state_dict(p["corrector_state_dict"], strict=True)
    optimizer.load_state_dict(p["optimizer_state_dict"]); scheduler.load_state_dict(p["scheduler_state_dict"])
    torch.set_rng_state(p["torch_rng_state"]); np.random.set_state(p["numpy_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in p: torch.cuda.set_rng_state_all(p["cuda_rng_state_all"])
    return p


def _resume_iterator(loader, sampler, epoch, batches_in_epoch):
    sampler.set_epoch(epoch); it = iter(loader)
    for _ in range(batches_in_epoch):
        try: next(it)
        except StopIteration as exc: raise RuntimeError("saved batches_in_epoch exceeds loader length") from exc
    return it


def _save_pair(out_dir, update, backbone, corrector, cfg):
    ck = out_dir / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    bp = ck / f"backbone_joint_{update:05d}.pth"
    torch.save({
        "model_state_dict": {k:v.detach().cpu() for k,v in backbone.state_dict().items()},
        "iteration": BACKBONE_REFERENCE_UPDATE, "joint_finetune_update": update,
        "feature_set": "P0-A", "feature_config": vars(cfg), "loss_weights": base.N2,
        "lambda_vort": base.LAMBDA_VORT, "extra_rel_stage_b": base.EXTRA_REL,
        "metadata": {"recipe":"fast_joint_50_16", "source_backbone_sha256":BACKBONE_SHA256,
                     "source_corrector_sha256":CORRECTOR_SHA256},
    }, bp)
    cp = ck / f"corrector_joint_{update:05d}.pth"
    torch.save({
        "corrector_state_dict": {k:v.detach().cpu() for k,v in corrector.state_dict().items()},
        "updates": CORRECTOR_REFERENCE_UPDATE, "joint_finetune_update": update,
        "backbone_sha256": base.sha256(bp), "backbone_iteration": BACKBONE_REFERENCE_UPDATE,
        "architecture": {"in_channels":42,"hidden":64,"blocks":2,"max_delta":0.04},
        "source_corrector_sha256": CORRECTOR_SHA256, "source_backbone_sha256": BACKBONE_SHA256,
    }, cp)
    return bp, cp


@torch.no_grad()
def evaluate(backbone, builder, corrector, dev_paths, args, device, out_dir, update):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds, loader = base.dev_loader(dev_paths, Namespace(eval_batch_size=args.eval_batch_size, workers=args.workers))
    bchunks, fchunks, targets = [], [], []
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        bp, fp = forward_final(backbone, builder, corrector, x)
        bchunks.append(bp.cpu().numpy().astype(np.float32)); fchunks.append(fp.cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
    bpred, fpred, target = np.concatenate(bchunks), np.concatenate(fchunks), np.concatenate(targets)
    braw = transfer.raw_physical_errors(args.kit_root, bpred, target)
    fraw = transfer.raw_physical_errors(args.kit_root, fpred, target)
    traj, anatomy = base.core.trajectory_rows(ds, fpred, target, args.kit_root)
    write_rows(out_dir / "trajectory_metrics.csv", traj); dump(out_dir / "trajectory_anatomy.json", anatomy)
    dump(out_dir / "horizon_error_summary.json", base.horizon_error_summary(fpred, target))
    result = {"update":update, "rel_l2":fraw["rel_l2"], "tke":fraw["tke"], "mvpe":fraw["mvpe"],
              "base_rel_l2":braw["rel_l2"], "base_tke":braw["tke"], "base_mvpe":braw["mvpe"],
              "selection_objective":selection_objective(fraw), "dev_windows":len(ds),
              "dev_trajectories":len(set(r.path.name for r in ds.refs))}
    dump(out_dir / "metrics.json", result); return result


def _assert_replay(row):
    for prefix, expected in (("", HISTORICAL_FINAL), ("base_", HISTORICAL_BASE)):
        for metric in ("rel_l2", "tke", "mvpe"):
            key = prefix + metric
            if abs(float(row[key]) - expected[metric]) > REPLAY_ATOL:
                raise RuntimeError(f"historical replay mismatch {key}: {row[key]} vs {expected[metric]}")


def run(args):
    if args.updates != UPDATES or tuple(args.milestones) != MILESTONES: raise ValueError("frozen schedule is 5k with 1k milestones")
    if args.micro_batch * args.accumulation_steps != EFFECTIVE_BATCH: raise ValueError("effective batch must equal 8")
    if args.aoa_max_deg != v3_backbone.AOA_MAX_DEG or args.aoa_probability != v3_backbone.AOA_PROBABILITY: raise ValueError("AoA semantics changed")
    if base.sha256(args.manifest) != base.MANIFEST_SHA: raise ValueError("frozen 50/16 manifest SHA mismatch")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume: raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_paths = base.split_paths(args.manifest, "train", args.data_root); dev_paths = base.split_paths(args.manifest, "dev", args.data_root)
    if (len(train_paths), len(dev_paths)) != (50,16): raise ValueError("expected Train50/Dev16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda": raise RuntimeError("CUDA required")
    base.core.set_seed(SEED)
    cfg, builder, backbone = _load_backbone(args.backbone_checkpoint, args.kit_root, device)
    corrector = _load_corrector(args.corrector_checkpoint, device)
    optimizer = torch.optim.AdamW([
        {"params":backbone.parameters(),"lr":BACKBONE_LR,"weight_decay":BACKBONE_WEIGHT_DECAY},
        {"params":corrector.parameters(),"lr":CORRECTOR_LR,"weight_decay":CORRECTOR_WEIGHT_DECAY},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    dump(args.out_dir / "run_metadata.json", {
        "experiment":"SOTA-V3_FAST_JOINT_50_16", "scope":"Train50/Dev16 only", "updates":UPDATES,
        "milestones":list(MILESTONES), "effective_batch":EFFECTIVE_BATCH, "backbone_lr":BACKBONE_LR,
        "corrector_lr":CORRECTOR_LR, "loss":"Stage-B physical loss on projected final", "projection":"spatial_tke_map",
        "aoa":{"max_deg":v3_backbone.AOA_MAX_DEG,"probability":v3_backbone.AOA_PROBABILITY,"train_only":True},
        "historical_comparator":HISTORICAL_FINAL, "locked_final_accessed":False,"full_data_accessed":False,"codabench_accessed":False,
    })

    history=[]; state=args.out_dir/"checkpoints"/"runner_latest.pth"; state.parent.mkdir(parents=True,exist_ok=True)
    start, epoch, batches = 1,0,0
    if args.resume and state.is_file():
        p=_restore_resume(state,backbone,corrector,optimizer,scheduler); start=int(p["joint_update"])+1; epoch=int(p.get("epoch",0)); batches=int(p.get("batches_in_epoch",0)); history=list(p.get("history",[]))
    if not history:
        row=evaluate(backbone,builder,corrector,dev_paths,args,device,args.out_dir/"eval_00000",0); _assert_replay(row); history.append(row); write_rows(args.out_dir/"aggregate_metrics.csv",history)

    ds,sampler,loader=base.dense_loader(train_paths,Namespace(seed=SEED,micro_batch=args.micro_batch,workers=args.workers))
    if len(ds)!=base.EXPECTED["dense_train"]: raise ValueError("Dense-All window audit mismatch")
    it=_resume_iterator(loader,sampler,epoch,batches); started=time.monotonic()
    for update in range(start,UPDATES+1):
        optimizer.zero_grad(set_to_none=True); total_loss=0.0; totals={k:0.0 for k in (*base.N2,"vorticity")}; asum=0.0; acount=0
        for _ in range(args.accumulation_steps):
            try: x,y,_,_=next(it); batches+=1
            except StopIteration:
                epoch+=1; batches=0; sampler.set_epoch(epoch); it=iter(loader); x,y,_,_=next(it); batches=1
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            x,y,angles=v3_backbone.augment_pair(x,y,max_deg=args.aoa_max_deg,probability=args.aoa_probability); asum+=float(angles.abs().sum().cpu()); acount+=angles.numel()
            backbone.train(); corrector.train(); _,final=forward_final(backbone,builder,corrector,x); loss,parts=stage_b_loss(final,y)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite joint loss @ {update}")
            (loss/args.accumulation_steps).backward(); total_loss+=float(loss.detach().cpu())/args.accumulation_steps
            for k in totals: totals[k]+=float(parts[k].detach().cpu())/args.accumulation_steps
        torch.nn.utils.clip_grad_norm_(backbone.parameters(),1.0); torch.nn.utils.clip_grad_norm_(corrector.parameters(),1.0); optimizer.step(); scheduler.step()
        if update%args.log_every==0: print(json.dumps({"update":update,"loss":total_loss,"backbone_lr":optimizer.param_groups[0]["lr"],"corrector_lr":optimizer.param_groups[1]["lr"],"mean_abs_aug_angle_deg":asum/max(acount,1),**totals}),flush=True)
        if update in MILESTONES:
            row=evaluate(backbone,builder,corrector,dev_paths,args,device,args.out_dir/f"eval_{update:05d}",update); row["elapsed_seconds"]=time.monotonic()-started; history.append(row); write_rows(args.out_dir/"aggregate_metrics.csv",history); _save_pair(args.out_dir,update,backbone,corrector,cfg)
        if update%args.recovery_every==0 or update==UPDATES: _save_resume(state,backbone,corrector,optimizer,scheduler,update,epoch,batches,history,cfg)

    selected=select_candidate(history); u=int(selected["update"]); bp=args.out_dir/"checkpoints"/f"backbone_joint_{u:05d}.pth"; cp=args.out_dir/"checkpoints"/f"corrector_joint_{u:05d}.pth"
    metrics={k:float(selected[k]) for k in ("rel_l2","tke","mvpe")}; gate=evaluate_gate(metrics)
    result={"status":"REVIEW_REQUIRED","gate":gate,"selected_update":u,"selected":selected,"selected_backbone":str(bp),"selected_backbone_sha256":base.sha256(bp),"selected_corrector":str(cp),"selected_corrector_sha256":base.sha256(cp),"historical_replay":history[0],"history":history,"runtime_seconds_last_process":time.monotonic()-started,"next_action":"If GO, run Dev SPS only; do not start full training without user review.","locked_final_accessed":False,"full_data_accessed":False,"codabench_accessed":False}
    dump(args.out_dir/"summary.json",result); dump(args.out_dir/"status.json",{"state":"DONE","status":"REVIEW_REQUIRED","gate":gate["status"]}); print(json.dumps(result,indent=2,sort_keys=True)); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--kit-root",type=Path,required=True); p.add_argument("--backbone-checkpoint",type=Path,required=True); p.add_argument("--corrector-checkpoint",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True); p.add_argument("--updates",type=int,default=UPDATES); p.add_argument("--milestones",type=int,nargs="+",default=list(MILESTONES)); p.add_argument("--micro-batch",type=int,default=2); p.add_argument("--accumulation-steps",type=int,default=4); p.add_argument("--eval-batch-size",type=int,default=8); p.add_argument("--workers",type=int,default=2); p.add_argument("--aoa-max-deg",type=float,default=v3_backbone.AOA_MAX_DEG); p.add_argument("--aoa-probability",type=float,default=v3_backbone.AOA_PROBABILITY); p.add_argument("--recovery-every",type=int,default=RECOVERY_EVERY); p.add_argument("--log-every",type=int,default=100); p.add_argument("--resume",action="store_true"); p.add_argument("--require-cuda",action="store_true"); run(p.parse_args())

if __name__ == "__main__": main()
