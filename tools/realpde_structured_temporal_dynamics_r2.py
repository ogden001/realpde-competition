#!/usr/bin/env python3
"""Structured Temporal Dynamics R2: latent-only matched arms."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
N2 = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}

def validate_r2_protocol(batch_size: int, accumulation_steps: int) -> None:
    if batch_size != 8 or accumulation_steps != 1:
        raise ValueError("R2 requires physical batch=8 and no gradient accumulation")

class LatentTemporalConv(nn.Module):
    def __init__(self, channels: int, hidden: int = 32):
        super().__init__(); self.input = nn.Conv3d(channels, hidden, (3,1,1), padding=(1,0,0)); self.activation = nn.GELU(); self.output = nn.Conv3d(hidden, channels, (3,1,1), padding=(1,0,0)); nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self, z: Tensor) -> Tensor:
        return z + self.output(self.activation(self.input(z)))

class LatentTemporalAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        if channels % heads: raise ValueError(f"latent channels {channels} not divisible by heads {heads}")
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output = nn.Linear(channels, channels); nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self, z: Tensor) -> Tensor:
        b, c, t, h, w = z.shape
        seq = z.permute(0, 3, 4, 2, 1).reshape(b*h*w, t, c)
        out, _ = self.attention(seq, seq, seq, need_weights=False)
        return z + self.output(out).reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)

class LatentCNO(nn.Module):
    def __init__(self, backbone: nn.Module, adapter: nn.Module | None):
        super().__init__(); self.backbone = backbone; self.adapter = adapter
    def forward(self, x: Tensor) -> Tensor:
        cno = self.backbone; x = cno.lift(x); skip=[]
        for i in range(cno.N_layers):
            y=x
            for j in range(cno.N_res): y=cno.res_nets[i*cno.N_res+j](y)
            skip.append(y); x=cno.encoder[i](x)
        for j in range(cno.N_res_neck): x=cno.res_nets[-j-1](x)
        for i in range(cno.N_layers):
            if i==0: x=cno.ED_expansion[cno.N_layers-i](x)
            else: x=torch.cat((x,cno.ED_expansion[cno.N_layers-i](skip[-i])),1)
            if cno.add_inv: x=cno.decoder_inv[i](x)
            x=cno.decoder[i](x)
        latent=torch.cat((x,cno.ED_expansion[0](skip[0])),1)
        if self.adapter is not None: latent=self.adapter(latent)
        x=cno.project(latent)
        if cno.out_dim_mult>1: x=x.reshape(x.shape[0],-1,x.shape[2],x.shape[3],cno.out_dim//cno.out_dim_mult)
        return x

def build_model(args, train_paths, device):
    import realpde_mf01 as direct
    builder, config = direct.build_features(train_paths, device)
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    backbone=direct.cno_direct(args.kit_root,len(builder.feature_names),device)
    state=payload.get("model_state_dict",payload); direct.adapt_input_weight(backbone,payload,len(builder.feature_names)) if "lift.inter_CNOBlock.convolution.weight" not in state or state["lift.inter_CNOBlock.convolution.weight"].shape[1] != len(builder.feature_names) else backbone.load_state_dict(state,strict=True)
    channels=backbone.project.inter_CNOBlock.convolution.in_channels
    adapter=None
    if args.arm=="A1": adapter=LatentTemporalConv(channels, min(32, channels)).to(device)
    if args.arm=="A2": adapter=LatentTemporalAttention(channels, 4 if channels%4==0 else 2).to(device)
    return backbone, LatentCNO(backbone,adapter) if adapter is not None else backbone, adapter, builder, config, payload, channels

def forward(model, builder, x):
    features=builder(x); pred=model(features.permute(0,4,1,2,3)).permute(0,2,3,4,1); pred=pred.clone(); pred[...,2]=0.; return pred

def loss_v1(pred,target):
    import realpde_loss_official_v9 as core
    parts=core.loss_parts(pred,target); return sum(N2[k]*parts[k] for k in N2),parts

def vorticity_loss(pred: Tensor, target: Tensor) -> Tensor:
    import realpde_structured_temporal_dynamics as legacy
    return (legacy.vorticity(pred, dx=1.0, dy=1.0)-legacy.vorticity(target, dx=1.0, dy=1.0)).square().mean()

@torch.no_grad()
def evaluate(model,builder,paths,args,device,out,arm,update,save):
    import realpde_loss_official_v9 as core
    import realpde_structured_temporal_dynamics as legacy
    out.mkdir(parents=True,exist_ok=True); ds,loader=core.loader(paths,args,shuffle=False); ps=[]; ys=[]; elapsed=0.
    model.eval()
    for x,y,_,_ in loader:
        x=x.to(device,non_blocking=True)
        if device.type=="cuda": torch.cuda.synchronize()
        st=time.perf_counter(); p=forward(model,builder,x)
        if device.type=="cuda": torch.cuda.synchronize()
        elapsed+=time.perf_counter()-st; ps.append(p.cpu().numpy().astype(np.float32)); ys.append(y.numpy().astype(np.float32))
    pred,target=np.concatenate(ps),np.concatenate(ys)
    if not np.isfinite(pred).all() or np.abs(pred[...,2]).max()!=0: raise FloatingPointError("invalid prediction")
    result=core.score_bundle(args.kit_root,pred,target,elapsed/len(ds),out); legacy.diagnostics(ds,pred,target,out,arm,update,args.kit_root)
    if save: np.savez_compressed(out/"predictions.npz",prediction=pred[...,:2],target=target[...,:2],trajectory=np.asarray([r.path.name for r in ds.refs]),window_start=np.asarray([r.start for r in ds.refs]))
    return result|{"windows":len(ds)}

def run(args):
    import realpde_loss_official_v9 as core
    import realpde_mf01 as direct
    import realpde_structured_temporal_dynamics as legacy
    if args.out_dir.exists() and any(args.out_dir.iterdir()): raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    validate_r2_protocol(args.batch_size, args.accumulation_steps)
    _,train=core.read_manifest(args.manifest,"train"); _,dev=core.read_manifest(args.manifest,"dev"); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone,model,adapter,builder,config,payload,channels=build_model(args,train,device)
    ds,loader=core.loader(train,args,shuffle=True); x,y,_,_=next(iter(loader)); x=x.to(device); y=y.to(device)
    base=forward(backbone,builder,x); initial=forward(model,builder,x); parity=float((initial-base).abs().max());
    if parity!=0.0: raise RuntimeError(f"initial parity failed: {parity}")
    if list(base.shape)[1]!=20: raise RuntimeError(f"latent/output time is not Future20: {base.shape}")
    base_opt=torch.optim.AdamW(list(backbone.parameters()),lr=args.lr); state_before=len(base_opt.state); base_opt.load_state_dict(payload["optimizer_state_dict"]); state_after=len(base_opt.state)
    if adapter is not None: base_opt.add_param_group({"params":list(adapter.parameters())})
    if adapter is not None and not all(any(p is q for q in g["params"]) for p in adapter.parameters() for g in base_opt.param_groups[1:]): raise RuntimeError("adapter param group missing")
    parts0=core.loss_parts(initial,y); weight=float(0.10*sum(N2[k]*parts0[k] for k in N2)/vorticity_loss(initial,y).clamp_min(1e-12)) if args.arm=="V1" else 0.0
    probe, _ = loss_v1(initial,y)
    if args.arm=="V1": probe=probe+weight*vorticity_loss(initial,y)
    probe.backward(); grad=max(float(p.grad.abs().max()) for p in adapter.parameters() if p.grad is not None) if adapter is not None else None
    if adapter is not None and not grad>0: raise RuntimeError("adapter gradient is zero")
    base_opt.zero_grad(set_to_none=True)
    pre={"status":"REVIEW_REQUIRED","arm":args.arm,"initial_prediction_max_abs_diff_from_c0":parity,"pressure_max_abs":float(initial[...,2].abs().max()),"latent_project_input_channels":channels,"latent_time":20,"project_preserves_time":True,"backbone_optimizer_state_before_restore":state_before,"backbone_optimizer_state_after_restore":state_after,"adapter_in_new_param_group":adapter is not None,"adapter_gradient_max_abs":grad,"lambda_vort":weight if args.arm=="V1" else None,"finite":bool(torch.isfinite(initial).all())}
    (args.out_dir/"preflight.json").write_text(json.dumps(pre,indent=2),encoding="utf-8")
    if args.preflight_only: return
    baseline=evaluate(model,builder,dev,args,device,args.out_dir/"eval_01500",args.arm,1500,False); history=[{"update":1500,**baseline["raw_errors"]}]; curve=[]; started=time.monotonic(); iterator=iter(loader); peak_a=peak_r=0
    for rel in range(1,args.updates+1):
        try: x,y,_,_=next(iterator)
        except StopIteration: iterator=iter(loader); x,y,_,_=next(iterator)
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); base_opt.zero_grad(set_to_none=True); model.train(); pred=forward(model,builder,x); loss,parts=loss_v1(pred,y)
        if args.arm=="V1": parts["vorticity"]=vorticity_loss(pred,y); loss=loss+weight*parts["vorticity"]
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(backbone.parameters())+([] if adapter is None else list(adapter.parameters())),1.0); base_opt.step(); curve.append({"update":1500+rel,"total":float(loss.detach().cpu()),**{k:float(v.detach().cpu()) for k,v in parts.items()}})
        if device.type=="cuda": peak_a=max(peak_a,torch.cuda.max_memory_allocated()); peak_r=max(peak_r,torch.cuda.max_memory_reserved())
        if rel in args.milestones:
            u=1500+rel; ev=evaluate(model,builder,dev,args,device,args.out_dir/f"eval_{u:05d}",args.arm,u,u==3000); history.append({"update":u,**ev["raw_errors"],"inference_time":ev["mean_t_neural_s"]}); torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":base_opt.state_dict(),"iteration":u,"preflight":pre},args.out_dir/f"model_update_{u:05d}.pth")
    with torch.no_grad():
        adapter_output_norm=0.0 if adapter is None else float((forward(model,builder,x)-forward(backbone,builder,x)).norm().detach().cpu())
    legacy.write_rows(args.out_dir/"training_curve.csv",curve); legacy.write_rows(args.out_dir/"aggregate_metrics.csv",[{"arm":args.arm,**r} for r in history]); (args.out_dir/"runtime.json").write_text(json.dumps({"parameter_count":sum(p.numel() for p in model.parameters()),"added_parameter_count":0 if adapter is None else sum(p.numel() for p in adapter.parameters()),"peak_gpu_memory_allocated":peak_a,"peak_gpu_memory_reserved":peak_r,"training_wall_seconds":time.monotonic()-started,"inference_latency":history[-1].get("inference_time"),"adapter_output_norm":adapter_output_norm,"adapter_parameter_norm":0.0 if adapter is None else float(sum(p.detach().norm() for p in adapter.parameters()))},indent=2),encoding="utf-8")
    meta={"status":"REVIEW_REQUIRED","arm":args.arm,"manifest_sha256":core.sha256(args.manifest),"scorer_sha256":core.sha256(args.kit_root/"scoring.py"),"initialization_checkpoint_sha256":core.sha256(args.checkpoint),"seed":args.seed,"optimizer":"AdamW","lr":args.lr,"physical_batch":8,"gradient_accumulation":False,"feature_config":vars(config),"locked_final_accessed":False,"full_data_accessed":False,"sps_accessed":False,"codabench_accessed":False}; (args.out_dir/"run_metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--arm",choices=("C0","V1","A1","A2"),required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--kit-root",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True); p.add_argument("--updates",type=int,default=1500); p.add_argument("--milestones",type=int,nargs="+",default=[500,1000,1500]); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--accumulation-steps",type=int,default=1); p.add_argument("--workers",type=int,default=2); p.add_argument("--max-windows",type=int); p.add_argument("--lr",type=float,default=1e-5); p.add_argument("--seed",type=int,default=20260901); p.add_argument("--preflight-only",action="store_true"); run(p.parse_args())
if __name__=="__main__": main()
