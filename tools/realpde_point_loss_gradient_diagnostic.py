#!/usr/bin/env python3
"""Train-only loss-gradient diagnostic for the frozen Point-V1 LOCAL3 model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import realpde_loss_official_v9 as core
from realpde_point_v1_local3_runner import BATCH, PackedDataset, Local3MLP, fixed_order, loader


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atom_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def write_status(out: Path, state: str, stage: str, **extra: object) -> None:
    atom_json(out / "status.json", {"status": state, "stage": stage, "pid": os.getpid(), "time": time.time(), **extra})


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def flat_grad(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> torch.Tensor:
    pieces = [g.detach().reshape(-1) for g, p in zip(grads, params) if g is not None and p.requires_grad]
    return torch.cat(pieces) if pieces else torch.zeros(1, device=params[0].device)


def classify(median_ratio: float) -> str:
    if median_ratio >= 10: return "TKE_GRADIENT_STRONGLY_DOMINANT"
    if median_ratio >= 5: return "TKE_GRADIENT_DOMINANT"
    if median_ratio >= 0.2: return "GRADIENT_BALANCED"
    return "MSE_GRADIENT_DOMINANT"


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--seed", type=int, default=20260901); p.add_argument("--batches", type=int, default=32); p.add_argument("--device", default="cuda"); p.add_argument("--smoke", action="store_true")
    args = p.parse_args(); out = args.out_dir
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); write_status(out, "RUNNING", "initializing", locked_final_accessed=False, dev_accessed=False, scorer_used=False, optimizer_step=False)
    try:
        if not args.checkpoint.is_file(): raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8")); paths = [args.data_root / row["file"] for row in manifest["train"]]
        missing = [str(x) for x in paths if not x.is_file()]
        if missing: raise FileNotFoundError(f"train files missing: {missing[:3]}")
        set_seed(args.seed); device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        ds = PackedDataset(paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False); count = 2 if args.smoke else args.batches
        order = fixed_order(len(ds), count * BATCH, args.seed); dl = loader(ds, order); ckpt = torch.load(args.checkpoint, map_location=device); model = Local3MLP().to(device); model.load_state_dict(ckpt["model_state_dict"], strict=True); model.eval(); params = [x for x in model.parameters() if x.requires_grad]
        atom_json(out / "run_metadata.json", {"checkpoint":str(args.checkpoint),"checkpoint_sha256":sha256(args.checkpoint),"runner_script_sha256":sha256(Path(__file__)),"local3_runner_sha256":sha256(Path(__import__('realpde_point_v1_local3_runner').__file__)),"manifest_sha256":sha256(args.manifest),"seed":args.seed,"batch_size":BATCH,"train_batches":count,"train_only":True,"pipeline":"B3_PACKED","device":str(device),"optimizer_step":False,"dev_accessed":False,"locked_final_accessed":False,"scorer_used":False})
        rows = []; write_status(out, "RUNNING", "gradient_batches", target_batches=count)
        for batch_id, (x, y, _, _) in enumerate(dl):
            if batch_id >= count: break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); model.zero_grad(set_to_none=True); pred = model(x); parts = core.loss_parts(pred, y); mse = parts["mse"]; tke_raw = parts["tke"]; tke_weighted = 0.05 * tke_raw; total = mse + tke_weighted
            model.zero_grad(set_to_none=True); gm = torch.autograd.grad(mse, params, retain_graph=True, allow_unused=True); model.zero_grad(set_to_none=True); gt = torch.autograd.grad(tke_weighted, params, retain_graph=False, allow_unused=True); model.zero_grad(set_to_none=True)
            vm, vt = flat_grad(gm, params), flat_grad(gt, params); nm, nt = torch.linalg.vector_norm(vm), torch.linalg.vector_norm(vt); ratio = nt / nm if float(nm) != 0 else torch.tensor(float("inf"), device=device); cosine = torch.dot(vm, vt) / (nm * nt) if float(nm) != 0 and float(nt) != 0 else torch.tensor(0.0, device=device)
            rows.append({"batch":batch_id,"mse":float(mse.detach().cpu()),"tke_raw":float(tke_raw.detach().cpu()),"tke_weighted":float(tke_weighted.detach().cpu()),"total_loss":float(total.detach().cpu()),"grad_mse_norm":float(nm.cpu()),"grad_tke_norm":float(nt.cpu()),"grad_ratio":float(ratio.cpu()),"cosine":float(cosine.cpu()),"dot":float(torch.dot(vm,vt).cpu()),"finite":bool(torch.isfinite(total).item() and torch.isfinite(nm).item() and torch.isfinite(nt).item())})
            if (batch_id + 1) % 8 == 0: write_status(out, "RUNNING", "gradient_batches", current_batch=batch_id+1, target_batches=count)
        if len(rows) != count: raise RuntimeError(f"expected {count} batches, got {len(rows)}")
        with (out / "gradient_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        def stats(key: str) -> dict:
            a = np.asarray([r[key] for r in rows], dtype=float); return {"mean":float(np.mean(a)),"median":float(np.median(a)),"p25":float(np.percentile(a,25)),"p75":float(np.percentile(a,75)),"min":float(np.min(a)),"max":float(np.max(a))}
        summary = {"batches":count,"all_finite":all(r["finite"] for r in rows),"grad_ratio":stats("grad_ratio"),"cosine":stats("cosine"),"mse":stats("mse"),"tke_raw":stats("tke_raw"),"tke_weighted":stats("tke_weighted"),"scalar_contribution_ratio":stats("tke_weighted") | {"mean_over_mse":float(np.mean([r["tke_weighted"] / r["mse"] if r["mse"] else float("inf") for r in rows]))},"diagnostic_label":classify(float(np.median([r["grad_ratio"] for r in rows]))),"cosine_interpretation":"negative=conflict, near-zero=orthogonal, positive=aligned; not a gate","optimizer_step":False,"dev_accessed":False,"locked_final_accessed":False,"scorer_used":False}
        atom_json(out / "summary.json", summary)
        lines = ["# Point LOCAL3 loss-gradient diagnostic", "", "TRAIN-ONLY diagnostic; no optimizer.step, dev, locked-final, scorer or Codabench.", "", f"- Checkpoint: `{args.checkpoint}` (SHA-256 `{sha256(args.checkpoint)}`)", f"- Manifest SHA-256: `{sha256(args.manifest)}`; runner SHA-256: `{sha256(Path(__file__))}`", f"- B3_PACKED; seed `{args.seed}`; batch `{BATCH}`; batches `{count}`; device `{device}`", f"- Scalar contribution ratio mean `(0.05*TKE)/MSE`: `{summary['scalar_contribution_ratio']['mean_over_mse']:.6g}`", f"- grad_ratio mean/median: `{summary['grad_ratio']['mean']:.6g}` / `{summary['grad_ratio']['median']:.6g}`", f"- cosine mean/median: `{summary['cosine']['mean']:.6g}` / `{summary['cosine']['median']:.6g}`", f"- Diagnostic label: **`{summary['diagnostic_label']}`**", "", "No follow-up loss change or training is authorized by this diagnostic."]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); (out / "README_FOR_CHATGPT.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); write_status(out, "DONE", "complete", diagnostic_label=summary["diagnostic_label"], batches=count, optimizer_step=False, dev_accessed=False, locked_final_accessed=False, scorer_used=False); (out / "DONE").touch()
    except Exception:
        write_status(out, "FAILED", "exception", traceback=traceback.format_exc(), optimizer_step=False, dev_accessed=False, locked_final_accessed=False, scorer_used=False); raise


if __name__ == "__main__": main()
