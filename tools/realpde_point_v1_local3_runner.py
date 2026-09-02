#!/usr/bin/env python3
"""Point-V1 LOCAL3 runner: RAM pipeline, LR sanity, and gated residual run.

The runner is intentionally self-contained and conservative about state: all
long stages update an atomic status.json and only the pre-registered Phase 3A
gate may extend training from 1500 to 7500 updates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realpde_loss_official_v9 as core  # noqa: E402
import realpde_point_v0 as point  # noqa: E402
from realpde_p0_data import H5WindowDataset  # noqa: E402

SEED = 20260901
BATCH = 8
IN_STEPS = 20
OUT_STEPS = 20
LOSS_WEIGHTS = {"mse": 1.0, "tke": 0.05}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atom_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def write_status(out: Path, state: str, stage: str, **extra: object) -> None:
    atom_json(out / "status.json", {"status": state, "stage": stage, "pid": os.getpid(), "time": time.time(), **extra})


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def host_available_mb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def proc_children(pid: int) -> list[int]:
    result = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            text = (p / "status").read_text()
            ppid = next(int(x.split()[1]) for x in text.splitlines() if x.startswith("PPid:"))
            if ppid == pid: result.append(int(p.name))
        except (OSError, StopIteration, ValueError):
            continue
    return result


def process_tree_rss_mb(root: int | None = None) -> float:
    root = os.getpid() if root is None else root
    todo, seen, total = [root], set(), 0
    while todo:
        pid = todo.pop()
        if pid in seen: continue
        seen.add(pid)
        try:
            for line in (Path(f"/proc/{pid}/status")).read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]); break
        except OSError: pass
        todo.extend(proc_children(pid))
    return total / 1024.0


class SequenceDataset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int]): self.base, self.indices = base, list(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.base[self.indices[i]]


class RAMDataset(H5WindowDataset):
    """Existing trajectory cache: cache u/v separately, stack only per window."""
    def __init__(self, paths: Sequence[Path], **kwargs):
        super().__init__(paths, **kwargs)
        self.cache: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
        self.cache_bytes = 0; started = time.perf_counter()
        for path in self.paths:
            with h5py.File(path, "r") as f:
                u = np.asarray(self._field(f, "u")[:, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
                v = np.asarray(self._field(f, "v")[:, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
                self.cache[str(path)] = (u, v, float(f["re"][()]), float(f["aoa"][()]))
                self.cache_bytes += u.nbytes + v.nbytes
        self.cache_build_seconds = time.perf_counter() - started

    def __getitem__(self, index: int):
        ref = self.refs[index]; u, v, re, aoa = self.cache[str(ref.path)]
        sl = slice(ref.start, ref.start + self.in_steps + self.out_steps)
        p = np.zeros_like(u[sl]); full = torch.from_numpy(np.stack([u[sl], v[sl], p], axis=-1))
        return full[:self.in_steps], full[self.in_steps:], torch.tensor([re, aoa], dtype=torch.float32), torch.tensor(index)


class PackedDataset(H5WindowDataset):
    """Packed cache: one contiguous float32 [T,H,W,3] per trajectory."""
    def __init__(self, paths: Sequence[Path], **kwargs):
        super().__init__(paths, **kwargs)
        self.cache: dict[str, tuple[np.ndarray, float, float]] = {}
        self.cache_bytes = 0; started = time.perf_counter()
        for path in self.paths:
            with h5py.File(path, "r") as f:
                u = np.asarray(self._field(f, "u")[:, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
                v = np.asarray(self._field(f, "v")[:, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
                packed = np.empty((*u.shape, 3), dtype=np.float32)
                packed[..., 0] = u; packed[..., 1] = v; packed[..., 2] = 0.0
                self.cache[str(path)] = (packed, float(f["re"][()]), float(f["aoa"][()]))
                self.cache_bytes += packed.nbytes
        self.cache_build_seconds = time.perf_counter() - started

    def __getitem__(self, index: int):
        ref = self.refs[index]; packed, re, aoa = self.cache[str(ref.path)]
        sl = slice(ref.start, ref.start + self.in_steps + self.out_steps)
        full = torch.from_numpy(packed[sl])
        return full[:self.in_steps], full[self.in_steps:], torch.tensor([re, aoa], dtype=torch.float32), torch.tensor(index)


def dataset_for(kind: str, paths: Sequence[Path]) -> H5WindowDataset:
    kw = dict(in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    if kind == "B3_RAM": return RAMDataset(paths, **kw)
    if kind == "B3_PACKED": return PackedDataset(paths, **kw)
    raise ValueError(kind)


def fixed_order(n: int, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed); base = np.arange(n, dtype=np.int64); rng.shuffle(base)
    return [int(base[i % n]) for i in range(count)]


def loader(ds: Dataset, order: Sequence[int], workers: int = 0) -> DataLoader:
    return DataLoader(SequenceDataset(ds, order), batch_size=BATCH, shuffle=False, num_workers=workers,
                      pin_memory=True, drop_last=True)


def data_equivalence(base: H5WindowDataset, cand: H5WindowDataset, order: Sequence[int], n: int) -> dict:
    max_x = max_y = 0.0; passed = True; rows = []
    for idx in list(order)[:n]:
        a, b = base[idx], cand[idx]
        for k, label in ((0, "x"), (1, "y")):
            aa, bb = a[k].numpy(), b[k].numpy(); d = float(np.max(np.abs(aa - bb)))
            if label == "x": max_x = max(max_x, d)
            else: max_y = max(max_y, d)
            passed &= aa.shape == bb.shape and aa.dtype == bb.dtype and d == 0.0
        passed &= base.refs[idx] == cand.refs[idx]
        if len(rows) < 5: rows.append({"index": int(idx), "trajectory": base.refs[idx].path.name, "start": base.refs[idx].start, "shape": list(a[0].shape), "dtype": str(a[0].dtype)})
    return {"windows_checked": min(n, len(order)), "max_abs_diff_x": max_x, "max_abs_diff_y": max_y,
            "passed": bool(passed), "trajectory_window_order_equal": bool(passed), "examples": rows}


def sync(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize()


def profile_candidate(kind: str, paths: list[Path], order: list[int], warmup: int, measured: int, repeats: int, out: Path, device: torch.device) -> dict:
    started = time.perf_counter(); ds = dataset_for(kind, paths); cache_build = time.perf_counter() - started
    base = H5WindowDataset(paths, in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    eq = data_equivalence(base, ds, order, 100); atom_json(out / "equivalence.json", eq)
    if not eq["passed"]: raise RuntimeError(f"DATA_EQUIVALENCE failed for {kind}")
    dl = loader(ds, order); vals = []; waits = []; h2ds = []; fwds = []; bwds = []; peak = process_tree_rss_mb(); total = warmup + measured
    model = point.PointMLP().to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-4); model.train()
    for rep in range(repeats):
        it = iter(dl)
        for i in range(total):
            t0 = time.perf_counter(); x, y, _, _ = next(it); t1 = time.perf_counter(); sync(device)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); sync(device); t2 = time.perf_counter()
            pred = model(x, residual=True); parts = core.loss_parts(pred, y); loss = parts["mse"] + 0.05 * parts["tke"]; sync(device); t3 = time.perf_counter()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sync(device); t4 = time.perf_counter(); peak = max(peak, process_tree_rss_mb())
            if i >= warmup:
                waits.append((t1-t0)*1000); h2ds.append((t2-t1)*1000); fwds.append((t3-t2)*1000); bwds.append((t4-t3)*1000); vals.append((t4-t0)*1000)
    a = np.asarray(vals); mean = float(a.mean()); wait = float(np.mean(waits));
    result = {"candidate": kind, "profile": "B_TRAINING_PATH", "warmup": warmup, "measured": measured, "repeats": repeats,
              "windows": len(vals)*BATCH, "windows_per_s": len(vals)*BATCH/(float(a.sum())/1000), "step_latency_ms": mean,
              "latency_p50_ms": float(np.percentile(a, 50)), "latency_p95_ms": float(np.percentile(a, 95)), "data_wait_ms": wait,
              "data_wait_ratio": wait/mean, "h2d_ms": float(np.mean(h2ds)), "forward_ms": float(np.mean(fwds)),
              "backward_ms": float(np.mean(bwds)), "cache_build_seconds": float(cache_build), "cache_bytes": int(getattr(ds, "cache_bytes", 0)),
              "host_mem_available_mb": host_available_mb(), "process_tree_peak_rss_mb": peak, "data_equivalence": "PASS"}
    atom_json(out / "metrics.json", result); return result


class Local3MLP(nn.Module):
    def __init__(self):
        super().__init__(); dims = (362, 256, 256, 256, 128, 40); layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if b != 40: layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        b, t, h, w, c = x.shape
        uv = x[..., :2].permute(0, 1, 4, 2, 3).reshape(b*t, 2, h, w)
        uv = F.pad(uv, (1, 1, 1, 1), mode="replicate")
        patches = F.unfold(uv, kernel_size=3, stride=1).transpose(1, 2).reshape(b, t, h*w, 18)
        patches = patches.permute(0, 2, 1, 3).reshape(b, h*w, 360)
        xx = torch.linspace(-1., 1., w, device=x.device, dtype=x.dtype); yy = torch.linspace(-1., 1., h, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij"); pos = torch.stack((gx, gy), dim=-1).reshape(1, h*w, 2).expand(b, -1, -1)
        out = self.net(torch.cat((pos, patches), dim=-1).reshape(b*h*w, 362)).reshape(b, h*w, 20, 2)
        out = out.reshape(b, h, w, 20, 2).permute(0, 3, 1, 2, 4) + x[:, -1:, :, :, :2]
        return torch.cat((out, torch.zeros_like(out[..., :1])), dim=-1)


def train_loss(model: Local3MLP, dl: DataLoader, lr: float, updates: int, seed: int, out: Path, device: torch.device, start_step: int = 0, optimizer: torch.optim.Optimizer | None = None) -> tuple[torch.optim.Optimizer, list[dict], int, float, bool]:
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed); model.train(); opt = optimizer or torch.optim.AdamW(model.parameters(), lr=lr); it = iter(dl)
    if start_step:
        for _ in range(start_step % max(len(dl), 1)):
            try: next(it)
            except StopIteration: it = iter(dl); next(it)
    history = []; started = time.monotonic(); finite = True
    for step in range(start_step + 1, updates + 1):
        try: x, y, _, _ = next(it)
        except StopIteration: it = iter(dl); x, y, _, _ = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); pred = model(x); parts = core.loss_parts(pred, y); loss = parts["mse"] + 0.05 * parts["tke"]
        opt.zero_grad(set_to_none=True); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step()
        finite = finite and bool(torch.isfinite(loss).item()) and math.isfinite(grad)
        history.append({"update": step, "total_loss": float(loss.detach().cpu()), "mse": float(parts["mse"].detach().cpu()), "tke": float(parts["tke"].detach().cpu()), "gradient_norm": grad, "finite": finite})
        if step % 20 == 0 or step == updates: write_status(out, "RUNNING", "phase2_lr_sanity" if updates == 500 else "phase3_train", current_update=step, target_update=updates)
        if not finite: break
    return opt, history, (history[-1]["update"] if history else start_step), time.monotonic() - started, finite


def save_history(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def evaluate_dev(model: nn.Module, paths: list[Path], kind: str, kit_root: Path, out: Path, device: torch.device) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds = dataset_for(kind, paths); order = list(range(len(ds))); dl = loader(ds, order); pred, target = [], []; elapsed = 0.0; model.eval()
    with torch.no_grad():
        for x, y, _, _ in dl:
            x = x.to(device, non_blocking=True); sync(device); t = time.perf_counter(); p = model(x); sync(device); elapsed += time.perf_counter() - t
            pred.append(p.cpu().numpy().astype(np.float32)); target.append(y.numpy().astype(np.float32))
    pa, ta = np.concatenate(pred), np.concatenate(target); result = core.score_bundle(kit_root, pa, ta, elapsed / len(ds), out); rows, anatomy = core.trajectory_rows(ds, pa, ta, kit_root); result["windows"] = len(ds); result["trajectories"] = len(rows); result["trajectory_anatomy"] = anatomy
    with (out / "trajectory_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / "predictions_targets.npz", prediction=pa, target=ta); atom_json(out / "evaluation.json", result); return result


def persistence_model() -> nn.Module:
    class P(nn.Module):
        def forward(self, x):
            uv = x[:, -1:, :, :, :2].expand(-1, 20, -1, -1, -1)
            return torch.cat((uv, torch.zeros_like(uv[..., :1])), dim=-1)
    return P()


def bootstrap_rows(base_csv: Path, cand_csv: Path, seed: int) -> dict:
    def read(p):
        with p.open() as f: return {r["trajectory_id"]: {k: float(r[k]) for k in ("rel_l2", "tke", "mvpe")} for r in csv.DictReader(f)}
    b, c = read(base_csv), read(cand_csv); names = sorted(set(b) & set(c));
    if len(names) != 16: raise ValueError(f"expected 16 paired dev trajectories, got {len(names)}")
    rng = np.random.default_rng(seed); result = {}
    for m in ("rel_l2", "tke", "mvpe"):
        vals = np.asarray([(b[n][m]-c[n][m])/b[n][m]*100 for n in names]); boots = np.asarray([vals[rng.integers(0,16,16)].mean() for _ in range(10000)])
        result[m] = {"macro_mean_improvement_pct": float(vals.mean()), "bootstrap_95_low_pct": float(np.percentile(boots,2.5)), "bootstrap_95_high_pct": float(np.percentile(boots,97.5)), "trajectory_win_rate": float(np.mean(vals > 0))}
    return result


def screen_gate(base: dict, cand: dict) -> tuple[bool, dict]:
    b, c = base["raw_errors"], cand["raw_errors"]; imp = {m: (b[m]-c[m])/b[m]*100 for m in ("rel_l2", "tke", "mvpe")}
    passed = imp["rel_l2"] > 0 and imp["mvpe"] > 0 and imp["tke"] >= -10
    return passed, {"improvement_pct": imp, "gate": "Rel-L2>0 and MVPE>0 and TKE degradation<=10%", "passed": passed}


def final_gate(base: dict, cand: dict, stability: dict) -> tuple[str, dict]:
    b, c = base["raw_errors"], cand["raw_errors"]; imp = {m: (b[m]-c[m])/b[m]*100 for m in ("rel_l2", "tke", "mvpe")}
    stable = all(stability[m]["macro_mean_improvement_pct"] > 0 for m in stability)
    if imp["rel_l2"] >= 10 and imp["mvpe"] >= 10 and imp["tke"] >= -5 and stable: d = "STRONG_GO_LOCAL_POINT"
    elif imp["rel_l2"] >= 5 and imp["mvpe"] >= 5 and imp["tke"] >= -10: d = "GO_LOCAL_POINT"
    else: d = "STOP_LOCAL_POINT"
    return d, {"improvement_pct": imp, "bootstrap_direction_stable": stable}


def spatial_diagnostic(prediction: np.ndarray, target: np.ndarray, out: Path) -> dict:
    """Pixel-space diagnostic using the frozen finite/valid mask convention."""
    uvp, uvy = prediction[..., :2], target[..., :2]
    valid = np.isfinite(uvp).all(axis=(0, 1, 4)) & np.isfinite(uvy).all(axis=(0, 1, 4))
    err = np.sqrt(np.mean(np.sum((uvp - uvy) ** 2, axis=(0, 1)), axis=-1))
    rows = [{"row": i, "column": j, "velocity_rmse": float(err[i, j]), "valid": bool(valid[i, j])}
            for i in range(err.shape[0]) for j in range(err.shape[1])]
    write_csv(out / "spatial_error.csv", rows)
    return {"valid_pixels": int(valid.sum()), "total_pixels": int(valid.size), "mask": "finite u/v prediction and target"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    keys = sorted({k for r in rows for k in r});
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def phase1(args, paths: list[Path], out: Path, device: torch.device) -> str:
    p1 = out / "phase1"; p1.mkdir(exist_ok=True); n = len(H5WindowDataset(paths, in_steps=20, out_steps=20, stride=20, sub_sample=2)); max_batches = max(args.coarse_warmup+args.coarse_measured, args.formal_warmup+args.formal_measured); order = fixed_order(n, max_batches*BATCH, args.seed); atom_json(p1 / "fixed_order.json", {"seed":args.seed,"indices":order,"source_windows":n,"batch_size":BATCH})
    rows = []
    for kind in ("B3_RAM", "B3_PACKED"):
        sub = p1 / kind; sub.mkdir(exist_ok=True); write_status(out, "RUNNING", "phase1_coarse", current_candidate=kind)
        rows.append(profile_candidate(kind, paths, order, args.coarse_warmup, args.coarse_measured, 1, sub, device)); write_csv(p1 / "coarse_metrics.csv", rows)
    winner = max(rows, key=lambda r: r["windows_per_s"])["candidate"]; atom_json(p1 / "coarse_summary.json", {"rows":rows,"winner":winner,"data_equivalence":"PASS"})
    write_status(out, "RUNNING", "phase1_formal", winner=winner)
    sub = p1 / "formal"; sub.mkdir(exist_ok=True); formal = profile_candidate(winner, paths, order, args.formal_warmup, args.formal_measured, args.formal_repeats, sub, device); atom_json(p1 / "formal_metrics.json", formal); atom_json(out / "phase1_freeze.json", {"POINT_RAM_PIPELINE_V1":winner,"coarse":rows,"formal":formal})
    return winner


def phase2(args, train_paths: list[Path], kind: str, out: Path, device: torch.device) -> float:
    p2 = out / "phase2_lr_sanity"; p2.mkdir(exist_ok=True); ds = dataset_for(kind, train_paths); order = fixed_order(len(ds), args.lr_updates*BATCH, args.seed); dl = loader(ds, order)
    finals = {}
    for lr in (1e-5, 1e-4):
        label = f"lr_{lr:.0e}"; sub = p2 / label; sub.mkdir(exist_ok=True); set_seed(args.seed); model = Local3MLP().to(device); _, hist, actual, seconds, finite = train_loss(model, dl, lr, args.lr_updates, args.seed, sub, device); save_history(sub / "loss_curve.csv", hist); final = hist[-1] if hist else {"total_loss":float("inf")}; finals[label] = {"lr":lr,"actual_updates":actual,"finite":finite,"wall_seconds":seconds,"final_total_loss":final["total_loss"],"final_mse":final.get("mse"),"final_tke":final.get("tke"),"mean_total_loss":float(np.mean([h["total_loss"] for h in hist])) if hist else float("inf")}
    a, b = finals["lr_1e-05"], finals["lr_1e-04"]; use_high = bool(b["finite"] and a["finite"] and (b["final_total_loss"] <= 0.95*a["final_total_loss"] or b["mean_total_loss"] <= 0.95*a["mean_total_loss"]))
    chosen = 1e-4 if use_high else 1e-5; atom_json(p2 / "summary.json", {"runs":finals,"chosen_lr":chosen,"criterion":"1e-4 finite and final or mean train loss at least 5% lower"}); return chosen


def phase3(args, train_paths: list[Path], dev_paths: list[Path], kind: str, lr: float, out: Path, kit: Path, device: torch.device, resume_source: Path | None = None) -> str:
    p3 = out / "phase3_local3"; p3.mkdir(exist_ok=True); ds = dataset_for(kind, train_paths); order = fixed_order(len(ds), 7500*BATCH, args.seed); dl = loader(ds, order)
    checkpoint_source = (resume_source / "phase3_local3" / "last@1500.pt") if resume_source else None
    if checkpoint_source and checkpoint_source.is_file():
        set_seed(args.seed); model = Local3MLP().to(device); opt = torch.optim.AdamW(model.parameters(), lr=lr); ckpt = torch.load(checkpoint_source, map_location=device); model.load_state_dict(ckpt["model_state_dict"]); opt.load_state_dict(ckpt["optimizer_state_dict"]); h1 = []; actual = int(ckpt.get("iteration", 1500)); finite = True; (p3 / "screening").mkdir(parents=True, exist_ok=True); shutil.copy2(checkpoint_source, p3 / "last@1500.pt")
        source_curve = checkpoint_source.parent / "screening" / "loss_curve.csv"
        if source_curve.is_file(): shutil.copy2(source_curve, p3 / "screening" / "loss_curve.csv")
    else:
        set_seed(args.seed); model = Local3MLP().to(device); opt, h1, actual, seconds, finite = train_loss(model, dl, lr, 1500, args.seed, p3 / "screening", device); save_history(p3 / "screening" / "loss_curve.csv", h1); torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"iteration":actual,"lr":lr}, p3 / "last@1500.pt")
    if not finite or actual != 1500: raise RuntimeError("Phase 3A training did not reach 1500 finite updates")
    cand = evaluate_dev(model, dev_paths, kind, kit, p3 / "dev@1500_local3", device); persist = evaluate_dev(persistence_model().to(device), dev_paths, kind, kit, p3 / "dev@1500_persist", device); passed, gate = screen_gate(persist, cand); atom_json(p3 / "screening_gate.json", gate)
    if not passed:
        atom_json(p3 / "summary.json", {"stage":"STOP_LOCAL3_EARLY","screening_gate":gate,"locked_final_accessed":False}); write_status(out,"DONE","phase3a_stop",decision="STOP_LOCAL3_EARLY"); (out / "STOP_LOCAL3_EARLY").touch(); return "STOP_LOCAL3_EARLY"
    write_status(out,"RUNNING","phase3b_train",current_update=1500,target_update=7500); opt, h2, actual, seconds2, finite = train_loss(model, dl, lr, 7500, args.seed, p3 / "full", device, start_step=1500, optimizer=opt); save_history(p3 / "full" / "loss_curve.csv", h1+h2); torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"iteration":actual,"lr":lr}, p3 / "last@7500.pt")
    if not finite or actual != 7500: raise RuntimeError("Phase 3B training did not reach 7500 finite updates")
    candf = evaluate_dev(model, dev_paths, kind, kit, p3 / "dev@7500_local3", device); persistf = evaluate_dev(persistence_model().to(device), dev_paths, kind, kit, p3 / "dev@7500_persist", device); stability = bootstrap_rows(p3 / "dev@7500_persist" / "trajectory_metrics.csv", p3 / "dev@7500_local3" / "trajectory_metrics.csv", args.seed); atom_json(p3 / "paired_bootstrap.json", stability)
    z = np.load(p3 / "dev@7500_local3" / "predictions_targets.npz"); horizon=[]
    for i in range(20):
        p, y = z["prediction"][:,i,...,:2].reshape(z["prediction"].shape[0],-1), z["target"][:,i,...,:2].reshape(z["target"].shape[0],-1); horizon.append({"horizon":i+1,"velocity_rel_l2_micro":float(np.linalg.norm(p-y)/max(np.linalg.norm(y),1e-12))})
    write_csv(p3 / "horizon_metrics.csv", horizon); spatial = spatial_diagnostic(z["prediction"], z["target"], p3); decision, detail = final_gate(persistf, candf, stability); atom_json(p3 / "summary.json", {"decision":decision,"gate":detail,"screening_gate":gate,"lr":lr,"pipeline":kind,"spatial":spatial,"locked_final_accessed":False,"codabench":False}); write_status(out,"DONE","complete",decision=decision,locked_final_accessed=False,codabench=False); (out / "DONE").touch(); return decision


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--kit-root", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--seed", type=int, default=SEED); p.add_argument("--device", default="cuda"); p.add_argument("--coarse-warmup", type=int, default=20); p.add_argument("--coarse-measured", type=int, default=100); p.add_argument("--formal-warmup", type=int, default=100); p.add_argument("--formal-measured", type=int, default=1000); p.add_argument("--formal-repeats", type=int, default=3); p.add_argument("--lr-updates", type=int, default=500); p.add_argument("--resume-source", type=Path, help="Reuse completed Phase 1/2 artifacts from a prior run and execute Phase 3 only"); p.add_argument("--smoke", action="store_true")
    args = p.parse_args(); out = args.out_dir
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); write_status(out,"RUNNING","initializing",locked_final_accessed=False,codabench=False)
    try:
        set_seed(args.seed); device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"); manifest = json.loads(args.manifest.read_text()); train_paths = [args.data_root / r["file"] for r in manifest["train"]]; dev_paths = [args.data_root / r["file"] for r in manifest["dev"]]
        if any(not x.is_file() for x in train_paths+dev_paths): raise FileNotFoundError("manifest path missing under --data-root")
        atom_json(out / "provenance.json", {"manifest_sha256":sha256(args.manifest),"runner_sha256":sha256(Path(__file__)),"seed":args.seed,"batch_size":BATCH,"train_trajectories":len(train_paths),"dev_trajectories":len(dev_paths),"locked_final_accessed":False,"codabench":False,"device":str(device),"formal":{"warmup":args.formal_warmup,"measured":args.formal_measured,"repeats":args.formal_repeats}})
        if args.smoke:
            paths = train_paths[:2]; n = len(H5WindowDataset(paths, in_steps=20,out_steps=20,stride=20,sub_sample=2,include_pressure=False)); order = fixed_order(n, 2*BATCH, args.seed); smoke = {}
            for kind in ("B3_RAM","B3_PACKED"):
                ds = dataset_for(kind, paths); smoke[kind] = data_equivalence(H5WindowDataset(paths,in_steps=20,out_steps=20,stride=20,sub_sample=2,include_pressure=False), ds, order, min(10,len(order))); x,y,_,_ = next(iter(loader(ds,order))); pred = Local3MLP().to(device)(x.to(device)); smoke[kind]["local3_shape"] = list(pred.shape); smoke[kind]["cache_bytes"] = int(ds.cache_bytes)
            atom_json(out / "smoke.json", smoke); write_status(out,"DONE","smoke_complete",smoke_pass=True); (out / "DONE").touch(); return
        if args.resume_source:
            source = args.resume_source.resolve(); frozen = json.loads((source / "frozen_choices.json").read_text(encoding="utf-8")); kind, lr = frozen["POINT_RAM_PIPELINE_V1"], float(frozen["POINT_V1_LR"]); atom_json(out / "resume_metadata.json", {"resumed_from":str(source),"phase1_phase2_reused":True,"phase1_freeze":str(source / "phase1_freeze.json"),"phase2_summary":str(source / "phase2_lr_sanity" / "summary.json")})
        else:
            kind = phase1(args, train_paths, out, device); write_status(out,"RUNNING","phase2_lr_sanity",pipeline=kind); lr = phase2(args, train_paths, kind, out, device); atom_json(out / "frozen_choices.json", {"POINT_RAM_PIPELINE_V1":kind,"POINT_V1_LR":lr})
        atom_json(out / "frozen_choices.json", {"POINT_RAM_PIPELINE_V1":kind,"POINT_V1_LR":lr}); write_status(out,"RUNNING","phase3a_screening",pipeline=kind,lr=lr); decision = phase3(args, train_paths, dev_paths, kind, lr, out, args.kit_root, device, args.resume_source); atom_json(out / "final_summary.json", {"phase1_pipeline":kind,"phase2_lr":lr,"phase3_decision":decision,"locked_final_accessed":False,"codabench":False})
    except Exception:
        write_status(out,"FAILED","exception",traceback=traceback.format_exc(),locked_final_accessed=False,codabench=False); raise


if __name__ == "__main__": main()
