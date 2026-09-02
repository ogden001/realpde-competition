#!/usr/bin/env python3
"""Point-V0 data-pipeline benchmark.

This is an engineering benchmark only: it never trains or scores a model.
Every candidate consumes one fixed, pre-generated sequence of train windows so
that differences are data-access differences, not sampling differences.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import realpde_point_v0 as point
from realpde_p0_data import H5WindowDataset, WindowRef


def atom_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def status(out: Path, state: str, stage: str, **extra: object) -> None:
    atom_json(out / "status.json", {"status": state, "stage": stage,
             "pid": os.getpid(), "time": time.time(), **extra})


class SequenceDataset(Dataset):
    """Repeat a fixed list of source indices without changing their order."""
    def __init__(self, base: Dataset, indices: Sequence[int]):
        self.base, self.indices = base, list(indices)
    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, i: int): return self.base[self.indices[i]]


class HandleDataset(H5WindowDataset):
    """H5WindowDataset with one lazy HDF5 handle per trajectory per worker."""
    def __init__(self, paths: Sequence[Path], **kwargs):
        super().__init__(paths, **kwargs)
        self._handles: dict[str, h5py.File] = {}

    def __getstate__(self):
        d = self.__dict__.copy(); d["_handles"] = {}; return d

    def __getitem__(self, index: int):
        ref = self.refs[index]; key = str(ref.path)
        f = self._handles.get(key)
        if f is None or not f.id.valid:
            f = h5py.File(ref.path, "r"); self._handles[key] = f
        total = self.in_steps + self.out_steps; sl = slice(ref.start, ref.start + total)
        u = np.asarray(self._field(f, "u")[sl, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
        v = np.asarray(self._field(f, "v")[sl, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
        if self.include_pressure and ("p" in f or "measured_data/p" in f):
            p = np.asarray(self._field(f, "p")[sl, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
        else: p = np.zeros_like(u)
        full = torch.from_numpy(np.stack([u, v, p], axis=-1))
        condition = torch.tensor([float(f["re"][()]), float(f["aoa"][()])], dtype=torch.float32)
        return full[:self.in_steps], full[self.in_steps:], condition, torch.tensor(index, dtype=torch.long)


class RAMDataset(H5WindowDataset):
    """Trajectory-level RAM cache, preserving HDF5 slice dtype and semantics."""
    def __init__(self, paths: Sequence[Path], **kwargs):
        super().__init__(paths, **kwargs)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = {}
        self.cache_bytes = 0
        for path in self.paths:
            with h5py.File(path, "r") as f:
                u = np.asarray(self._field(f, "u")[:, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
                v = np.asarray(self._field(f, "v")[:, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
                if self.include_pressure and ("p" in f or "measured_data/p" in f):
                    p = np.asarray(self._field(f, "p")[:, ::self.sub_sample, ::self.sub_sample], dtype=np.float32)
                else: p = np.zeros_like(u)
                self._cache[str(path)] = (u, v, p, float(f["re"][()]), float(f["aoa"][()]))
                self.cache_bytes += u.nbytes + v.nbytes + p.nbytes

    def __getitem__(self, index: int):
        ref = self.refs[index]; u, v, p, re, aoa = self._cache[str(ref.path)]
        sl = slice(ref.start, ref.start + self.in_steps + self.out_steps)
        full = torch.from_numpy(np.stack([u[sl], v[sl], p[sl]], axis=-1))
        return full[:self.in_steps], full[self.in_steps:], torch.tensor([re, aoa], dtype=torch.float32), torch.tensor(index, dtype=torch.long)


@dataclass(frozen=True)
class Candidate:
    name: str
    dataset_kind: str
    workers: int
    persistent: bool
    prefetch: int = 2


def candidates() -> list[Candidate]:
    out = [Candidate("B0_CURRENT", "h5", 2, False)]
    for w in (0, 2, 4, 8):
        out.append(Candidate(f"B1_W{w}_P0", "h5", w, False))
        if w:
            out.append(Candidate(f"B1_W{w}_P1", "h5", w, True))
    out.append(Candidate("B2_HANDLE_W2", "handle", 2, False))
    out.append(Candidate("B3_RAM_W0", "ram", 0, False))
    return out


def make_dataset(kind: str, paths: Sequence[Path]) -> H5WindowDataset:
    kwargs = dict(in_steps=20, out_steps=20, stride=20, sub_sample=2, include_pressure=False)
    if kind == "h5": return H5WindowDataset(paths, **kwargs)
    if kind == "handle": return HandleDataset(paths, **kwargs)
    if kind == "ram": return RAMDataset(paths, **kwargs)
    raise ValueError(kind)


def make_loader(ds: Dataset, indices: Sequence[int], c: Candidate, batch: int) -> DataLoader:
    seq = SequenceDataset(ds, indices)
    kw = dict(batch_size=batch, shuffle=False, num_workers=c.workers, pin_memory=True,
              drop_last=True)
    if c.workers:
        kw["persistent_workers"] = c.persistent
        kw["prefetch_factor"] = c.prefetch
    return DataLoader(seq, **kw)


def rss_mb() -> float:
    # Linux reports KiB; macOS reports bytes.  Remote benchmark is Linux.
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(v / 1024.0 if v > 10_000 else v / (1024.0 * 1024.0))


def eq_check(ref_ds: Dataset, cand_ds: Dataset, base_indices: Sequence[int], n: int) -> dict:
    n = min(n, len(base_indices)); max_x = max_y = 0.0; passed = True; details = []
    for pos in range(n):
        idx = int(base_indices[pos]); a = ref_ds[idx]; b = cand_ds[idx]
        for k, label in ((0, "x"), (1, "y")):
            aa, bb = a[k].numpy(), b[k].numpy()
            d = float(np.max(np.abs(aa - bb))) if aa.size else 0.0
            if label == "x": max_x = max(max_x, d)
            else: max_y = max(max_y, d)
            passed &= aa.shape == bb.shape and aa.dtype == bb.dtype and d == 0.0
        passed &= a[3].item() == b[3].item()
        if len(details) < 3: details.append({"position": pos, "index": idx, "shape": list(a[0].shape)})
    return {"windows_checked": n, "max_abs_diff_x": max_x, "max_abs_diff_y": max_y,
            "passed": bool(passed), "ordering": "fixed source-index order", "examples": details}


def timed_data(loader: DataLoader, warmup: int, measured: int, repeats: int, out: Path, c: Candidate) -> dict:
    vals = []; ready = 0; iterator = iter(loader)
    total_batches = warmup + measured
    for i in range(total_batches):
        t = time.perf_counter(); batch = next(iterator); dt = (time.perf_counter() - t) * 1000
        if i >= warmup: vals.append(dt); ready += int(batch[0].shape[0])
    # Repeats are performed by rebuilding the same loader; no model state involved.
    for rep in range(1, repeats):
        iterator = iter(loader)
        for i in range(total_batches):
            t = time.perf_counter(); batch = next(iterator); dt = (time.perf_counter() - t) * 1000
            if i >= warmup: vals.append(dt); ready += int(batch[0].shape[0])
    a = np.asarray(vals, dtype=np.float64)
    return {"profile": "A_DATA_ONLY", "candidate": c.name, "batches": int(a.size),
            "windows": ready, "batches_per_s": float(1000 / a.mean()), "windows_per_s": float(ready / (a.sum() / 1000)),
            "latency_mean_ms": float(a.mean()), "latency_p50_ms": float(np.percentile(a, 50)),
            "latency_p95_ms": float(np.percentile(a, 95)), "data_wait_ms": float(a.mean()),
            "data_wait_ratio": 1.0, "rss_mb": rss_mb(), "cache_bytes": getattr(loader.dataset.base, "cache_bytes", 0)}


def timed_train(loader: DataLoader, c: Candidate, warmup: int, measured: int, repeats: int, device: torch.device) -> dict:
    vals = []; waits = []; h2ds = []; fwds = []; bwds = []; opt = None
    model = point.PointMLP().to(device); model.train(); opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total = warmup + measured
    for rep in range(repeats):
        iterator = iter(loader)
        for i in range(total):
            t0 = time.perf_counter(); x, y, _, _ = next(iterator); t1 = time.perf_counter()
            if device.type == "cuda": torch.cuda.synchronize()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if device.type == "cuda": torch.cuda.synchronize()
            t2 = time.perf_counter(); pred = model(x, residual=True); parts = point.core.loss_parts(pred, y)
            loss = parts["mse"] + 0.05 * parts["tke"]
            if device.type == "cuda": torch.cuda.synchronize()
            t3 = time.perf_counter(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if device.type == "cuda": torch.cuda.synchronize()
            t4 = time.perf_counter()
            if i >= warmup:
                waits.append((t1-t0)*1000); h2ds.append((t2-t1)*1000); fwds.append((t3-t2)*1000); bwds.append((t4-t3)*1000); vals.append((t4-t0)*1000)
    def m(a): return float(np.mean(a))
    total_ms, wait_ms = m(vals), m(waits)
    return {"profile": "B_TRAINING_PATH", "candidate": c.name, "batches": len(vals), "windows": len(vals) * 8,
            "batches_per_s": 1000/total_ms, "windows_per_s": len(vals)*8/(sum(vals)/1000),
            "latency_mean_ms": total_ms, "latency_p50_ms": float(np.percentile(vals,50)), "latency_p95_ms": float(np.percentile(vals,95)),
            "data_wait_ms": wait_ms, "h2d_ms": m(h2ds), "forward_ms": m(fwds), "backward_ms": m(bwds),
            "total_step_ms": total_ms, "data_wait_ratio": wait_ms/total_ms, "rss_mb": rss_mb(), "cache_bytes": getattr(loader.dataset.base, "cache_bytes", 0)}


def paths_from_manifest(manifest: Path, root: Path) -> list[Path]:
    obj = json.loads(manifest.read_text(encoding="utf-8")); paths = [root / r["file"] for r in obj["train"]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError(f"missing train files beneath --data-root: {missing[:3]}")
    return paths


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--warmup", type=int, default=100); p.add_argument("--measured", type=int, default=1000); p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--eq-windows", type=int, default=100); p.add_argument("--device", default="cuda"); p.add_argument("--profiles", default="A,B")
    p.add_argument("--candidates", default="all", help="all or comma-separated candidate names")
    args = p.parse_args(); out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    try:
        torch.manual_seed(args.seed); np.random.seed(args.seed); paths = paths_from_manifest(args.manifest, args.data_root)
        base = make_dataset("h5", paths); needed = (args.warmup + args.measured) * args.batch_size
        order = [i % len(base) for i in range(needed)]
        atom_json(out / "benchmark_indices.json", {"seed": args.seed, "batch_size": args.batch_size, "indices": order, "source_windows": len(base), "train_trajectories": len(paths)})
        atom_json(out / "provenance.json", {"manifest_sha256": sha256(args.manifest),
                  "benchmark_script_sha256": sha256(Path(__file__)),
                  "point_v0_script_sha256": sha256(Path(point.__file__)),
                  "data_loader_script_sha256": sha256(Path(__import__('realpde_p0_data').__file__)),
                  "seed": args.seed, "batch_size": args.batch_size, "warmup": args.warmup,
                  "measured": args.measured, "repeats": args.repeats, "device": args.device,
                  "train_only": True, "official_scorer_used": False})
        selected = candidates() if args.candidates == "all" else [c for c in candidates() if c.name in set(args.candidates.split(","))]
        if not selected: raise ValueError("no candidates selected")
        status(out, "RUNNING", "equivalence", candidates=[c.name for c in selected])
        eq_rows = []; result_rows = []
        # B0 is always measured first and is the sole DATA_EQUIVALENCE reference.
        ref_c = selected[0] if selected[0].name == "B0_CURRENT" else next(c for c in candidates() if c.name == "B0_CURRENT")
        ref_ds = make_dataset(ref_c.dataset_kind, paths); ref_seq = order[:args.eq_windows]
        for c in selected:
            ds = make_dataset(c.dataset_kind, paths); eq = eq_check(ref_ds, ds, ref_seq, args.eq_windows)
            eq["candidate"] = c.name; eq_rows.append(eq); atom_json(out / "equivalence.json", eq_rows)
            if not eq["passed"]: raise RuntimeError(f"DATA_EQUIVALENCE failed for {c.name}: {eq}")
            loader = make_loader(ds, order, c, args.batch_size); status(out, "RUNNING", "benchmark", current_candidate=c.name)
            if "A" in args.profiles: result_rows.append(timed_data(loader, args.warmup, args.measured, args.repeats, out, c))
            if "B" in args.profiles and torch.cuda.is_available(): result_rows.append(timed_train(loader, c, args.warmup, args.measured, args.repeats, torch.device(args.device)))
            atom_json(out / "results.json", result_rows); write_csv(out / "metrics.csv", result_rows)
        atom_json(out / "equivalence.json", eq_rows); atom_json(out / "results.json", result_rows)
        write_csv(out / "metrics.csv", result_rows)
        best = max(result_rows, key=lambda r: r.get("windows_per_s", 0))
        report = ["# Point data pipeline benchmark", "", f"- Seed: `{args.seed}`; batch: `{args.batch_size}`; warmup/measured/repeats: `{args.warmup}/{args.measured}/{args.repeats}`", f"- Train trajectories only: `{len(paths)}`; source windows: `{len(base)}`", f"- Manifest SHA-256: `{sha256(args.manifest)}`; benchmark script SHA-256: `{sha256(Path(__file__))}`", "- DATA_EQUIVALENCE: PASS for every measured candidate (100 fixed windows, exact float equality).", "", "| profile | candidate | workers | persistent | windows/s | step ms | data wait ms | data wait ratio | RSS MB |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
        byname = {c.name:c for c in selected}
        for r in result_rows:
            c = byname[r["candidate"]]; report.append(f"| {r['profile']} | {c.name} | {c.workers} | {c.persistent} | {r['windows_per_s']:.2f} | {r.get('total_step_ms', r['latency_mean_ms']):.2f} | {r['data_wait_ms']:.2f} | {r['data_wait_ratio']:.3f} | {r['rss_mb']:.1f} |")
        report += ["", f"Fastest measured row: `{best['profile']}/{best['candidate']}` at `{best['windows_per_s']:.2f}` windows/s.", "This benchmark does not change model semantics, accuracy, or Point-V0 decision."]
        (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        (out / "README_FOR_CHATGPT.md").write_text("Read `report.md`, `metrics.csv`, `equivalence.json`, `benchmark_indices.json`, `provenance.json`, and `status.json`. This is a train-only pipeline benchmark; no dev/final targets or scorer were accessed. B0 is the current H5WindowDataset baseline; candidates are B1 worker scan, B2 worker-local H5 handles, and B3 trajectory RAM cache.\n", encoding="utf-8")
        status(out, "DONE", "complete", rows=len(result_rows), data_equivalence="PASS")
    except Exception:
        status(out, "FAILED", "exception", traceback=traceback.format_exc())
        raise


if __name__ == "__main__": main()
