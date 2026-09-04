"""Single matched Direct@1500 -> Direct@3000 MF closeout continuation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import realpde_mf01 as mf01
import realpde_loss_official_v9 as core


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    core.set_seed(args.seed)
    args.out_dir.mkdir(parents=True)
    _, train_paths = core.read_manifest(args.manifest, "train")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder, config = mf01.build_features(train_paths, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = mf01.cno_direct(args.kit_root, len(builder.feature_names), device)
    state = checkpoint.get("model_state_dict", checkpoint)
    input_key = "lift.inter_CNOBlock.convolution.weight"
    if input_key in state and state[input_key].shape[1] == len(builder.feature_names):
        model.load_state_dict(state, strict=True)
    else:
        mf01.adapt_input_weight(model, checkpoint, len(builder.feature_names))
    metadata = {
        "experiment_id": args.experiment_id,
        "mode": "direct_continuation",
        "parent_experiment_id": "T1-ID-MF01-CONTROL-S20260904",
        "start_iteration": args.start_iteration,
        "updates_added": args.updates_added,
        "final_iteration": args.start_iteration + args.updates_added,
        "milestones": args.milestones,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "manifest_sha256": core.sha256(args.manifest),
        "checkpoint_sha256": core.sha256(args.checkpoint),
        "scorer_sha256": core.sha256(args.kit_root / "scoring.py"),
        "feature_names": list(builder.feature_names),
        "feature_config": vars(config),
        "loss_weights": mf01.N2_WEIGHTS,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "device": str(device),
        "execution_commit": args.execution_commit,
        "dependency_sha256": {
            "realpde_mf01.py": core.sha256(Path(mf01.__file__)),
            "realpde_loss_official_v9.py": core.sha256(Path(core.__file__)),
            "realpde_p0_data.py": core.sha256(Path(__import__("realpde_p0_data").__file__)),
            "realpde_p0_features.py": core.sha256(Path(__import__("realpde_p0_features").__file__)),
        },
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    train_ds, train_loader = core.loader(train_paths, args, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    baseline = mf01.evaluate(model, builder, dev_paths, args, device, args.kit_root, args.out_dir / "eval_01500")
    history = [{"iteration": args.start_iteration, **baseline["raw_errors"], "official_v9_subscores": baseline["official_v9_subscores"]}]
    iterator = iter(train_loader)
    started = time.monotonic()
    for relative_step in range(1, args.updates_added + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.train()
        pred = mf01.forward(model, builder, x)
        parts = core.loss_parts(pred, y)
        loss = sum(mf01.N2_WEIGHTS[name] * parts[name] for name in mf01.N2_WEIGHTS)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if relative_step in args.milestones:
            absolute_step = args.start_iteration + relative_step
            evaluation = mf01.evaluate(model, builder, dev_paths, args, device, args.kit_root, args.out_dir / f"eval_{absolute_step:05d}")
            row = {"iteration": absolute_step, **evaluation["raw_errors"], "official_v9_subscores": evaluation["official_v9_subscores"], "elapsed_seconds": time.monotonic() - started}
            history.append(row)
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": absolute_step, "metadata": metadata, "feature_set": "P0-A", "loss_weights": mf01.N2_WEIGHTS}, args.out_dir / f"model_update_{absolute_step:05d}.pth")
            print(json.dumps(row, sort_keys=True), flush=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": args.start_iteration + args.updates_added, "metadata": metadata, "feature_set": "P0-A", "loss_weights": mf01.N2_WEIGHTS}, args.out_dir / "model_last.pth")
    (args.out_dir / "summary.json").write_text(json.dumps({"metadata": metadata | {"train_windows": len(train_ds), "elapsed_seconds": time.monotonic() - started}, "history": history}, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-iteration", type=int, default=1500)
    parser.add_argument("--updates-added", type=int, default=1500)
    parser.add_argument("--milestones", type=int, nargs="+", default=[500, 1000, 1500])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--execution-commit", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
