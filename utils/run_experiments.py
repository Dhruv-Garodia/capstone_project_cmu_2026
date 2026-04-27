#!/usr/bin/env python3

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "utils" / "train.py"


def parse_csv_list(text: str, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run a small grid of PFIB-SEM training experiments.")
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--out", default="experiments")
    parser.add_argument("--losses", default="ce,cedice,focaldice,tversky,focaltversky",
                        help="Comma-separated list of loss names")
    parser.add_argument("--lrs", default="1e-3,3e-4", help="Comma-separated learning rates")
    parser.add_argument("--seeds", default="2025,2026", help="Comma-separated random seeds")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--extra_args", default="",
                        help="Quoted extra args forwarded to train.py, e.g. '--gamma_p 0.3 --flip_v_p 0.2'")
    args = parser.parse_args()

    losses = parse_csv_list(args.losses, str)
    lrs = parse_csv_list(args.lrs, float)
    seeds = parse_csv_list(args.seeds, int)
    extra_args = args.extra_args.split() if args.extra_args else []

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    commands = []
    for loss, lr, seed in itertools.product(losses, lrs, seeds):
        exp_name = f"loss-{loss}_lr-{lr:g}_seed-{seed}"
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--img_dir", args.img_dir,
            "--mask_dir", args.mask_dir,
            "--out", str(out_root),
            "--experiment_name", exp_name,
            "--loss", loss,
            "--lr", str(lr),
            "--seed", str(seed),
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
        ]
        if args.device:
            cmd += ["--device", args.device]
        cmd += extra_args
        commands.append(cmd)

    print(f"[Info] Launching {len(commands)} experiments")
    for idx, cmd in enumerate(commands, start=1):
        print(f"\n[Run {idx}/{len(commands)}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
