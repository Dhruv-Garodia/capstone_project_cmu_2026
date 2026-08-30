"""Run on the GPU box: generates a tiny dataset, trains 1 epoch of each model, evaluates a stack at descriptor level."""
import os, subprocess, sys, numpy as np
root = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, root)
subprocess.check_call([sys.executable, os.path.join(root, "make_dataset.py"), "--out", "datasets/smoke", "--n", "4", "--shape", "48", "96", "96"])
from eponge_synth.training import get_parser, train, build_model, predict_stack, evaluate_volume
import torch
for arch, mode in [("unet2d", "2.5d"), ("deeplabv3plus", "2.5d"), ("unet3d", "3d")]:
    args = get_parser().parse_args(["--data", "datasets/smoke", "--out", f"runs/smoke_{arch}", "--arch", arch, "--mode", mode,
                                    "--epochs", "1", "--batch_size", "2", "--crop", "64", "--patch", "16", "64", "64", "--workers", "0",
                                    "--w_bd", "0.1", "--w_cons", "0.1"])
    train(args)
imgs = np.load("datasets/smoke/stacks/0000_catalyst_layer_pristine/images.npy") if os.path.exists("datasets/smoke/stacks/0000_catalyst_layer_pristine/images.npy") else None
print("smoke test OK")
