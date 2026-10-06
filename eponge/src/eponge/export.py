"""Export a trained checkpoint to ONNX for the browser app (onnxruntime-web) or other runtimes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .models import load_checkpoint


def _fold(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d | None) -> tuple[np.ndarray, np.ndarray]:
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    if bn is not None:
        s = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
        w = w * s[:, None, None, None]
        b = (b - bn.running_mean.double()) * s + bn.bias.detach().double()
    # PyTorch [out, in, kh, kw] -> TF.js [kh, kw, in, out]
    return w.permute(2, 3, 1, 0).float().numpy(), b.float().numpy()


def export_web_weights(ckpt_path: str | Path, out_dir: str | Path) -> dict:
    """Export a checkpoint for the browser app: BN folded into convs, float16 weights + JSON manifest.

    The browser re-implements the (tiny) U-Net graph with TensorFlow.js ops, so it needs no
    WebAssembly and runs on WebGL. Layout: blocks ``stem``, ``down0..``, ``up0..`` each with
    ``conv1``, ``conv2`` and optional ``skip`` (1x1), then ``head``.
    """
    model, ckpt = load_checkpoint(ckpt_path, "cpu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tensors, layers = [], []

    def add(name, conv, bn):
        w, b = _fold(conv, bn)
        layers.append({"name": name, "wshape": list(w.shape), "bshape": list(b.shape)})
        tensors.extend([w.ravel(), b.ravel()])

    def block(prefix, blk):
        add(prefix + ".conv1", blk.conv1, blk.bn1)
        add(prefix + ".conv2", blk.conv2, blk.bn2)
        if isinstance(blk.skip, torch.nn.Conv2d):
            add(prefix + ".skip", blk.skip, None)

    block("stem", model.stem)
    for i, b in enumerate(model.down):
        block(f"down{i}", b)
    for i, b in enumerate(model.up):
        block(f"up{i}", b)
    add("head", model.head, None)
    flat = np.concatenate(tensors).astype(np.float16)
    (out_dir / "weights.bin").write_bytes(flat.tobytes())
    manifest = {"config": model.config, "k": ckpt.get("k", 0), "layers": layers, "dtype": "float16",
                "classes": ["solid", "pore", "outside_roi"], "normalization": "percentile 1-99 -> [0,1]",
                "train_pixel_nm": 6.0, "val_metrics": {k: v for k, v in ckpt.get("val", {}).items()
                                                       if isinstance(v, float)}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    # reference input/output for numerical checks of the JS implementation
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, model.config["in_channels"], 96, 160, generator=g)
    with torch.no_grad():
        y = torch.softmax(model(x), 1)
    np.save(out_dir / "_ref_input.npy", x.numpy()); np.save(out_dir / "_ref_probs.npy", y.numpy())
    return {"bytes": int(flat.nbytes), "n_layers": len(layers)}


def export_onnx(ckpt_path: str | Path, out_path: str | Path, opset: int = 17, check: bool = True) -> dict:
    model, ckpt = load_checkpoint(ckpt_path, "cpu")
    cin = model.config["in_channels"]
    dummy = torch.rand(1, cin, 256, 256)
    out_path = Path(out_path)
    torch.onnx.export(model, dummy, str(out_path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                    "logits": {0: "batch", 2: "height", 3: "width"}},
                      opset_version=opset, dynamo=False)
    meta = {"config": model.config, "k": ckpt.get("k", 0), "multiple": model.multiple,
            "classes": ["solid", "pore", "outside_roi"], "normalization": "percentile 1-99 -> [0,1]",
            "train_pixel_nm": 6.0, "size_bytes": out_path.stat().st_size}
    if check:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path))
        x = torch.rand(1, cin, 320, 480)
        ref = model(x).detach().numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        meta["max_abs_diff_vs_torch"] = float(np.abs(ref - got).max())
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta
