"""Command line interface.

    eponge analyze  IMAGE_OR_STACK  --out results/ [--pixel-nm 6]
    eponge train    --stack ... --labels ... --out runs/x [--k 2 --register]
    eponge fix-annotations --stack ... --masks data/manual_annotation --out data/annotations_v2
    eponge audit-annotations --stack ... --masks data/manual_annotation
    eponge export-onnx CKPT OUT.onnx
    eponge export-web  CKPT OUT_DIR       (weights for the browser app)
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__); return
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        from .train import main as train_main
        return train_main(rest)
    ap = argparse.ArgumentParser(prog=f"eponge {cmd}")
    if cmd == "analyze":
        ap.add_argument("path"); ap.add_argument("--out", default="eponge_results")
        ap.add_argument("--pixel-nm", type=float, default=6.0); ap.add_argument("--ckpt", default=None)
        ap.add_argument("--no-register", action="store_true"); ap.add_argument("--device", default=None)
        a = ap.parse_args(rest)
        from .models import load_checkpoint
        from .pipeline import analyze_file
        kw = {"pixel_nm": a.pixel_nm, "device": a.device}
        if a.ckpt:
            model, ck = load_checkpoint(a.ckpt, a.device or "cpu")
            kw["model"] = model
            if ck.get("k", 0):
                kw["k"] = ck["k"]
        res = analyze_file(a.path, **kw)
        out = res.save(a.out)
        m = res.metrics
        print(json.dumps({"porosity": m["porosity"], "d50_nm": m["pore_size_nm"]["d50"],
                          "cl_thickness_nm": m["depth_profile"]["thickness_nm_mean"],
                          "tortuosity": {k: v["tau"] for k, v in m.get("tortuosity", {}).items() if isinstance(v, dict)},
                          "uncertainty": {k: v for k, v in res.uncertainty.items() if k != "profile_band"},
                          "output": str(out)}, indent=2, default=float))
    elif cmd in {"fix-annotations", "audit-annotations"}:
        ap.add_argument("--stack", required=True); ap.add_argument("--masks", required=True)
        ap.add_argument("--out", default="annotations_v2"); ap.add_argument("--frames", type=int, default=118)
        a = ap.parse_args(rest)
        from .annotation import correct_folder, implied_threshold
        from .io import load_masks, load_stack
        stack = load_stack(a.stack, range(a.frames))
        if cmd == "audit-annotations":
            for z, m in sorted(load_masks(a.masks).items()):
                t, agree = implied_threshold(stack[z], m)
                print(f"frame {z:4d}: labels == (raw < {t}) for {agree*100:.2f}% of ROI pixels; "
                      f"porosity {(m[m != 128] == 0).mean():.3f}")
        else:
            s = correct_folder(stack, a.masks, a.out)
            print(f"wrote {s['n_masks']} corrected masks to {a.out}; missing frames {s['missing_frames']}")
    elif cmd == "export-onnx":
        ap.add_argument("ckpt"); ap.add_argument("out")
        a = ap.parse_args(rest)
        from .export import export_onnx
        print(json.dumps(export_onnx(a.ckpt, a.out), indent=2))
    elif cmd == "export-web":
        ap.add_argument("ckpt"); ap.add_argument("out")
        a = ap.parse_args(rest)
        from .export import export_web_weights
        print(json.dumps(export_web_weights(a.ckpt, a.out), indent=2))
    else:
        print(f"unknown command {cmd}\n{__doc__}"); return 2


if __name__ == "__main__":
    main()
