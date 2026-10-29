"""Pack a generated package into one .pfibv file for the viewer (viewer/index.html).

    python tools/export_viewer.py PACKAGE_DIR [OUT.pfibv] [--max-xy 512] [--bin 1]

The frames drift ~1.1 px per slice as the cut advances; the export removes that drift, so
the volume is the material in its own coordinates and the 3-D view is one object. It
keeps the catalyst layer and a margin of what is around it.

Format: gzip of  [uint32 header length][JSON header][image bytes][label bytes]
image: uint8 (nz, ny, nx); label: uint8, 0 pore, 1 solid, 2 outside the layer.
"""
import argparse, gzip, json, struct
from pathlib import Path
import numpy as np, tifffile
from PIL import Image


def export(pkg, out=None, max_xy=512, bin_=1, frames=None):
    pkg = Path(pkg)
    meta = json.loads((pkg / "stack.json").read_text()); tj = json.loads((pkg / "truth/truth.json").read_text())
    stack = tifffile.imread(pkg / meta["stack_file"])
    n = min(frames or len(stack), len(stack))
    drift = float(tj["optics"]["drift"])
    T = [np.array(Image.open(pkg / f"truth/masks/slice_{z:04d}.png")) for z in range(n)]
    H, W = stack.shape[1:]
    sh = int(np.ceil(drift * (n - 1)))
    lay_rows = np.where((T[0] != 128).mean(1) > 0.5)[0]
    top = max(0, int(lay_rows[0]) - 24) if len(lay_rows) else 0
    bot = min(H - sh, int(lay_rows[-1]) + 24) if len(lay_rows) else H - sh
    ny = min(bot - top, max_xy * bin_); nx = min(W, max_xy * bin_)
    x0 = (W - nx) // 2
    img = np.empty((n, ny // bin_, nx // bin_), np.uint8); lab = np.empty_like(img)
    for z in range(n):
        y = top + int(round(drift * z))
        a = stack[z, y:y + ny, x0:x0 + nx].astype(np.float32); t = T[z][y:y + ny, x0:x0 + nx]
        if bin_ > 1:
            a = a[:ny // bin_ * bin_, :nx // bin_ * bin_].reshape(ny // bin_, bin_, nx // bin_, bin_).mean((1, 3))
            t = t[::bin_, ::bin_][:ny // bin_, :nx // bin_]
        img[z] = np.clip(a, 0, 255).astype(np.uint8)
        lab[z] = np.where(t == 0, 0, np.where(t == 255, 1, 2))
    hdr = dict(name=meta["name"], nz=int(n), ny=int(img.shape[1]), nx=int(img.shape[2]), bin=bin_,
               drift=drift, style=tj.get("style_calibrated_to"), family=tj.get("material_family"),
               true_porosity=tj.get("true_porosity"), start_porosity=tj.get("start_mask_porosity"),
               zlen=float(json.loads((Path(__file__).resolve().parent.parent / "pfibsynth/slicing.json").read_text())
                          .get(tj.get("style_calibrated_to", "comp"), {}).get("zlen", 1.0)),
               labels={"0": "pore", "1": "solid", "2": "outside the layer"})
    h = json.dumps(hdr).encode()
    out = Path(out) if out else pkg / f"{meta['name']}.pfibv"
    with gzip.open(out, "wb", compresslevel=6) as f:
        f.write(struct.pack("<I", len(h))); f.write(h); f.write(img.tobytes()); f.write(lab.tobytes())
    print(f"wrote {out}  {img.shape}  {out.stat().st_size / 1e6:.1f} MB", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg"); ap.add_argument("out", nargs="?")
    ap.add_argument("--max-xy", type=int, default=512); ap.add_argument("--bin", type=int, default=1)
    ap.add_argument("--frames", type=int, default=None)
    a = ap.parse_args()
    export(a.pkg, a.out, a.max_xy, a.bin, a.frames)
