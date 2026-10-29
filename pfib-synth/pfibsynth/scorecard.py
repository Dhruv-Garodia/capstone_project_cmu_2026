#!/usr/bin/env python3
"""One-command realism scorecard for a synthetic stack against its real one.

Each number is reported next to a real-vs-real control (first half of the
real stack against the second half): "can't tell" means the synthetic score
sits at or under the control, not at 0.5.

  patch AUC   64 px patches inside the layer (texture, grain, edges)
  frame AUC   whole frames shrunk to 256 px wide (layout: cap, voids,
              boundaries, large objects) -- patches never see these
  edge ratio  pore top-edge / bottom-edge contrast step (the oblique-view
              cue: an overhang above a pore is sharper than the slope below)
  block sd    sd of pore fraction over 64 px blocks (discrete heterogeneity)
  zcorr       band-pass correlation 1 / 4 / 8 slices apart, drift-registered
  skew        skew of the registered slice-to-slice change (one-sidedness)

    python scorecard.py --style comp --synth path/to/stack.tif
"""
import argparse, json
import numpy as np, tifffile
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import skew as _skew
from discriminate import features, logistic_auc, REAL

WIN = {"comp": (150, 450), "uncomp": (180, 480), "pristine": (400, 750)}
DRIFT = {"comp": 1.1, "uncomp": 1.1, "pristine": 0.1}


def load(path, zs):
    with tifffile.TiffFile(path) as t:
        n = len(t.pages)
        return [t.pages[z].asarray() for z in zs if z < n], n


def nframes(path):
    with tifffile.TiffFile(path) as t:
        return len(t.pages)


def patch_feats(frames, y0, y1, rng, n=12, P=64):
    out = []
    for f in frames:
        for _ in range(n):
            y = rng.integers(y0, y1 - P); x = rng.integers(100, f.shape[1] - 100 - P)
            out.append(features(f[y:y + P, x:x + P]))
    return out


def frame_feats(frames):
    out = []
    for f in frames:
        im = Image.fromarray(f); im = im.resize((256, max(8, int(f.shape[0] * 256 / f.shape[1]))), Image.BILINEAR)
        a = np.asarray(im, np.float32)
        out.append(features(a))
    return out


NAMES = ([f"hist{i}" for i in range(16)] + [f"grad_p{q}" for q in (25, 50, 75, 90, 99)] + ["grain_sd", "grain_p90"]
         + [f"spec{i}" for i in range(8)] + ["mean", "sd"])


def auc_rep(A, B, k=6, top=None):
    X = np.array(A + B); y = np.array([0] * len(A) + [1] * len(B))
    res = [logistic_auc(X, y, np.random.default_rng(s)) for s in range(k)]
    a = float(np.mean([r[0] for r in res]))
    if top is not None:
        w = np.mean([r[1] for r in res], 0)
        top.extend(f"{NAMES[i]}{'+' if w[i] > 0 else '-'}" for i in np.argsort(-np.abs(w))[:5])
    return round(max(a, 1 - a), 3)


def edge_ratio(frames, y0, y1):
    r, w = [], []
    for img in frames:
        img = img[y0:y1, 300:-300].astype(np.float32)
        s = ndi.gaussian_filter(img, 2.0); b = ndi.binary_opening(s < np.percentile(s, 40))
        a0, a1 = b[7:-9], b[8:-8]
        st = {}
        for k, e in (("top", (~a0) & a1), ("bot", a0 & (~a1))):
            ys, xs = np.where(e); ys = ys + 8; sel = slice(None, None, 3)
            pr = np.stack([img[ys[sel] - 8 + j, xs[sel]] for j in range(16)], 1).mean(0)
            g = np.abs(np.diff(pr)); st[k] = (g.max(), (g > 0.5 * g.max()).sum())
        r.append(st["top"][0] / st["bot"][0]); w.append((st["top"][1] + st["bot"][1]) / 2)
    return round(float(np.mean(r)), 2), round(float(np.mean(w)), 1)


def block_sd(frames, y0, y1, B=64):
    v = []
    for img in frames:
        s = ndi.gaussian_filter(img[y0:y1].astype(np.float32), 2.0); p = s < np.percentile(s, 40)
        for y in range(0, p.shape[0] - B + 1, B):
            for x in range(100, p.shape[1] - 100 - B, B):
                v.append(p[y:y + B, x:x + B].mean())
    return round(float(np.std(v)), 3)


def zstats(path, z0, n, y0, dr):
    fs, _ = load(path, range(z0, z0 + n))
    v = [f.astype(np.float32) for f in fs]
    bp = lambda x: ndi.gaussian_filter(x, 1.0) - ndi.gaussian_filter(x, 6)
    cc = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    zc = []
    for g in (1, 4, 8):
        o = []
        for z in range(0, len(v) - g, 4):
            A = bp(v[z][y0:y0 + 300, 500:1100]); best = -1
            for dy in range(int(np.floor(dr * g)) - 2, int(np.ceil(dr * g)) + 3):
                B = bp(v[z + g][y0 + dy:y0 + dy + 300, 500:1100]); best = max(best, cc(A[10:-10], B[10:-10]))
            o.append(best)
        zc.append(round(float(np.mean(o)), 2))
    d = []
    for i in range(len(v) - 1):
        A = ndi.gaussian_filter(v[i], 1.5); B = ndi.gaussian_filter(v[i + 1], 1.5)
        y = y0 + int(round(dr * i)); y2 = y0 + int(round(dr * (i + 1)))
        d.append((B[y2:y2 + 150, 500:1100] - A[y:y + 150, 500:1100]).ravel())
    return zc, round(float(_skew(np.concatenate(d))), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True); ap.add_argument("--synth", required=True)
    ap.add_argument("--real-frames", type=int, default=60); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rp = str(REAL[a.style][0]); y0, y1 = WIN[a.style]; rng = np.random.default_rng(0)
    nr, ns = nframes(rp), nframes(a.synth)
    rz = np.linspace(5, nr - 6, a.real_frames).astype(int); sz = np.linspace(2, ns - 3, min(ns - 4, 40)).astype(int)
    R, _ = load(rp, rz); Sy, _ = load(a.synth, sz)
    h = len(R) // 2
    out = {}
    tp = []
    out["patch_auc"] = auc_rep(patch_feats(R, y0, y1, rng), patch_feats(Sy, y0, y1, rng), top=tp)
    out["patch_tells"] = tp
    PR = [patch_feats([f], y0, y1, rng) for f in R]; bb = (np.arange(len(R)) // 6) % 2 == 0
    out["patch_auc_control"] = auc_rep(sum([q for q, b in zip(PR, bb) if b], []), sum([q for q, b in zip(PR, bb) if not b], []))
    tf = []
    out["frame_auc"] = auc_rep(frame_feats(R), frame_feats(Sy), top=tf)
    out["frame_tells"] = tf
    blk = (np.arange(len(R)) // 6) % 2 == 0        # interleaved 6-frame blocks: both span the stack
    FR = frame_feats(R)
    out["frame_auc_control"] = auc_rep([f for f, b in zip(FR, blk) if b], [f for f, b in zip(FR, blk) if not b])
    out["edge_ratio_real"], out["edge_width_real"] = edge_ratio(R[::4], y0, y1)
    out["edge_ratio"], out["edge_width"] = edge_ratio(Sy[::3], y0, y1)
    out["block_sd_real"], out["block_sd"] = block_sd(R[::4], y0, y1), block_sd(Sy[::3], y0, y1)
    zr = (100 if a.style != "pristine" else 200)
    out["zcorr_real"], out["skew_real"] = zstats(rp, zr, 40, y0, DRIFT[a.style])
    out["zcorr"], out["skew"] = zstats(a.synth, 0, min(ns, 40), y0, DRIFT[a.style])
    print(json.dumps(out))
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
