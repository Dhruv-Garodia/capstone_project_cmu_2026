#!/usr/bin/env python3
"""Is the synthetic stack realistic THROUGH the slices? Three tests.

The patch classifier in scorecard.py looks at one frame at a time, so a stack
whose frames each look right but change wrongly between slices (v5) passes
it. These tests look only at how things change:

1. Temporal fingerprint classifier. 9-slice clips of 64 px patches,
   drift-registered. Features are dynamics, mostly as ratios so texture
   matters little: band-pass correlation at lags 1/2/4/8 for fine, mid and
   coarse detail; the shape of the slice-to-slice change (normalised sd,
   skew, kurtosis, share of sharp darkenings and brightenings); temporal
   flicker (high- vs low-frequency power of each pixel's series through the
   clip); local motion (block-matched displacement between slices: mean
   residual after drift, spread in y and x); and the brightness trend in the
   slices before a sharp darkening (real: a gradual rise, then the drop).
2. Reslice classifier. x-z cuts through the stack (64 px wide, 32 slices)
   scored with the single-frame feature set: flicker shows as horizontal
   stripes, facets switching on as steps, persistence as streak length.
3. Arrow of time. A classifier trained on REAL clips to tell forward from
   reversed learns the physics of cutting (gradual brightening, sudden
   opening, which way buried material slides). Applied to synthetic clips:
   its forward/reversed AUC there should match its AUC on held-out real.

Every classifier AUC is reported with a real-vs-real control (interleaved
blocks of the real stack, so both halves span it); "realistic in time" means
at or under the control.

    python temporal_scorecard.py --style comp --synth stack.tif
"""
import argparse, json
import numpy as np, tifffile
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
from discriminate import features, logistic_auc, REAL

WIN = {"comp": (150, 450), "uncomp": (180, 480), "pristine": (400, 750)}
DRIFT = {"comp": 1.1, "uncomp": 1.1, "pristine": 0.1}
NAMES = ([f"corr_{b}_l{l}" for b in ("fine", "mid", "coarse") for l in (1, 2, 4, 8)]
         + ["dsd", "dskew", "dkurt", "sharp_dark", "sharp_bright", "flick_hi", "flick_lo",
            "flow_resid", "flow_sdy", "flow_sdx", "pre_dark_rise", "dark_bright_ratio"])


def frames(path, zs):
    with tifffile.TiffFile(path) as t:
        n = len(t.pages)
        return {z: t.pages[z].asarray().astype(np.float32) for z in zs if 0 <= z < n}, n


def clip(F, z0, y, x, dr, L=9, P=64):
    """Drift-registered L-slice clip of a P x P patch starting at slice z0."""
    out = []
    for k in range(L):
        f = F[z0 + k]; yy = y + dr * k; iy = int(np.floor(yy)); fr = yy - iy
        a = f[iy:iy + P, x:x + P]; b = f[iy + 1:iy + 1 + P, x:x + P]
        if a.shape != (P, P) or b.shape != (P, P):
            return None
        out.append((1 - fr) * a + fr * b)
    return np.stack(out)


def band(a, s1, s2):
    return (ndi.gaussian_filter(a, (0, s1, s1)) if s1 else a) - ndi.gaussian_filter(a, (0, s2, s2))


def cc(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def tfeat(c):
    f = []
    for s1, s2 in ((0.0, 1.2), (1.2, 4.0), (4.0, 12.0)):
        B = band(c, s1, s2)
        f += [np.mean([cc(B[i], B[i + l]) for i in range(len(c) - l)]) for l in (1, 2, 4, 8)]
    sm = ndi.gaussian_filter(c, (0, 1.2, 1.2))
    d = np.diff(sm, axis=0).ravel(); sd = d.std() + 1e-6
    f += [sd / (sm.std() + 1e-6), float(skew(d)), float(np.clip(kurtosis(d), -5, 50)),
          float((d < -3 * sd).mean()), float((d > 3 * sd).mean())]
    ts = sm - sm.mean(0, keepdims=True); P = np.abs(np.fft.rfft(ts, axis=0)) ** 2
    tot = P[1:].sum() + 1e-9; f += [float(P[-2:].sum() / tot), float(P[1:3].sum() / tot)]
    # block-matched local motion between slice i and i+2 (16 px blocks)
    dys, dxs = [], []
    for i in (0, 3, 6):
        if i + 2 >= len(c): break
        A, Bf = sm[i], sm[i + 2]
        for by in range(8, 48, 16):
            for bx in range(8, 48, 16):
                a = A[by:by + 16, bx:bx + 16]; best = (-2, 0, 0)
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        b = Bf[by + dy:by + dy + 16, bx + dx:bx + dx + 16]
                        if b.shape == a.shape:
                            v = cc(a, b)
                            if v > best[0]: best = (v, dy, dx)
                dys.append(best[1]); dxs.append(best[2])
    f += [float(np.mean(dys)), float(np.std(dys)), float(np.std(dxs))]
    # what happens before a sharp darkening: mean rise over the 3 slices before
    dd = np.diff(sm, axis=0); ev = np.argwhere(dd[3:] < -3 * sd)
    rise = [float(sm[t + 3, y, x] - sm[t, y, x]) / sd for t, y, x in ev[::7][:200]] if len(ev) else [0.0]
    f += [float(np.mean(rise)), float((d < -3 * sd).mean() / ((d > 3 * sd).mean() + 1e-4))]
    return np.array(f, np.float32)


def sample_clips(path, style, n_clips, rng, z_lo=0, z_hi=None):
    y0, y1 = WIN[style]; dr = DRIFT[style]
    with tifffile.TiffFile(path) as t: nz = len(t.pages)
    z_hi = min(nz - 10, z_hi or nz - 10)
    starts = np.linspace(z_lo, z_hi, max(2, n_clips // 8)).astype(int)
    F, _ = frames(path, sorted({z + k for z in starts for k in range(9)}))
    W = next(iter(F.values())).shape[1]; out = []
    for z in starts:
        for _ in range(8):
            y = int(rng.integers(y0, y1 - 64 - int(dr * 9) - 2)); x = int(rng.integers(100, W - 164))
            c = clip(F, z, y, x, dr)
            if c is not None: out.append((z, c))
    return out


def reslice_feats(path, style, rng, n=120):
    y0, y1 = WIN[style]; dr = DRIFT[style]
    with tifffile.TiffFile(path) as t: nz = len(t.pages)
    out = []
    zs = np.linspace(0, nz - 33, max(1, n // 12)).astype(int)
    F, _ = frames(path, sorted({z + k for z in zs for k in range(32)}))
    W = next(iter(F.values())).shape[1]
    for z in zs:
        for _ in range(12):
            y = int(rng.integers(y0, y1 - int(dr * 32) - 2)); x = int(rng.integers(100, W - 164))
            r = np.stack([F[z + k][int(round(y + dr * k)), x:x + 64] for k in range(32)])
            out.append(features(np.repeat(r, 2, axis=0)))     # 64 x 64, slices doubled
    return out


def auc_rep(A, B, k=6, top=None, names=None):
    X = np.array(list(A) + list(B)); y = np.array([0] * len(A) + [1] * len(B))
    res = [logistic_auc(X, y, np.random.default_rng(s)) for s in range(k)]
    a = float(np.mean([r[0] for r in res]))
    if top is not None and names:
        w = np.mean([r[1] for r in res], 0)
        top.extend(f"{names[i]}{'+' if w[i] > 0 else '-'}" for i in np.argsort(-np.abs(w))[:5])
    return round(max(a, 1 - a), 3)


def arrow_of_time(real_clips, syn_clips, rng):
    """Train forward-vs-reversed on half the real clips; AUC on held-out real and on synthetic."""
    def fx(cl): return [tfeat(c) for c in cl]
    idx = rng.permutation(len(real_clips)); h = len(idx) // 2
    tr = [real_clips[i] for i in idx[:h]]; te = [real_clips[i] for i in idx[h:]]
    Xtr = np.array(fx(tr) + fx([c[::-1] for c in tr])); ytr = np.array([1] * len(tr) + [0] * len(tr))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xn = (Xtr - mu) / sd
    w = np.zeros(Xn.shape[1]); b = 0.0
    for _ in range(1500):
        p = 1 / (1 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        w -= 0.1 * (Xn.T @ (p - ytr) / len(Xn) + 1e-3 * w); b -= 0.1 * float((p - ytr).mean())
    from discriminate import auc
    def score(cl):
        X = (np.array(fx(cl) + fx([c[::-1] for c in cl])) - mu) / sd
        return round(auc(np.array([1] * len(cl) + [0] * len(cl)), X @ w + b), 3)
    return score(te), score(syn_clips)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--style", required=True); ap.add_argument("--synth", required=True)
    ap.add_argument("--n", type=int, default=240); ap.add_argument("--json", default=None)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    rp = str(REAL[a.style][0])
    R = sample_clips(rp, a.style, a.n, rng); S = sample_clips(a.synth, a.style, a.n, rng)
    FR = [tfeat(c) for _, c in R]; FS = [tfeat(c) for _, c in S]
    blk = [(z // 24) % 2 == 0 for z, _ in R]
    out = {}; tells = []
    out["temporal_auc"] = auc_rep(FR, FS, top=tells, names=NAMES); out["temporal_tells"] = tells
    out["temporal_control"] = auc_rep([f for f, b in zip(FR, blk) if b], [f for f, b in zip(FR, blk) if not b])
    RR = reslice_feats(rp, a.style, rng); RS = reslice_feats(a.synth, a.style, rng)
    out["reslice_auc"] = auc_rep(RR, RS)
    out["reslice_control"] = auc_rep(RR[0::2], RR[1::2])
    out["arrow_real_heldout"], out["arrow_synth"] = arrow_of_time([c for _, c in R], [c for _, c in S], rng)
    mr, ms = np.mean(FR, 0), np.mean(FS, 0)
    out["means"] = {n: [round(float(x), 3), round(float(y), 3)] for n, x, y in zip(NAMES, mr, ms)}
    print(json.dumps({k: v for k, v in out.items() if k != "means"}))
    if a.json: open(a.json, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
