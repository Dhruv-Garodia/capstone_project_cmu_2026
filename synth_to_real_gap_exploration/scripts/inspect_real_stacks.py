#!/usr/bin/env python3
"""
inspect_real_stacks.py — deep inspection of the real pFIB-SEM stacks (pristine/uncomp/comp_full.tif),
producing a compact report (< 10 MB) that can be shared back through Drive or chat.

Implements the measurement playbook of docs/FINDINGS_AND_TRANSFER.md §5.2: every simulator artefact
is measured SEPARATELY from the real slices so the simulator can be set to measured values instead of
fitted ones.  Memory-safe: slices are read page-by-page (tifffile), never the whole 0.4-1.5 GB stack.

Run in Google Colab (recommended):
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install tifffile scikit-image scipy scikit-learn matplotlib
    !python inspect_real_stacks.py --tif "/content/drive/MyDrive/<path>/comp_full.tif" --out /content/drive/MyDrive/inspection
    (repeat for uncomp_full.tif and pristine_full.tif; the shared folder may need a shortcut in My Drive)
Or locally:  python inspect_real_stacks.py --tif comp_full.tif --out inspection --voxel_nm 6

Outputs per stack, under <out>/<stack_name>/:
    metadata.json        shape, dtype, TIFF/ImageJ tags, voxel size, byte size check
    per_slice_stats.csv  mean/std/p5/p50/p95 for every Nth slice (drift), plus low-pass field amplitude
    measurements.json    noise (Poisson scale, read noise), PSF sigma, charging field (amp, corr length,
                         slice-to-slice AR), curtaining (amp, sigma), shine-through anisotropy & decay,
                         structure through the paper-like operator (porosity, PSD mu/sigma, mean pore),
                         C2ST floor between disjoint z-blocks (if scikit-learn available)
    crops/*.png          native-resolution 512x512 crops at several depths and positions
    fig_*.png            overview figures (slice + histogram, drift, noise fit, spectra, side view, PSD)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

try:
    import tifffile
except ImportError:
    sys.exit("pip install tifffile")
from scipy import ndimage as ndi, stats

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


class Stack:
    """Page-wise access to a (possibly huge) TIFF stack."""

    def __init__(self, path):
        self.path = path
        self.tf = tifffile.TiffFile(path)
        s = self.tf.series[0]
        self.shape = tuple(int(x) for x in s.shape)
        self.dtype = str(s.dtype)
        self.n = self.shape[0] if len(self.shape) == 3 else 1
        self.ij = self.tf.imagej_metadata or {}
        p0 = self.tf.pages[0]
        self.tags = {t.name: (str(t.value)[:200]) for t in p0.tags.values() if t.name in
                     ("ImageWidth", "ImageLength", "BitsPerSample", "XResolution", "YResolution", "ResolutionUnit",
                      "ImageDescription", "Software", "DateTime", "Compression", "SamplesPerPixel")}

    def slice(self, i):
        a = tifffile.imread(self.path, key=int(i))
        return np.asarray(a)

    def to8(self, a):
        if a.dtype == np.uint8:
            return a
        lo, hi = np.percentile(a, [0.1, 99.9])
        return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def lowpass(a, sigma):
    return ndi.gaussian_filter(a.astype(np.float32), sigma)


def radial_spectrum(a):
    f = a.astype(np.float32) - a.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    h, w = P.shape
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    rad = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return rad


# --------------------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------------------
def measure_noise(img, win=7, n_bins=12):
    """Variance vs mean in locally flat windows -> Poisson scale (counts per grey level) and read noise.
    Flat windows = lowest-gradient quartile, so structure does not inflate the variance."""
    a = img.astype(np.float32)
    m = ndi.uniform_filter(a, win)
    v = ndi.uniform_filter(a * a, win) - m * m
    g = np.hypot(*np.gradient(ndi.gaussian_filter(a, 1.5)))
    flat = g < np.percentile(g, 25)
    mm, vv = m[flat], np.clip(v[flat], 0, None)
    edges = np.percentile(mm, np.linspace(2, 98, n_bins + 1))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mm >= lo) & (mm < hi)
        if sel.sum() > 200:
            xs.append(float(mm[sel].mean())); ys.append(float(np.percentile(vv[sel], 20)))  # lower envelope
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 4:
        return dict(ok=False)
    slope, intercept = np.polyfit(xs, ys, 1)
    # var = slope*mean + intercept  ->  Poisson: var = mean/k  ->  k (counts per grey level) = 1/slope
    return dict(ok=True, var_vs_mean_slope=float(slope), var_intercept=float(intercept),
                poisson_counts_per_grey_level=float(1 / slope) if slope > 1e-6 else None,
                gaussian_read_noise_grey=float(np.sqrt(max(intercept, 0))),
                median_flat_std=float(np.sqrt(np.median(vv))), bins_mean=xs.tolist(), bins_var=ys.tolist())


def measure_psf(img, n_edges=400, half=6):
    """Edge-spread function across the strongest straight-ish edges -> Gaussian PSF sigma (px)."""
    a = img.astype(np.float32)
    gy, gx = np.gradient(ndi.gaussian_filter(a, 1.0))
    g = np.hypot(gx, gy)
    ys, xs = np.unravel_index(np.argsort(g.ravel())[::-1][: n_edges * 20], g.shape)
    profiles = []
    for y, x in zip(ys, xs):
        if half < y < a.shape[0] - half and half < x < a.shape[1] - half:
            nx, ny = gx[y, x] / (g[y, x] + 1e-6), gy[y, x] / (g[y, x] + 1e-6)
            t = np.arange(-half, half + 1)
            py, px = y + t * ny, x + t * nx
            prof = ndi.map_coordinates(a, [py, px], order=1)
            profiles.append((prof - prof[0]) / (prof[-1] - prof[0] + 1e-6))
        if len(profiles) >= n_edges:
            break
    if not profiles:
        return dict(ok=False)
    esf = np.median(np.array(profiles), axis=0)
    lsf = np.gradient(esf)
    t = np.arange(-half, half + 1)
    w = np.clip(lsf, 0, None); w /= w.sum() + 1e-9
    sigma = float(np.sqrt(np.sum(w * t * t) - np.sum(w * t) ** 2))
    return dict(ok=True, psf_sigma_px=sigma, esf=esf.tolist(), n_profiles=len(profiles))


def measure_field(img, sigma):
    lp = lowpass(img, sigma)
    amp = float(lp.std() / max(lp.mean(), 1e-6))
    f = lp - lp.mean(); F = np.fft.fft2(f); ac = np.fft.ifft2(np.abs(F) ** 2).real; ac /= ac[0, 0] + 1e-9
    cl = int(np.argmax(ac[0, : lp.shape[1] // 2] < np.exp(-1)))
    return lp, dict(relative_amplitude=amp, corr_length_px=cl)


def measure_curtaining(img, sigma_lp):
    """Stripes along the milling direction (columns): spectrum of column means after removing the low-pass field."""
    a = img.astype(np.float32) - lowpass(img, sigma_lp)
    col = a.mean(0); row = a.mean(1)
    col -= col.mean(); row -= row.mean()
    def dom(v):
        P = np.abs(np.fft.rfft(v)) ** 2; P[0] = 0
        k = int(np.argmax(P[1:]) + 1)
        return float(len(v) / k), float(P[k] / P.sum())
    wl_c, pk_c = dom(col); wl_r, pk_r = dom(row)
    return dict(column_stripe_relative_std=float(col.std() / max(img.mean(), 1e-6)),
                row_stripe_relative_std=float(row.std() / max(img.mean(), 1e-6)),
                column_dominant_wavelength_px=wl_c, column_peak_fraction=pk_c,
                row_dominant_wavelength_px=wl_r, row_peak_fraction=pk_r,
                stripes_are_vertical=bool(col.std() > 1.5 * row.std()))


def measure_shine_through(img):
    """Autocorrelation anisotropy in dark (pore-like) regions and brightness decay from solid edges into pores
    along +y / -y / +x / -x: the 52-degree shine-through shows as a longer decay along one y direction."""
    a = img.astype(np.float32)
    thr = np.percentile(a, 40)
    pore = ndi.gaussian_filter(a, 2) < thr
    f = (a - a.mean()) * pore
    F = np.fft.fft2(f); ac = np.fft.ifft2(np.abs(F) ** 2).real; ac /= ac[0, 0] + 1e-9
    def clen(line):
        b = np.where(line < np.exp(-1))[0]
        return int(b[0]) if len(b) else len(line)
    cy, cx = clen(ac[: a.shape[0] // 2, 0]), clen(ac[0, : a.shape[1] // 2])
    solid = ~pore
    dist = ndi.distance_transform_edt(pore)
    out = {}
    for name, shift in [("plus_y", (1, 0)), ("minus_y", (-1, 0)), ("plus_x", (0, 1)), ("minus_x", (0, -1))]:
        # brightness of pore pixels as a function of distance from solid, restricted to pixels whose nearest solid
        # lies in direction `shift`: approximated by comparing directional dilation bands
        prof = []
        band = solid.copy()
        for d in range(1, 13):
            band = np.roll(band, shift, axis=(0, 1)) & pore
            sel = band & (dist >= d - 0.5) & (dist < d + 1.5)
            prof.append(float(a[sel].mean()) if sel.sum() > 15 else np.nan)
        prof = np.array(prof)
        ok = np.isfinite(prof)
        if ok.sum() >= 4:
            d = np.arange(1, 13)[ok]; p = prof[ok] - np.nanmin(prof)
            p = np.clip(p, 1e-3, None)
            slope = np.polyfit(d, np.log(p), 1)[0]
            out[name] = dict(decay_length_px=float(-1 / slope) if slope < 0 else None, profile=prof.tolist())
    return dict(pore_corr_len_y_px=cy, pore_corr_len_x_px=cx, anisotropy_y_over_x=float(cy / max(cx, 1)),
                directional_decay=out)


def paper_like_segmentation(vol, clean_radius=2):
    """Gaussian smoothing -> Otsu threshold -> 3-D open/close with a ball of radius r (the paper's clean-up)."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import ball
    v = ndi.gaussian_filter(vol.astype(np.float32), 1.0)
    thr = threshold_otsu(v)
    solid = v > thr
    b = ball(clean_radius)
    solid = ndi.binary_closing(ndi.binary_opening(solid, b), b)
    return solid, float(thr)


def local_thickness(pore, radii=None):
    pore = pore.astype(bool)
    dt = ndi.distance_transform_edt(pore)
    if radii is None:
        rmax = int(np.ceil(dt.max()))
        radii = sorted(set(list(range(1, min(rmax, 12) + 1)) + [int(r) for r in np.geomspace(12, max(rmax, 12), 10)]))
    lt = np.zeros(pore.shape, np.float32)
    for r in sorted(radii, reverse=True):
        centres = dt >= r
        if not centres.any():
            continue
        reach = ndi.distance_transform_edt(~centres) <= r
        sel = reach & pore & (lt == 0)
        lt[sel] = 2.0 * r
    lt[pore & (lt == 0)] = 1.0
    return lt


def measure_structure(stack, z0, y0, x0, size, voxel_nm, clean_radius):
    vol = np.stack([stack.slice(z)[y0:y0 + size, x0:x0 + size] for z in range(z0, z0 + size)])
    solid, thr = paper_like_segmentation(vol, clean_radius)
    pore = ~solid
    lt = local_thickness(pore) * voxel_nm
    d = lt[lt > 0]
    mu, sg = float(np.log(d).mean()), float(np.log(d).std())
    lab, nlab = ndi.label(pore)
    sizes = np.bincount(lab.ravel())[1:]
    return dict(crop=[int(z0), int(y0), int(x0), int(size)], otsu_threshold=thr, clean_radius=clean_radius,
                porosity=float(pore.mean()), pore_mean_nm=float(d.mean()), pore_std_nm=float(d.std()),
                pore_max_nm=float(d.max()), lognormal_mu=mu, lognormal_sigma=sg,
                pore_connected_fraction=float(sizes.max() / sizes.sum()) if nlab else 0.0,
                slice_porosity_std=float((1 - solid.reshape(size, -1).mean(1)).std()),
                psd_hist=np.histogram(d, bins=np.linspace(0, 650, 27))[0].tolist()), vol, solid


def c2st_floor(img_a, img_b, patch=11, n=2500, seed=0):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
        from scipy.spatial import cKDTree
    except ImportError:
        return dict(ok=False, reason="scikit-learn not installed")
    rng = np.random.default_rng(seed)
    def samp(img):
        H, W = img.shape; ys = rng.integers(0, H - patch, n); xs = rng.integers(0, W - patch, n)
        return np.stack([img[y:y + patch, x:x + patch].ravel() for y, x in zip(ys, xs)]).astype(np.float32)
    A, B = samp(img_a), samp(img_b)
    X = np.concatenate([A, B]); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    Xn = (X - A.mean()) / (A.std() + 1e-6)
    _, I = cKDTree(Xn).query(Xn, k=6)
    knn = float((y[I[:, 1:]] == y[:, None]).mean())
    prob = cross_val_predict(RandomForestClassifier(150, max_depth=8, random_state=seed, n_jobs=-1), Xn, y, cv=5, method="predict_proba")[:, 1]
    return dict(ok=True, knn_acc=knn, rf_auc=float(roc_auc_score(y, prob)))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True); ap.add_argument("--out", default="inspection")
    ap.add_argument("--voxel_nm", type=float, default=6.0); ap.add_argument("--slice_step", type=int, default=25)
    ap.add_argument("--crop", type=int, default=512); ap.add_argument("--struct_size", type=int, default=256)
    ap.add_argument("--clean_radius", type=int, default=2); ap.add_argument("--max_slices_full_stats", type=int, default=200)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    name = os.path.splitext(os.path.basename(a.tif))[0]
    out = os.path.join(a.out, name); os.makedirs(os.path.join(out, "crops"), exist_ok=True)
    st = Stack(a.tif)
    Z, Y, X = st.shape if len(st.shape) == 3 else (1,) + tuple(st.shape)
    meta = dict(file=a.tif, bytes=os.path.getsize(a.tif), shape=[Z, Y, X], dtype=st.dtype, n_slices=Z,
                imagej_metadata={k: (str(v)[:300]) for k, v in st.ij.items()}, tiff_tags=st.tags,
                voxel_nm_assumed=a.voxel_nm, field_um=[round(Y * a.voxel_nm / 1000, 2), round(X * a.voxel_nm / 1000, 2)],
                depth_um=round(Z * a.voxel_nm / 1000, 2), bytes_per_voxel=os.path.getsize(a.tif) / max(Z * Y * X, 1))
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=2)
    log(f"{name}: shape {st.shape} dtype {st.dtype} ({meta['bytes_per_voxel']:.2f} B/voxel)")

    # ---- per-slice statistics (drift) and slowly-varying field across slices
    zs = list(range(0, Z, max(1, a.slice_step)))[: a.max_slices_full_stats]
    rows, prev_lp, ar = [], None, []
    for z in zs:
        im = st.to8(st.slice(z)); lp, fld = measure_field(im, sigma=min(Y, X) / 12)
        if prev_lp is not None and prev_lp.shape == lp.shape:
            ar.append(float(np.corrcoef((prev_lp - prev_lp.mean()).ravel(), (lp - lp.mean()).ravel())[0, 1]))
        prev_lp = lp
        rows.append(dict(z=z, mean=float(im.mean()), std=float(im.std()), p5=float(np.percentile(im, 5)),
                         p50=float(np.percentile(im, 50)), p95=float(np.percentile(im, 95)),
                         field_rel_amp=fld["relative_amplitude"], field_corr_len_px=fld["corr_length_px"]))
    with open(os.path.join(out, "per_slice_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    means = np.array([r["mean"] for r in rows]); stds = np.array([r["std"] for r in rows])
    drift = dict(slice_mean_median=float(np.median(means)), slice_mean_std=float(means.std()),
                 adjacent_slice_mean_rel_change=float(np.median(np.abs(np.diff(means)) / np.maximum(means[:-1], 1))),
                 slice_std_median=float(np.median(stds)), field_slice_to_slice_corr=float(np.median(ar)) if ar else None,
                 sampled_slices=zs)
    log("drift measured")

    # ---- single-slice measurements at three depths, three positions
    mid = [Z // 4, Z // 2, 3 * Z // 4]
    per = []
    for z in mid:
        im = st.to8(st.slice(z))
        noise = measure_noise(im); psf = measure_psf(im); _, fld = measure_field(im, min(Y, X) / 12)
        curt = measure_curtaining(im, min(Y, X) / 12)
        sh = measure_shine_through(im[Y // 4: 3 * Y // 4, X // 4: 3 * X // 4] if min(Y, X) >= 400 else im)
        per.append(dict(z=z, noise=noise, psf=psf, field=fld, curtaining=curt, shine_through=sh,
                        histogram=np.histogram(im, bins=32, range=(0, 255))[0].tolist()))
        c = a.crop
        for tag, (yy, xx) in {"centre": (Y // 2 - c // 2, X // 2 - c // 2), "topleft": (Y // 8, X // 8), "bottomright": (7 * Y // 8 - c, 7 * X // 8 - c)}.items():
            yy, xx = int(np.clip(yy, 0, max(Y - c, 0))), int(np.clip(xx, 0, max(X - c, 0)))
            from PIL import Image
            Image.fromarray(im[yy:yy + c, xx:xx + c]).save(os.path.join(out, "crops", f"z{z:05d}_{tag}_y{yy}_x{xx}.png"))
        log(f"slice {z}: noise/psf/field/curtaining/shine-through measured")

    # ---- structure through the paper-like operator on a central cube
    s = min(a.struct_size, Z, Y, X)
    struct, vol, solid = measure_structure(st, Z // 2 - s // 2, Y // 2 - s // 2, X // 2 - s // 2, s, a.voxel_nm, a.clean_radius)
    log(f"structure: porosity {struct['porosity']:.3f} mean pore {struct['pore_mean_nm']:.0f} nm mu {struct['lognormal_mu']:.3f} sigma {struct['lognormal_sigma']:.3f}")

    # ---- C2ST floor between disjoint z-blocks (how different is real from real?)
    floor = c2st_floor(st.to8(st.slice(Z // 4)), st.to8(st.slice(3 * Z // 4)))
    floor_struct = c2st_floor(ndi.zoom(st.to8(st.slice(Z // 4)).astype(np.float32), 1 / 3, order=1), ndi.zoom(st.to8(st.slice(3 * Z // 4)).astype(np.float32), 1 / 3, order=1)) if floor.get("ok") else floor
    json.dump(dict(metadata=meta, drift=drift, per_slice=per, structure={k: v for k, v in struct.items()},
                   c2st_floor_texture_11px=floor, c2st_floor_structure_33px=floor_struct),
              open(os.path.join(out, "measurements.json"), "w"), indent=2, default=float)

    # ---- figures
    im = st.to8(st.slice(Z // 2))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(im, cmap="gray", vmin=0, vmax=255); ax[0].set_title(f"{name} slice {Z//2} ({Y}x{X})"); ax[0].axis("off")
    ax[1].hist(im.ravel(), bins=64, range=(0, 255), color="k"); ax[1].set_title("intensity histogram")
    ax[2].plot(zs, means, label="mean"); ax[2].plot(zs, stds, label="std"); ax[2].set_xlabel("slice"); ax[2].legend(); ax[2].set_title("per-slice drift")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_overview.png"), dpi=110); plt.close(fig)
    p = per[1]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    if p["noise"].get("ok"):
        ax[0].plot(p["noise"]["bins_mean"], p["noise"]["bins_var"], "o-"); ax[0].set_xlabel("local mean"); ax[0].set_ylabel("local variance (flat windows)")
        ax[0].set_title(f"noise: {p['noise']['poisson_counts_per_grey_level'] and round(p['noise']['poisson_counts_per_grey_level'],2)} counts/grey, read {p['noise']['gaussian_read_noise_grey']:.1f}")
    if p["psf"].get("ok"):
        ax[1].plot(range(-6, 7), p["psf"]["esf"], "o-"); ax[1].set_title(f"edge spread: PSF sigma {p['psf']['psf_sigma_px']:.2f} px")
    side_len = min(512, Y, X); rad = radial_spectrum(im[:side_len, :side_len]); kmax = min(200, len(rad))
    ax[2].loglog(np.arange(2, kmax), rad[2:kmax]); ax[2].set_title("radial power spectrum"); ax[2].set_xlabel(f"cycles / {side_len} px")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_noise_psf_spectrum.png"), dpi=110); plt.close(fig)
    side = np.stack([st.slice(z)[:, X // 2] for z in range(0, Z, max(1, Z // 400))])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(st.to8(side), cmap="gray", aspect="auto"); ax[0].set_title("y-z side view (column x = X/2; z subsampled)"); ax[0].axis("off")
    centres = np.linspace(12.5, 637.5, 26); ax[1].bar(centres, np.array(struct["psd_hist"]) / max(sum(struct["psd_hist"]), 1), width=22)
    ax[1].set_xlabel("pore diameter (nm)"); ax[1].set_title(f"PSD through paper-like operator: mu {struct['lognormal_mu']:.2f} sigma {struct['lognormal_sigma']:.2f}, porosity {struct['porosity']:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_sideview_psd.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(vol[s // 2], cmap="gray"); ax[0].set_title("central cube, mid slice"); ax[0].axis("off")
    ax[1].imshow(solid[s // 2], cmap="gray"); ax[1].set_title("paper-like segmentation (solid = white)"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_segmentation_check.png"), dpi=110); plt.close(fig)
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
