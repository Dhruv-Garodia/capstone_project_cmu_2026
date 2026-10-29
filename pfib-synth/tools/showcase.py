"""Showcase GIFs for one generated package. Synthetic frames only.

    python tools/showcase.py PACKAGE_DIR OUT_DIR [--label NAME]

Writes into OUT_DIR:
  <name>_milling.gif     the block being milled: the beam takes one slice at a time and the
                         cut face (what the detector sees) moves into the material
  <name>_rotate.gif      the material in 3-D (truth solid, drift removed), turning
  <name>_slices.gif      image | truth | truth on image, slice by slice
  <name>_reslice.gif     an x-z cut sweeping through the block: image and truth
  <name>_shine.gif       where the threshold rule misses pores, coloured by how far down the
                         lit floor is (red 1 slice, orange 2-3, yellow 4-8)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("pkg"); ap.add_argument("out"); ap.add_argument("--label", default=None)
ap.add_argument("--size", type=int, default=224, help="crop size in px")
ap.add_argument("--frames", type=int, default=None, help="slices to use (default: all, up to 118)")
ap.add_argument("--only", default=None, help="comma list of milling,rotate,slices,reslice,shine")
a = ap.parse_args()
pkg, out = Path(a.pkg), Path(a.out); out.mkdir(parents=True, exist_ok=True)
meta = json.loads((pkg / "stack.json").read_text()); tj = json.loads((pkg / "truth/truth.json").read_text())
name = a.label or meta["name"].replace("synth_", "")
only = set(a.only.split(",")) if a.only else {"milling", "rotate", "slices", "reslice", "shine"}
drift = float(tj["optics"]["drift"]); par = float(tj["optics"]["parallax"])
stack = tifffile.imread(pkg / meta["stack_file"])
N = min(a.frames or len(stack), len(stack))
T = np.stack([np.array(Image.open(pkg / f"truth/masks/slice_{z:04d}.png")) for z in range(N)])
H, W = stack.shape[1:]
S = a.size
# window inside the layer: the rows where truth is mostly not 'outside' on frame 0, drift-followed
lay = (T[0] != 128).mean(1)
rows = np.where(lay > 0.9)[0]
y0 = int(rows[0] + max(0, (len(rows) - S - int(drift * N)) // 2)) if len(rows) else H // 3
y0 = min(y0, H - S - int(np.ceil(drift * N)) - 1)
x0 = max(0, W // 2 - S // 2)
yy = lambda z: y0 + int(round(drift * z))
img = np.stack([stack[z, yy(z):yy(z) + S, x0:x0 + S] for z in range(N)])        # (N, S, S), drift removed
tru = np.stack([T[z, yy(z):yy(z) + S, x0:x0 + S] for z in range(N)])
g3 = lambda x: np.repeat(x[..., None], 3, 2)
tv = lambda t: np.where(t == 0, 0, np.where(t == 255, 255, 110)).astype(np.uint8)


def lab(arr, s, at=(0, 0)):
    im = Image.fromarray(arr); d = ImageDraw.Draw(im)
    d.rectangle([at[0], at[1], at[0] + 7 * len(s) + 8, at[1] + 15], fill=(0, 0, 0)); d.text((at[0] + 4, at[1] + 2), s, fill=(255, 255, 255))
    return np.array(im)


def save(frames, fn, dur=110, colors=64):
    fr = [Image.fromarray(f).quantize(colors) if isinstance(f, np.ndarray) else f.quantize(colors) for f in frames]
    fr[0].save(out / fn, save_all=True, append_images=fr[1:], duration=dur, loop=0, optimize=True)
    print(f"  {fn} {(out / fn).stat().st_size / 1e6:.1f} MB", flush=True)


# ---------------------------------------------------------------- milling
def face(canvas, tex, quad):
    """Paste texture (h, w, 3) onto the parallelogram quad = (p00, p01, p10): image of
    texture corners (0,0), (0,w), (h,0). Affine, so orthographic faces stay exact."""
    (ax, ay), (bx, by), (cx, cy) = quad
    h, w = tex.shape[:2]
    # output (X, Y) = A @ (u, v) + p00, with u along width and v along height
    A = np.array([[(bx - ax) / w, (cx - ax) / h], [(by - ay) / w, (cy - ay) / h]])
    Ai = np.linalg.inv(A)
    coeffs = (Ai[0, 0], Ai[0, 1], -(Ai[0, 0] * ax + Ai[0, 1] * ay), Ai[1, 0], Ai[1, 1], -(Ai[1, 0] * ax + Ai[1, 1] * ay))
    t = Image.fromarray(tex).transform(canvas.size, Image.AFFINE, coeffs, resample=Image.BILINEAR)
    m = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(m).polygon([(ax, ay), (bx, by), (bx + cx - ax, by + cy - ay), (cx, cy)], fill=255)
    canvas.paste(t, (0, 0), m)


def milling():
    # depth axis drawn up-right (the block recedes away from the viewer); the beam comes in at 52 deg
    D = N                                  # slices in the block
    k = 1.15                               # px of canvas per slice of depth
    dx, dy = 0.62 * k, -0.42 * k
    Wc, Hc = int(S + D * dx + 120), int(S + D * -dy + 110)
    ox, oy = 30, int(D * -dy + 70)
    shade = lambda x, f: (x.astype(np.float32) * f).astype(np.uint8)
    frames = []
    for i in range(0, N - 1, max(2, N // 50)):
        c = Image.new("RGB", (Wc, Hc), (18, 20, 24)); d = ImageDraw.Draw(c)
        rem = N - i                         # slices left
        fx, fy = ox + i * dx, oy + i * dy   # front-face top-left moves back as slices go
        front = g3(img[i])
        top = g3(img[i:, 0, :])             # x-z plane along the top edge: (rem, S)
        side = g3(img[i:, :, -1].T)         # y-z plane along the right edge: (S, rem)
        # top face: width S along x, height rem along depth
        face(c, shade(top, 0.82)[::-1], ((fx + rem * dx, fy + rem * dy), (fx + S + rem * dx, fy + rem * dy), (fx, fy)))
        face(c, shade(side, 0.62), ((fx + S, fy), (fx + S + rem * dx, fy + rem * dy), (fx + S, fy + S)))
        face(c, front, ((fx, fy), (fx + S, fy), (fx, fy + S)))
        # milled-away region: faint outline of where the block was
        d.line([(ox, oy), (fx, fy)], fill=(70, 74, 82)); d.line([(ox + S, oy), (fx + S, fy)], fill=(70, 74, 82))
        d.line([(ox, oy + S), (fx, fy + S)], fill=(70, 74, 82)); d.line([(ox + S, oy + S), (fx + S, fy + S)], fill=(70, 74, 82))
        # ion beam: a sheet coming down onto the face
        # the ion beam: a fan from the column, grazing the front edge, removing one slice
        ov = Image.new("RGBA", c.size, (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
        od.polygon([(fx + S * 0.35, 12), (fx + S * 0.65, 12), (fx + S, fy), (fx, fy)], fill=(90, 200, 255, 55))
        c.paste(Image.alpha_composite(c.convert("RGBA"), ov).convert("RGB")); d = ImageDraw.Draw(c)
        d.line([(fx, fy), (fx + S, fy)], fill=(120, 220, 255), width=2)
        d.text((fx + S * 0.66 + 6, 14), "ion beam", fill=(120, 220, 255))
        d.text((fx + 6, fy + S + 8), "cut face = the frame the electron beam images (52 deg)", fill=(200, 200, 200))
        d.text((10, Hc - 18), f"{name}: slice {i + 1} / {N}", fill=(240, 240, 240))
        frames.append(c)
    save(frames, f"{name}_milling.gif", dur=90)


# ---------------------------------------------------------------- 3-D rotation
def rotate():
    zl = float(tj.get("slicing", {}).get("zlen", 0) or 1.0)
    sol = (tru != 0)                                   # solid + outside: render as material
    sol = sol & (tru != 128)
    v = ndi.gaussian_filter(sol[:, ::2, ::2].astype(np.float32), 0.7)   # (N, S/2, S/2): z, y, x
    v = np.pad(ndi.zoom(v, (min(1.0, (S / 2) / N), 1, 1), order=1), 2)
    frames = []
    for ang in range(0, 360, 8):
        r = ndi.rotate(v, ang, axes=(0, 2), reshape=True, order=1)
        r = ndi.rotate(r, -22, axes=(0, 1), reshape=True, order=1) > 0.5   # tilt toward the viewer
        hit = r.any(0); dep = np.argmax(r, 0).astype(np.float32)
        dep[~hit] = np.nan
        gy, gx = np.gradient(ndi.gaussian_filter(np.nan_to_num(dep, nan=np.nanmax(dep) if hit.any() else 0), 0.8))
        n = np.stack([-gx, -gy, np.ones_like(gx)]); n /= np.linalg.norm(n, axis=0)
        L = np.array([0.35, -0.75, 0.6]); L /= np.linalg.norm(L)
        lam = np.clip((n * L[:, None, None]).sum(0), 0, 1)
        fog = 1 - 0.55 * (np.nan_to_num(dep) / max(r.shape[0], 1))
        base = np.array([214, 186, 150], np.float32)
        col = (base[None, None] * (0.25 + 0.75 * lam)[..., None] * fog[..., None])
        col[~hit] = (18, 20, 24)
        im = Image.fromarray(col.clip(0, 255).astype(np.uint8))
        im = im.resize((im.width * 2, im.height * 2), Image.NEAREST)
        c = Image.new("RGB", (int(S * 1.9), int(S * 1.9)), (18, 20, 24))
        c.paste(im, ((c.width - im.width) // 2, (c.height - im.height) // 2))
        ImageDraw.Draw(c).text((8, 6), f"{name}: solid phase in 3-D (exact truth)", fill=(240, 240, 240))
        frames.append(c)
    save(frames, f"{name}_rotate.gif", dur=80)


# ---------------------------------------------------------------- slices
def slices():
    fr = []
    for i in range(0, N, max(2, N // 50)):
        s, t = img[i], tru[i]
        pore = t == 0; edge = pore & ~ndi.binary_erosion(pore)
        ov = g3(s).copy(); ov[pore] = (0.6 * ov[pore] + 0.4 * np.array([255, 60, 40])).astype(np.uint8); ov[edge] = (255, 40, 20)
        row = [lab(g3(s), f"{name}  slice {i + 1}"), lab(g3(tv(t)), "truth: black = pore"), lab(ov, "truth on image")]
        fr.append(np.concatenate([np.pad(x, ((0, 0), (0, 4), (0, 0)), constant_values=255) for x in row], 1)[:, :-4])
    save(fr, f"{name}_slices.gif", colors=128)


# ---------------------------------------------------------------- reslice
def reslice():
    fr = []
    for y in range(0, S, 4):
        a_ = np.repeat(img[:, y, :], 2, 0); b_ = np.repeat(tv(tru[:, y, :]), 2, 0)
        row = [lab(g3(a_), f"x-z cut, row {y}"), lab(g3(b_), "truth")]
        fr.append(np.concatenate([np.pad(x, ((0, 0), (0, 4), (0, 0)), constant_values=255) for x in row], 1)[:, :-4])
    save(fr, f"{name}_reslice.gif", dur=90)


# ---------------------------------------------------------------- shine-through
def shine():
    sm = pkg / "start_masks"
    if not sm.exists():
        return
    shift = drift + par
    fr = []
    for i in range(0, N - 9, max(1, (N - 9) // 50)):
        t0 = tru[i]; pore = t0 == 0
        d = np.full(t0.shape, 9, np.int16)
        open_ = pore.copy()
        for k in range(1, 9):
            tk = T[i + k, yy(i) + int(round(par * k)):, x0:x0 + S][:S]
            if tk.shape[0] < S:
                break
            hit = open_ & (tk == 255); d[hit] = k; open_ &= ~hit
        rule = np.array(Image.open(sm / f"slice_{i:04d}.png"))[yy(i):yy(i) + S, x0:x0 + S] == 0
        miss = pore & ~rule
        ov = g3(img[i]).copy()
        for lo, hi, colr in ((1, 1, (230, 40, 40)), (2, 3, (255, 150, 30)), (4, 8, (250, 230, 60))):
            m = miss & (d >= lo) & (d <= hi); ov[m] = (0.35 * ov[m] + 0.65 * np.array(colr)).astype(np.uint8)
        ok = pore & rule; ov[ok & ~ndi.binary_erosion(ok)] = (60, 200, 255)
        row = [lab(g3(img[i]), f"{name}  slice {i + 1}"), lab(ov, "rule misses: red 1, orange 2-3, yellow 4-8 deep")]
        fr.append(np.concatenate([np.pad(x, ((0, 0), (0, 4), (0, 0)), constant_values=255) for x in row], 1)[:, :-4])
    save(fr, f"{name}_shine.gif", colors=128)


print(name, f"{N} slices, crop {S} px at y0 {y0}, x0 {x0}", flush=True)
for f in ("slices", "reslice", "shine", "rotate", "milling"):
    if f in only:
        globals()[f]()
