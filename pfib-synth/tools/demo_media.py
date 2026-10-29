"""README media from generated packages: per-material GIFs (image | truth | truth on image),
an all-materials GIF, a static gallery and an x-z reslice. Synthetic frames only.
    python tools/demo_media.py PACKAGES_DIR docs/media
PACKAGES_DIR holds team_review_synth_<material>/ or <material>/ folders from generate.py.
"""
import sys, json
from pathlib import Path
import numpy as np, tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
MATS = ["comp_like", "uncomp_like", "pristine_like", "mudcrack", "fibre_mat", "fibre_coated", "faceted"]
Y0 = {"comp": 150, "uncomp": 180, "pristine": 400}
N, SZ, X0, DR = 48, 256, 700, 1.1


def pkg(m):
    for p in (src / f"team_review_synth_{m}", src / m):
        if p.exists():
            return p


def lab(a, s):
    im = Image.fromarray(a); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 7 * len(s) + 8, 15], fill=(0, 0, 0)); d.text((4, 2), s, fill=(255, 255, 255)); return np.array(im)


def load(m):
    p = pkg(m); meta = json.loads((p / "stack.json").read_text()); tj = json.loads((p / "truth/truth.json").read_text())
    st = tifffile.imread(p / meta["stack_file"]); T = lambda z: np.array(Image.open(p / f"truth/masks/slice_{z:04d}.png"))
    return st, T, Y0[tj["style_calibrated_to"]]


def crop(a, i, y0, sz=SZ):
    y = y0 + int(round(DR * i)); return a[y:y + sz, X0:X0 + sz]


g3 = lambda a: np.repeat(a[..., None], 3, 2)
tiles = {}
for m in MATS:
    st, T, y0 = load(m); fr = []
    for i in range(N):
        s, t = crop(st[i], i, y0), crop(T(i), i, y0)
        tv = np.where(t == 0, 0, np.where(t == 255, 255, 110)).astype(np.uint8)
        pore = t == 0; edge = pore & ~ndi.binary_erosion(pore)
        ov = g3(s).copy(); ov[pore] = (0.6 * ov[pore] + 0.4 * np.array([255, 60, 40])).astype(np.uint8); ov[edge] = (255, 40, 20)
        row = [lab(g3(s), f"{m}  slice {i}"), lab(g3(tv), "truth: black = pore"), lab(ov, "truth on image")]
        fr.append(Image.fromarray(np.concatenate([np.pad(a, ((0, 0), (0, 4), (0, 0)), constant_values=255) for a in row], 1)[:, :-4]).quantize(64))
    fr[0].save(out / f"{m}.gif", save_all=True, append_images=fr[1:], duration=120, loop=0, optimize=True)
    tiles[m] = [crop(st[i], i, y0, 200) for i in range(N)]
    if m == "comp_like":
        # x-z reslice through the stack: one object cut slice by slice
        y = y0 + 60; xs = slice(X0 - 200, X0 + 312)
        im = np.stack([st[z][y + int(round(DR * z)), xs] for z in range(len(st))])
        tr = np.stack([T(z)[y + int(round(DR * z)), xs] for z in range(len(st))])
        tr = np.where(tr == 0, 0, np.where(tr == 255, 255, 110)).astype(np.uint8)
        im = np.repeat(im, 2, 0); tr = np.repeat(tr, 2, 0)
        Image.fromarray(np.concatenate([im, np.full((im.shape[0], 6), 255, np.uint8), tr], 1)).save(out / "reslice_comp_like.png")
    print(m, round((out / f"{m}.gif").stat().st_size / 1e6, 1), "MB", flush=True)

# all materials, image only, 4 x 2 with a caption tile
fr = []
for i in range(N):
    c = Image.new("L", (4 * 204 + 4, 2 * 220 + 4), 255); d = ImageDraw.Draw(c)
    for k, m in enumerate(MATS):
        x, y = 4 + (k % 4) * 204, 4 + (k // 4) * 220
        c.paste(Image.fromarray(tiles[m][i]), (x, y + 16)); d.text((x + 2, y + 2), m.replace("_", " "), fill=0)
    x, y = 4 + 3 * 204, 4 + 220
    for j, s in enumerate(["synthetic pFIB-SEM", "seven materials", "exact truth for", "every pixel", "", f"slice {i + 1} / {N}"]):
        d.text((x + 12, y + 24 + j * 18), s, fill=0)
    fr.append(c.quantize(24))
fr[0].save(out / "all_materials.gif", save_all=True, append_images=fr[1:], duration=120, loop=0, optimize=True)
# static gallery: image over truth for each material
G = Image.new("L", (7 * 204 + 4, 2 * 200 + 30), 255); d = ImageDraw.Draw(G)
for k, m in enumerate(MATS):
    st, T, y0 = load(m); s, t = crop(st[20], 20, y0, 200), crop(T(20), 20, y0, 200)
    G.paste(Image.fromarray(s), (4 + k * 204, 22)); G.paste(Image.fromarray(np.where(t == 0, 0, np.where(t == 255, 255, 110)).astype(np.uint8)), (4 + k * 204, 226))
    d.text((6 + k * 204, 6), m.replace("_", " "), fill=0)
G.save(out / "gallery.png", optimize=True)
for f in sorted(out.iterdir()):
    print(f.name, round(f.stat().st_size / 1e6, 2), "MB")
