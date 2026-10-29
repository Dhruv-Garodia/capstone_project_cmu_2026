"""REAL | synth image | synth truth | truth over image. Window follows drift."""
import sys, numpy as np, tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
out, real, rz0, syn, tru, y0, x0, dr, sz, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])
tr, ts = tifffile.TiffFile(real), tifffile.TiffFile(syn)
import os
from pathlib import Path as _P
if os.path.isdir(tru):
    class _T:
        def __init__(s, d): s.f = sorted(_P(d).glob('slice_*.png'))
        @property
        def pages(s): return [type('pg', (), {'asarray': (lambda self, f=f: np.array(Image.open(f)))})() for f in s.f]
    tt = _T(tru)
else:
    tt = tifffile.TiffFile(tru)
def lab(a, s):
    im = Image.fromarray(a); d = ImageDraw.Draw(im); d.rectangle([0, 0, 8 * len(s) + 6, 14], fill=(0, 0, 0)); d.text((3, 2), s, fill=(255, 255, 255)); return np.array(im)
frames = []
for i in range(n):
    y = y0 + int(round(dr * i)); c = lambda a: a[y:y + sz, x0:x0 + sz]
    r = c(tr.pages[rz0 + i].asarray()); s = c(ts.pages[i].asarray()); t = c(tt.pages[i].asarray())
    g = lambda a: np.repeat(a[..., None], 3, 2)
    tv = np.where(t == 0, 0, np.where(t == 255, 255, 110)).astype(np.uint8)
    pore = t == 0; edge = pore & ~ndi.binary_erosion(pore)
    ov = g(s).copy(); ov[pore] = (0.6 * ov[pore] + 0.4 * np.array([255, 60, 40])).astype(np.uint8); ov[edge] = (255, 40, 20)
    tiles = [lab(g(r), "REAL"), lab(g(s), f"SYNTH {i}"), lab(g(tv), "TRUTH (black=pore)"), lab(ov, "truth pores on image")]
    frames.append(Image.fromarray(np.concatenate([np.pad(a, ((0, 0), (0, 4), (0, 0))) for a in tiles], 1)[:, :-4]).quantize(96))
frames[0].save(out, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)
print(out)
