"""One felt iteration: generate at 0.125 um voxels, render plan view, box-downsample 2x to the real image's 0.25 um/px, compare."""
import sys, time, json, os; sys.path.insert(0, '/home/claude/eponge'); os.chdir('/home/claude/eponge')
import numpy as np
from PIL import Image, ImageDraw
from eponge_synth.felt import sheet_felt
from eponge_synth.sem_render import SEMSimulator, SEMParams
from eponge_synth.descriptors import image_stats
REAL = np.array(Image.open('samples/real_uploads/flake_felt.png'))[:224, 238:462]   # held-out right half only

def hf_fraction(a):  # energy above 0.25 cycles/px : crispness
    F = np.abs(np.fft.fftshift(np.fft.fft2(a.astype(float) - a.mean()))) ** 2
    h, w = F.shape; yy, xx = np.indices(F.shape); r = np.hypot(yy - h // 2, xx - w // 2) / (min(h, w) / 2)
    return float(F[r > 0.5].sum() / F.sum())

def corr_len(a):
    a = a.astype(float) - a.mean(); F = np.fft.fft2(a); ac = np.fft.ifft2(np.abs(F) ** 2).real; ac /= ac[0, 0]
    return int(np.argmax(ac[0, :a.shape[1] // 2] < np.exp(-1))), int(np.argmax(ac[:a.shape[0] // 2, 0] < np.exp(-1)))

def summarize(a):
    st = image_stats(a); st['hf_fraction'] = hf_fraction(a); st['corr_len'] = corr_len(a)
    st['p5'], st['p50'], st['p95'] = [float(np.percentile(a, q)) for q in (5, 50, 95)]
    return st

def run(tag, gen_kw, sem_kw, seed=3, shape=(96, 448, 448), z_view=0, save_material=False, reuse=None):
    t = time.time()
    if reuse is not None:
        f = reuse
    else:
        f = sheet_felt(shape=shape, voxel_um=0.125, seed=seed, n_sheets=40000, **gen_kw)
    tg = time.time() - t; t = time.time()
    P = SEMParams(**sem_kw)
    img, _ = SEMSimulator(f.material, 125.0, P, seed=1, edge_map=f.edges, normal_sigma=1.6).image(z_view)
    img2 = img.reshape(img.shape[0] // 2, 2, img.shape[1] // 2, 2).mean((1, 3)).astype(np.uint8)   # -> 0.25 um/px
    st = summarize(img2); rs = summarize(REAL)
    print(f"[{tag}] gen {tg:.0f}s render {time.time()-t:.0f}s deposited={f.meta['n_deposited']} porosity={f.meta['realized_porosity']:.3f}")
    keys = ['mean', 'std', 'entropy_bits', 'grad_mag_mean', 'local_contrast', 'spectrum_slope', 'hf_fraction', 'p5', 'p50', 'p95', 'corr_len']
    print('   real :', ' '.join(f"{k}={rs[k]:.2f}" if not isinstance(rs[k], tuple) else f"{k}={rs[k]}" for k in keys))
    print('   synth:', ' '.join(f"{k}={st[k]:.2f}" if not isinstance(st[k], tuple) else f"{k}={st[k]}" for k in keys))
    Image.fromarray(img2).save(f'samples/felt_{tag}.png')
    if save_material:
        np.save('samples/felt_material_hr.npy', f.material); np.save('samples/felt_edges_hr.npy', f.edges); json.dump(dict(gen=gen_kw, sem=sem_kw, meta=f.meta), open('samples/felt_hr_meta.json', 'w'), indent=2, default=float)
    run.last = f
    return img2

def montage(tags, out):
    S = 224; m = Image.new('L', (S * (len(tags) + 1) + 10 * len(tags), S + 22), 255); d = ImageDraw.Draw(m)
    m.paste(Image.fromarray(REAL), (0, 22)); d.text((4, 4), 'REAL', fill=0)
    for i, t in enumerate(tags):
        m.paste(Image.open(f'samples/felt_{t}.png'), ((i + 1) * (S + 10), 22)); d.text(((i + 1) * (S + 10) + 4, 4), t, fill=0)
    m.save(out)
