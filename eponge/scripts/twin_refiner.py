"""Learned appearance for the digital twin: a conditional GAN from 3D structure to microscope image.

The physics twin (simulate.py) places shine-through correctly but its images look like a two-tone mask:
flat pores, no depth blur, no particle texture, and no model of pristine's dark-rim contrast. The final
hand-annotated masks give paired data (3D structure -> real image of the same slice), so the appearance can
be learned instead of hand-modelled (pix2pix, Isola et al. 2017):

  condition (per pixel, computed from the 3D structure along the slid viewing ray, as in simulate.py):
      pore at the cut plane, outside-the-layer, depth to the first solid behind the plane (/ t_max),
      solid at depths 1, 2, 4, 8, 16 behind the plane, acquisition one-hot (comp, uncomp, pristine), noise
  generator: U-Net (ResNet-34 encoder) -> grey image;  discriminator: 70 px PatchGAN (LSGAN)
  loss: L1 x lambda + adversarial

Trained on train slices 0-83 only; slices 87-117 are held out to measure realism on unseen structure.
Applied to any structure with known phase (SliceGAN volumes, hand-label volumes), it gives realistic images
whose masks stay exact, because the structure is the input.

    python eponge/scripts/twin_refiner.py train  [--iters 6000]
    python eponge/scripts/twin_refiner.py eval       # held-out realism vs the physics twin -> runs/twin_refiner/eval.json, panel.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eponge import simulate as SIM  # noqa: E402
from eponge.inference import _device  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT.parent / "final_dataset"
OUT = ROOT / "runs/twin_refiner"
STACKS = ("comp", "uncomp", "pristine")
T_MAX, DEPTHS = 24, (1, 2, 4, 8, 16)
NOISE = {}            # per-acquisition grain strength, chosen on validation slices (runs/<gan>/noise.json)
NC = 3 + len(DEPTHS) + len(STACKS) + 1


def slide(stack):
    return json.loads((ROOT / f"runs/twin_{stack}/calibration.json").read_text())["params"]["streak"]


def ray_condition(cls, z, s, stack_id, rng=None):
    """Condition maps (NC, H, W) for cut plane z of a class volume (0 solid, 1 pore, 2 outside)."""
    Z, H, W = cls.shape
    rows = np.arange(H)
    solid = cls == 0
    depth = np.full((H, W), np.inf, np.float32)
    behind = {}
    for t in range(0, T_MAX + 1):
        if z + t >= Z:
            break
        r = np.clip(np.round(rows + s * t).astype(int), 0, H - 1)
        sl = solid[z + t][r]
        depth[sl & np.isinf(depth)] = t
        if t in DEPTHS:
            behind[t] = sl
    ch = [cls[z] == 1, cls[z] == 2, np.where(np.isinf(depth), 1.0, depth / T_MAX)]
    ch += [behind.get(t, np.zeros((H, W), bool)) for t in DEPTHS]
    ch += [np.full((H, W), float(i == stack_id)) for i in range(len(STACKS))]
    ch += [(rng or np.random.default_rng()).standard_normal((H, W))]
    return np.stack(ch).astype(np.float32)


def load_pairs(stack):
    """Class volume over labelled slices (missing slices take the nearest label) and the real images."""
    images = tifffile.imread(FINAL / stack / "images.tif")
    files = {int(f.stem.split("_")[1]): f for f in (FINAL / stack / "masks").glob("slice_*.png")}
    zs = sorted(files)
    m = {z: np.array(Image.open(files[z])) for z in zs}
    cls = np.stack([np.select([m[q] == 255, m[q] == 0], [0, 1], 2).astype(np.uint8)
                    for q in [min(zs, key=lambda u: abs(u - t)) for t in range(zs[0], zs[-1] + 1)]])
    return cls, images[zs[0]:zs[-1] + 1], json.loads((FINAL / stack / "split.json").read_text())


class PairCrops(torch.utils.data.Dataset):
    def __init__(self, frames_by_stack, crop=256, n=10 ** 6, seed=0):
        self.items, self.crop, self.n, self.rng = [], crop, n, np.random.default_rng(seed)
        self.data = {}
        for si, st in enumerate(STACKS):
            cls, img, _ = load_pairs(st)
            self.data[st] = (cls, img, slide(st), si)
            self.items += [(st, z) for z in frames_by_stack[st]]
        self.cond = {}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        st, z = self.items[self.rng.integers(len(self.items))]
        cls, img, s, si = self.data[st]
        if (st, z) not in self.cond:
            self.cond[(st, z)] = ray_condition(cls, z, s, si)[:-1].astype(np.float16)   # cache without noise
        c = self.cond[(st, z)]
        H, W = c.shape[1:]
        y0, x0 = self.rng.integers(0, H - self.crop + 1), self.rng.integers(0, W - self.crop + 1)
        c = c[:, y0:y0 + self.crop, x0:x0 + self.crop].astype(np.float32)
        c = np.concatenate([c, self.rng.standard_normal((1, self.crop, self.crop)).astype(np.float32)])
        x = img[z, y0:y0 + self.crop, x0:x0 + self.crop].astype(np.float32) / 127.5 - 1
        if self.rng.random() < 0.5:                       # x flip only: y carries the slide direction
            c, x = c[:, :, ::-1].copy(), x[:, ::-1].copy()
        return torch.from_numpy(c), torch.from_numpy(x[None])


def generator():
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=NC, classes=1)


class PatchD(nn.Module):
    def __init__(self, cin=NC + 1, ch=64):
        super().__init__()
        def blk(a, b, stride, norm=True):
            return [nn.Conv2d(a, b, 4, stride, 1)] + ([nn.InstanceNorm2d(b)] if norm else []) + [nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*blk(cin, ch, 2, False), *blk(ch, 2 * ch, 2), *blk(2 * ch, 4 * ch, 2),
                                 *blk(4 * ch, 8 * ch, 1), nn.Conv2d(8 * ch, 1, 4, 1, 1))

    def forward(self, c, x):
        return self.net(torch.cat([c, x], 1))


# ---------------------------------------------------------------- v2: realism to the eye (pix2pixHD-style)
N_EXTRA = 2          # extra noise maps for v2: fine correlated grain, per-column (curtaining) noise


def extra_noise(b, h, w, dev):
    g = torch.randn(b, 1, h, w, device=dev)
    g = F.avg_pool2d(F.pad(g, (1, 1, 1, 1), mode="reflect"), 3, 1)                       # correlated grain
    col = torch.randn(b, 1, 1, w, device=dev).expand(b, 1, h, w)                          # constant down each column
    return torch.cat([g * 3, col], 1)


def generator_v2():
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=NC + N_EXTRA, classes=1)


class PatchD2(nn.Module):
    """Spectral-norm PatchGAN that also returns its intermediate features (for feature matching)."""

    def __init__(self, cin, ch=64, n=4):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers, c = [], cin
        for i in range(n):
            o = ch * 2 ** min(i, 3)
            layers.append(nn.Sequential(sn(nn.Conv2d(c, o, 4, 2 if i < n - 1 else 1, 1)), nn.LeakyReLU(0.2, True)))
            c = o
        self.layers = nn.ModuleList(layers)
        self.out = sn(nn.Conv2d(c, 1, 4, 1, 1))

    def forward(self, c, x):
        h, feats = torch.cat([c, x], 1), []
        for l in self.layers:
            h = l(h); feats.append(h)
        return self.out(h), feats


class MultiD(nn.Module):
    def __init__(self, cin, scales=2):
        super().__init__()
        self.ds = nn.ModuleList([PatchD2(cin) for _ in range(scales)])

    def forward(self, c, x):
        out = []
        for i, d in enumerate(self.ds):
            if i:
                c, x = F.avg_pool2d(c, 2), F.avg_pool2d(x, 2)
            out.append(d(c, x))
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        f = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for q in f.parameters():
            q.requires_grad_(False)
        self.slices = nn.ModuleList([f[:4], f[4:9], f[9:16], f[16:23]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, a, b):
        a = ((a + 1) / 2).repeat(1, 3, 1, 1); b = ((b + 1) / 2).repeat(1, 3, 1, 1)
        a, b = (a - self.mean) / self.std, (b - self.mean) / self.std
        loss = 0.0
        for s in self.slices:
            a, b = s(a), s(b)
            loss = loss + (a - b).abs().mean()
        return loss


def train_v2(a):
    dev = _device(None)
    torch.manual_seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keep = a.stacks.split(",")
    frames = {st: (load_pairs(st)[2]["train"] if st in keep else []) for st in STACKS}
    dl = torch.utils.data.DataLoader(PairCrops(frames, a.crop), batch_size=a.bs, num_workers=0)
    G, D, V = generator_v2().to(dev), MultiD(NC + N_EXTRA + 1).to(dev), VGGLoss().to(dev)
    start = 0
    if a.init_from:                                          # fine-tune another run's generator (fresh discriminator)
        G.load_state_dict(torch.load(a.init_from, map_location=dev, weights_only=False)["G"])
        print("generator initialised from", a.init_from, "stacks", keep, flush=True)
    if a.resume and (out / "G.pt").exists():                 # generator resumes; discriminator restarts
        ck = torch.load(out / "G.pt", map_location=dev, weights_only=False)
        G.load_state_dict(ck["G"]); start = ck["iter"]
        print(f"resumed generator from iteration {start}", flush=True)
    og = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    lam = lambda i: 1.0 if i + start < a.iters // 2 else 2 * (1 - (i + start) / a.iters) + 1e-3
    sg = torch.optim.lr_scheduler.LambdaLR(og, lam)
    sd = torch.optim.lr_scheduler.LambdaLR(od, lam)
    t0 = time.time()
    log = json.loads((out / "train_log.json").read_text()) if start and (out / "train_log.json").exists() else []
    for it, (c, x) in enumerate(dl, start + 1):
        c, x = c.to(dev), x.to(dev)
        c = torch.cat([c, extra_noise(c.shape[0], c.shape[2], c.shape[3], dev)], 1)
        fake = torch.tanh(G(c))
        # discriminator, hinge loss
        ld = sum(F.relu(1 - r).mean() + F.relu(1 + f).mean()
                 for (r, _), (f, _) in zip(D(c, x), D(c, fake.detach())))
        od.zero_grad(set_to_none=True); ld.backward(); od.step()
        # generator: adversarial + feature matching + perceptual + L1
        df = D(c, fake)
        with torch.no_grad():
            dr = D(c, x)
        adv = sum(-f.mean() for f, _ in df)
        fm = sum((hf - hr).abs().mean() for (_, hfs), (_, hrs) in zip(df, dr) for hf, hr in zip(hfs, hrs))
        vg = V(fake, x)
        l1 = (fake - x).abs().mean()
        lg = adv + a.fm * fm + a.vgg * vg + a.lam * l1
        og.zero_grad(set_to_none=True); lg.backward(); og.step()
        sg.step(); sd.step()
        if it % 100 == 0:
            log.append({"iter": it, "d": ld.item(), "adv": adv.item(), "fm": fm.item(), "vgg": vg.item(), "l1": l1.item()})
            print(f"it {it} D {ld.item():.3f} adv {adv.item():.3f} fm {fm.item():.3f} vgg {vg.item():.3f} L1 {l1.item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if it % 1000 == 0 or it == a.iters:
            torch.save({"G": G.state_dict(), "iter": it, "nc": NC + N_EXTRA, "v2": True}, out / "G.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
        if it >= a.iters:
            break


def train(a):
    dev = _device(None)
    torch.manual_seed(0)
    frames = {}
    for st in STACKS:
        _, _, sp = load_pairs(st)
        frames[st] = [z for z in sp["train"]]
    dl = torch.utils.data.DataLoader(PairCrops(frames, a.crop), batch_size=a.bs, num_workers=0)
    G, D = generator().to(dev), PatchD().to(dev)
    og = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    OUT.mkdir(parents=True, exist_ok=True)
    t0, log = time.time(), []
    for it, (c, x) in enumerate(dl, 1):
        c, x = c.to(dev), x.to(dev)
        fake = torch.tanh(G(c))
        # discriminator (LSGAN)
        ld = 0.5 * ((D(c, x) - 1).pow(2).mean() + D(c, fake.detach()).pow(2).mean())
        od.zero_grad(set_to_none=True); ld.backward(); od.step()
        # generator
        l1 = (fake - x).abs().mean()
        lg = (D(c, fake) - 1).pow(2).mean() + a.lam * l1
        og.zero_grad(set_to_none=True); lg.backward(); og.step()
        if it % 100 == 0:
            log.append({"iter": it, "d": ld.item(), "g_adv": lg.item() - a.lam * l1.item(), "l1": l1.item()})
            print(f"it {it} D {ld.item():.3f} G_adv {log[-1]['g_adv']:.3f} L1 {l1.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if it % 1000 == 0 or it == a.iters:
            torch.save({"G": G.state_dict(), "iter": it, "nc": NC}, OUT / "G.pt")
        if it >= a.iters:
            break
    (OUT / "train_log.json").write_text(json.dumps(log, indent=1))


def load_G(dev, run=None):
    """Generator of a run folder (default OUT); also loads that run's per-acquisition grain strengths."""
    run = Path(run) if run else OUT
    if (run / "noise.json").exists():
        NOISE.update(json.loads((run / "noise.json").read_text()))
    ck = torch.load(run / "G.pt", map_location=dev, weights_only=False)
    G = (generator_v2() if ck.get("v2") else generator()).to(dev)
    G.load_state_dict(ck["G"])
    G.v2 = bool(ck.get("v2"))
    G.eval()
    G.per_stack = {}
    if (run / "per_stack.json").exists():                  # e.g. {"pristine": "runs/twin_refiner_v3_pristine"}
        for st, other in json.loads((run / "per_stack.json").read_text()).items():
            o = Path(other) if Path(other).is_absolute() else ROOT / other
            ck2 = torch.load(o / "G.pt", map_location=dev, weights_only=False)
            g2 = generator_v2().to(dev); g2.load_state_dict(ck2["G"]); g2.v2 = True; g2.eval(); g2.per_stack = {}
            G.per_stack[st] = g2
            print(f"{st}: generator from {o}", flush=True)
    return G


def for_stack(G, st):
    return getattr(G, "per_stack", {}).get(st, G)


@torch.no_grad()
def refine(G, cond, dev, tile=512, noise=1.0):
    """(NC, H, W) condition -> uint8 image. ``noise`` scales the generator's noise inputs (grain strength)."""
    _, H, W = cond.shape
    cond = cond.copy(); cond[-1] *= noise
    ph, pw = (-H) % 32, (-W) % 32
    c = torch.from_numpy(np.pad(cond, ((0, 0), (0, ph), (0, pw)), mode="reflect"))[None].to(dev)
    if getattr(G, "v2", False):
        c = torch.cat([c, noise * extra_noise(1, c.shape[2], c.shape[3], dev)], 1)
    y = torch.tanh(G(c))[0, 0, :H, :W].cpu().numpy()
    return np.clip(np.round((y + 1) * 127.5), 0, 255).astype(np.uint8)


def gan_volume(G, solid, stack, dev, rng, noise=None):
    """GAN-twin images of every renderable cut plane of a solid volume (same planes as simulate.render_stack)."""
    si, s = STACKS.index(stack), slide(stack)
    nz = NOISE.get(stack, 1.0) if noise is None else noise
    cls = np.where(solid, 0, 1).astype(np.uint8)
    g = for_stack(G, stack)
    return np.stack([refine(g, ray_condition(cls, z, s, si, rng), dev, noise=nz) for z in range(solid.shape[0] - T_MAX)])


def stats(img, cls):
    pct = [10, 50, 90]
    gy, gx = np.gradient(img.astype(np.float32))
    f = np.abs(np.fft.rfft2(img.astype(np.float32) - img.mean())) ** 2
    fy = np.fft.fftfreq(img.shape[0])[:, None]; fx = np.fft.rfftfreq(img.shape[1])[None]
    rr = np.sqrt(fy ** 2 + fx ** 2)
    hi = float(f[(rr > 0.15)].sum() / f[(rr > 0.01)].sum())          # share of fine-detail power
    return {"pore_pct": np.percentile(img[cls == 1], pct).tolist(), "solid_pct": np.percentile(img[cls == 0], pct).tolist(),
            "grad_mean": float(np.hypot(gx, gy)[cls != 2].mean()), "fine_power_share": hi}


def evaluate(a):
    dev = _device(None)
    G = load_G(dev)
    res, rows = {}, []
    for si, st in enumerate(STACKS):
        cls, img, sp = load_pairs(st)
        s = slide(st)
        c = json.loads((ROOT / f"runs/twin_{st}/calibration.json").read_text())["params"]
        p = SIM.Imaging(**{k: v for k, v in c.items() if k in SIM.Imaging.__dataclass_fields__})
        rng = np.random.default_rng(1)
        held = [z for z in sp["val"] + sp["test"] if z + 8 < len(cls)]
        agg = {"real": [], "gan": [], "physics": []}
        for z in held:
            cond = ray_condition(cls, z, s, si, rng)
            gan = refine(for_stack(G, st), cond, dev)
            phys, _ = SIM.render_slice(cls == 0, z, p, rng)
            for k, im in (("real", img[z]), ("gan", gan), ("physics", phys)):
                agg[k].append(stats(im, cls[z]))
            if z == held[len(held) // 2]:
                rr = np.where((cls[z] != 2).any(1))[0]                          # centre the crop on the layer
                yc = int(rr.mean()) if len(rr) else cls.shape[1] // 2
                y0 = int(np.clip(yc - 100, 0, max(cls.shape[1] - 200, 0)))
                ys = slice(y0, y0 + 200); xs = slice(cls.shape[2] // 2 - 200, cls.shape[2] // 2 + 200)
                sep = np.full((img[z][ys, xs].shape[0], 6), 255, np.uint8)
                rows.append(np.hstack([img[z][ys, xs], sep, gan[ys, xs], sep, phys[ys, xs], sep,
                                       np.where(cls[z][ys, xs] == 1, 0, np.where(cls[z][ys, xs] == 0, 255, 128)).astype(np.uint8)]))
        res[st] = {k: {m: np.mean([d[m] for d in v], 0).tolist() for m in v[0]} for k, v in agg.items()}
        res[st]["held_out_slices"] = held
        print(st, json.dumps({k: {m: np.round(x, 3).tolist() for m, x in v.items()} for k, v in res[st].items() if k != "held_out_slices"}), flush=True)
    (OUT / "eval.json").write_text(json.dumps(res, indent=1))
    w = max(r.shape[1] for r in rows)
    panel = np.vstack(sum([[np.pad(r, ((0, 0), (0, w - r.shape[1])), constant_values=255), np.full((6, w), 255, np.uint8)] for r in rows], [])[:-1])
    Image.fromarray(panel).save(OUT / "panel.png")
    print("panel ->", OUT / "panel.png")


def downstream(a):
    """Task-level realism: a model trained on real data should score the same on twin images of a structure
    as on the real images of that structure. Held-out slices 87-117; contexts 84-117 are generated."""
    from eponge.evaluate import evaluate_frames
    from eponge.inference import segment_stack
    from eponge.models import load_checkpoint
    dev = _device(None)
    G = load_G(dev)
    model, ck = load_checkpoint(ROOT.parent / f"runs/final_models/{a.model}/best.pt", dev)
    k = ck.get("k", 3)
    res = {}
    for si, st in enumerate(STACKS):
        if st not in a.stacks.split(","):
            continue
        cls, img, sp = load_pairs(st)
        s = slide(st)
        c = json.loads((ROOT / f"runs/twin_{st}/calibration.json").read_text())["params"]
        p = SIM.Imaging(**{kk: v for kk, v in c.items() if kk in SIM.Imaging.__dataclass_fields__})
        rng = np.random.default_rng(3)
        held = [z for z in (sp["val"] + sp["test"] if a.split == "all" else sp[a.split]) if z < len(cls)]
        nz = NOISE.get(st, 1.0) if a.noise is None else a.noise
        gen, phys = img[:len(cls)].copy(), img[:len(cls)].copy()
        for z in range(min(held) - k, len(cls)):
            gen[z] = refine(for_stack(G, st), ray_condition(cls, z, s, si, rng), dev, noise=nz)
            phys[z] = SIM.render_slice(cls == 0, z, p, rng)[0]
        gts = {z: np.select([cls[z] == 0, cls[z] == 1], [0, 1], 2).astype(np.uint8) for z in held}
        res[st] = {}
        for name, arr in (("real", img[:len(cls)]), ("gan_twin", gen), ("physics_twin", phys)):
            pr, _ = segment_stack(model, arr, k, dev, tta=False, frames=held)
            m = evaluate_frames(dict(zip(held, pr)), gts)
            res[st][name] = {kk: m[kk] for kk in ("mean_iou", "pore_iou", "porosity_mae", "pred_porosity_mean", "gt_porosity_mean")}
        print(st, {n: round(v["mean_iou"], 4) for n, v in res[st].items()}, flush=True)
        res[st]["noise"] = nz
    tag = f"_{a.split}" + (f"_noise{a.noise}" if a.noise is not None else "")
    (OUT / f"downstream_{a.model}{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "train_v2", "eval", "downstream"])
    ap.add_argument("--out", default=str(OUT), help="run folder (G.pt, eval outputs)")
    ap.add_argument("--fm", type=float, default=10.0)
    ap.add_argument("--vgg", type=float, default=5.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stacks", default="comp,uncomp,pristine")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--split", default="all", choices=["all", "val", "test"])
    ap.add_argument("--noise", type=float, default=None, help="grain strength for every stack (default: noise.json or 1)")
    ap.add_argument("--model", default="millnet_sa2_long_k3_s0")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lam", type=float, default=20.0)
    a = ap.parse_args()
    OUT = Path(a.out)
    if (OUT / "noise.json").exists():
        NOISE.update(json.loads((OUT / "noise.json").read_text()))
    if a.cmd == "train_v2" and a.lam == 20.0:
        a.lam = 5.0
    {"train": train, "train_v2": train_v2, "eval": evaluate, "downstream": downstream}[a.cmd](a)
