"""
cut_train.py — learn the simulated -> real appearance map with Contrastive Unpaired Translation
(Park et al. 2020, CUT), including the single-image variant (SinCUT) for the case where only one
real micrograph exists (your image 2).

Why this and not a diffusion model or a plain GAN:
  * unpaired: we have no (simulated, real) pixel pairs, only two distributions;
  * PatchNCE keeps every output patch maximally informative about the input patch at the same
    location -> geometry is preserved, so the LABEL of the simulated image stays valid;
  * an optional mask-consistency term (a frozen segmenter applied to the output must reproduce the
    input mask) makes that explicit;
  * SinCUT trains on random crops of ONE high-resolution image with patch discriminators, which is
    exactly our data situation; with thousands of real slices (the catalyst-layer stacks) use
    normal CUT with the whole real set as the target domain.
Acceptance test: `c2st` from eponge_synth.patch_transfer on HELD-OUT real crops (never used for
training) — the target is the real-vs-real control score, not 0.5.

    python cut_train.py --synth datasets/felt/stacks --real samples/real_uploads/flake_felt.png --single_image \
        --out runs/sincut_felt --iters 20000
    python cut_train.py --synth datasets/v2/stacks --real real_slices/ --out runs/cut_cl --iters 60000

Not executed in the authoring sandbox (no torch there).
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_gray(path):
    return np.array(Image.open(path).convert("L"))


class CropSampler:
    """Random crops (and x-flips only: the SEM frame is canonical) from a list of grayscale images."""

    def __init__(self, images, crop=128):
        self.images, self.crop = images, crop

    def batch(self, n):
        out = []
        for _ in range(n):
            im = random.choice(self.images)
            H, W = im.shape
            y, x = random.randint(0, H - self.crop), random.randint(0, W - self.crop)
            c = im[y:y + self.crop, x:x + self.crop].astype(np.float32) / 127.5 - 1.0
            if random.random() < 0.5:
                c = c[:, ::-1]
            out.append(np.ascontiguousarray(c)[None])
        return torch.from_numpy(np.stack(out))


def load_synth(root):
    ims = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        p = os.path.join(d, "images.npy")
        if os.path.exists(p):
            st = np.load(p, mmap_mode="r")
            ims += [np.array(st[i]) for i in range(0, st.shape[0], max(1, st.shape[0] // 16))]
    if not ims:  # a folder of PNGs also works
        ims = [load_gray(p) for p in sorted(glob.glob(os.path.join(root, "*.png")))]
    return ims


# --------------------------------------------------------------------------------------
# networks (ResNet generator with exposed encoder features, PatchGAN discriminator, NCE heads)
# --------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
                               nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))

    def forward(self, x):
        return x + self.b(x)


class Generator(nn.Module):
    def __init__(self, ngf=64, n_blocks=9):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(1, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(ngf, ngf * 2, 3, 2, 1), nn.InstanceNorm2d(ngf * 2), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1), nn.InstanceNorm2d(ngf * 4), nn.ReLU(True)),
            nn.Sequential(*[ResBlock(ngf * 4) for _ in range(n_blocks // 2)]),
            nn.Sequential(*[ResBlock(ngf * 4) for _ in range(n_blocks - n_blocks // 2)]),
        ])
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, 7), nn.Tanh())

    def forward(self, x, return_feats=False):
        feats = [x]
        for layer in self.enc:
            x = layer(x)
            feats.append(x)
        return (self.dec(x), feats) if return_feats else self.dec(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1), nn.InstanceNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1))

    def forward(self, x):
        return self.net(x)


class NCEHeads(nn.Module):
    """One 2-layer MLP per sampled feature layer, projecting patch features to a unit sphere."""

    def __init__(self, channels, dim=256):
        super().__init__()
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(c, dim), nn.ReLU(True), nn.Linear(dim, dim)) for c in channels])

    def forward(self, feats, n_patches=256, ids=None):
        out, new_ids = [], []
        for i, f in enumerate(feats):
            B, C, H, W = f.shape
            flat = f.permute(0, 2, 3, 1).reshape(B, H * W, C)
            idx = ids[i] if ids is not None else torch.randperm(H * W, device=f.device)[:n_patches]
            z = self.mlps[i](flat[:, idx].reshape(-1, C))
            out.append(F.normalize(z, dim=1)); new_ids.append(idx)
        return out, new_ids


def patch_nce(q, k, tau=0.07):
    """q from the output, k from the input at the same locations; other locations are negatives."""
    B = q.shape[0]
    logits = q @ k.t() / tau
    labels = torch.arange(B, device=q.device)
    return F.cross_entropy(logits, labels)


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synth = load_synth(args.synth)
    real = [load_gray(args.real)] if os.path.isfile(args.real) else [load_gray(p) for p in sorted(glob.glob(os.path.join(args.real, "*.png")))]
    if args.single_image:   # SinCUT: hold out a vertical strip of the single real image for the acceptance test
        im = real[0]; cut = int(im.shape[1] * 0.6)
        real, holdout = [im[:, :cut]], im[:, cut:]
    else:
        n_hold = max(1, len(real) // 5); holdout, real = real[-n_hold:], real[:-n_hold]
    S, R = CropSampler(synth, args.crop), CropSampler(real, args.crop)
    G, D = Generator().to(dev), PatchDiscriminator().to(dev)
    with torch.no_grad():
        _, feats = G(S.batch(1).to(dev), return_feats=True)
    layers = [0, 2, 3, 4, 5]
    heads = NCEHeads([feats[l].shape[1] for l in layers]).to(dev)
    opt_g = torch.optim.Adam(list(G.parameters()) + list(heads.parameters()), 2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    seg = None
    if args.segmenter:   # optional mask-consistency: frozen 2.5D/2D segmenter must agree on input vs output
        from eponge_synth.training import build_model
        ck = torch.load(args.segmenter, map_location=dev); seg = build_model(ck["args"]["arch"], 1).to(dev)
        seg.load_state_dict(ck["model"]); seg.eval()
    os.makedirs(args.out, exist_ok=True)
    for it in range(1, args.iters + 1):
        xs, xr = S.batch(args.batch).to(dev), R.batch(args.batch).to(dev)
        # --- D step (LSGAN)
        with torch.no_grad():
            fake = G(xs)
        loss_d = 0.5 * (((D(xr) - 1) ** 2).mean() + (D(fake) ** 2).mean())
        opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()
        # --- G step: adversarial + PatchNCE(x -> G(x)) + identity PatchNCE(y -> G(y))
        fake, f_fake = G(xs, return_feats=True)
        _, f_src = G(xs, return_feats=True)
        q, ids = heads([f_fake[l] for l in layers], args.n_patches)
        k, _ = heads([f_src[l] for l in layers], args.n_patches, ids)
        loss_nce = sum(patch_nce(qi, ki) for qi, ki in zip(q, k)) / len(q)
        idt, f_idt = G(xr, return_feats=True)
        _, f_real = G(xr, return_feats=True)
        q2, ids2 = heads([f_idt[l] for l in layers], args.n_patches)
        k2, _ = heads([f_real[l] for l in layers], args.n_patches, ids2)
        loss_idt = sum(patch_nce(qi, ki) for qi, ki in zip(q2, k2)) / len(q2)
        loss_adv = ((D(fake) - 1) ** 2).mean()
        loss_g = loss_adv + args.lambda_nce * (loss_nce + loss_idt) / 2
        if seg is not None:
            with torch.no_grad():
                ref_mask = seg((xs + 1) / 2 * 0 + ((xs + 1) / 2 - 0.35) / 0.25).argmax(1)
            loss_g = loss_g + args.lambda_mask * F.cross_entropy(seg(((fake + 1) / 2 - 0.35) / 0.25), ref_mask)
        opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()
        if it % 200 == 0:
            print(f"it {it}: D {loss_d.item():.3f} adv {loss_adv.item():.3f} nce {loss_nce.item():.3f} idt {loss_idt.item():.3f}")
        if it % 2000 == 0 or it == args.iters:
            torch.save(G.state_dict(), os.path.join(args.out, "G.pt"))
            with torch.no_grad():
                x = S.batch(4).to(dev); y = G(x)
            grid = torch.cat([x, y], 3)[:, 0].cpu().numpy()
            Image.fromarray(np.clip((np.concatenate(grid, 0) + 1) * 127.5, 0, 255).astype(np.uint8)).save(os.path.join(args.out, f"sample_{it}.png"))
    # --- acceptance test on held-out real data
    from eponge_synth.patch_transfer import c2st
    with torch.no_grad():
        x = S.batch(8).to(dev); y = ((G(x)[:, 0].cpu().numpy() + 1) * 127.5).astype(np.uint8)
    hold = holdout[0] if isinstance(holdout, list) else holdout
    res = {"translated": c2st(hold, np.concatenate(list(y), 0), patch=11), "control": c2st(hold, real[0], patch=11)}
    print("C2ST (texture 11px):", res)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synth", required=True); p.add_argument("--real", required=True); p.add_argument("--out", default="runs/cut")
    p.add_argument("--single_image", action="store_true"); p.add_argument("--crop", type=int, default=128)
    p.add_argument("--batch", type=int, default=4); p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--n_patches", type=int, default=256); p.add_argument("--lambda_nce", type=float, default=1.0)
    p.add_argument("--segmenter", default=None); p.add_argument("--lambda_mask", type=float, default=0.5)
    train(p.parse_args())
