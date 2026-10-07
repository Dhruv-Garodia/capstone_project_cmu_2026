"""Generate IrO2 catalyst-layer geometry with SliceGAN (Kench & Cooper 2021) for the digital twin.

The layer is anisotropic: y (through the layer) differs from x and z (both in-plane). SliceGAN's
three-plane mode takes one 2D training image per orthogonal plane of the generated volume (Z, Y, X):

  plane normal to Z -> (y, x): real cut faces, the most trustworthy view (no shine-through bias in z)
  plane normal to X -> (z, y): cut faces transposed, using in-plane equivalence of x and z
  plane normal to Y -> (z, x): the in-plane view, taken from the segmented stack

Only slices 0-179 are used, so twin test geometry from slices 250-353 stays independent.
Requires the official SliceGAN code in external/SliceGAN (github.com/stke9/SliceGAN).

    python eponge/scripts/slicegan_twin.py prepare
    python eponge/scripts/slicegan_twin.py train --minutes 60
    python eponge/scripts/slicegan_twin.py generate --n 14
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SG = ROOT.parent / "external/SliceGAN"
OUT = ROOT / "runs/slicegan"
sys.path.insert(0, str(SG))


def prepare():
    from eponge.metrics import largest_inner_rectangle
    cls = np.load(ROOT / "runs/twin/comp_full_millnet_classes.npy")[:180]
    OUT.mkdir(parents=True, exist_ok=True)
    faces = []
    for z in range(0, 180, 4):                                   # cut faces, interior of the layer only
        rs, cs = largest_inner_rectangle(cls[z] != 2)
        if rs.stop - rs.start >= 96:
            faces.append(cls[z][rs.start:rs.start + 96, cs.start:cs.start + min(1024, cs.stop - cs.start)])
    w = min(f.shape[1] for f in faces)
    face_img = np.concatenate([f[:, :w] for f in faces], 0)       # rows y, cols x
    rs, cs = largest_inner_rectangle((cls != 2).all(0))
    inplane = np.concatenate([cls[:, y, cs] for y in range(rs.start, rs.stop, 6)], 0)   # rows z, cols x
    imgs = {"plane_z_yx.png": face_img, "plane_y_zx.png": inplane, "plane_x_zy.png": face_img.T}
    for name, im in imgs.items():
        Image.fromarray(np.where(im == 1, 0, 255).astype(np.uint8)).save(OUT / name)   # pore black, solid white
        print(name, im.shape, "porosity", round(float((im == 1).mean()), 3))


def nets():
    from slicegan import networks
    img_channels, z_channels, lays, laysd = 2, 32, 5, 6
    dk, gk = [4] * laysd, [4] * lays
    ds, gs = [2] * laysd, [2] * lays
    # generator 2x narrower than SliceGAN's default (1024 -> 512 first layer): ample for a two-phase
    # structure and ~4x faster on an Apple GPU, where 3D transposed convolutions are slow
    df, gf = [img_channels, 64, 128, 256, 512, 1], [z_channels, 512, 256, 128, 32, img_channels]
    dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]
    return networks.slicegan_rc_nets(str(OUT / "iro2"), True, "nphase", dk, ds, df, dp, gk, gs, gf, gp), z_channels


def train(minutes):
    """SliceGAN's WGAN-GP loop (slicegan/model.py) on the Apple GPU with a time budget."""
    import torch
    from slicegan import preprocessing, util
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    (Disc, Gen), nz = nets()
    # SliceGAN compares dataset[0] with fake slices normal to volume axis 2 (our Z), whose image axes
    # are (Y, X); dataset[1] with slices normal to axis 3 (our Y), axes (Z, X); dataset[2] with slices
    # normal to axis 4 (our X), axes (Z, Y). The file names state plane and axis order.
    data = [str(OUT / n) for n in ("plane_z_yx.png", "plane_y_zx.png", "plane_x_zy.png")]
    l, bs, critic, lam, lz = 64, 8, 5, 10, 4
    ns = 64   # fake slices per discriminator / generator step (random subset of the 512); speeds training ~8x
    dsets = preprocessing.batch(data, "png", l, 1)
    loaders = [torch.utils.data.DataLoader(d, batch_size=bs, shuffle=True) for d in dsets]
    netG = Gen().to(dev)
    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.9, 0.99))
    netDs = [Disc().to(dev) for _ in range(3)]
    optDs = [torch.optim.Adam(d.parameters(), lr=1e-4, betas=(0.9, 0.99)) for d in netDs]
    t0, it = time.time(), 0
    while time.time() - t0 < minutes * 60:
        for batch in zip(*loaders):
            it += 1
            fake = netG(torch.randn(bs, nz, lz, lz, lz, device=dev)).detach()
            for dim, (netD, opt, real, perm) in enumerate(zip(netDs, optDs, batch, [(0, 2, 1, 3, 4), (0, 3, 1, 2, 4), (0, 4, 1, 2, 3)])):
                netD.zero_grad()
                real = real[0].to(dev)
                fperm = fake.permute(*perm).reshape(l * bs, 2, l, l)
                fperm = fperm[torch.randperm(l * bs, device=dev)[:ns]]
                gp = util.calc_gradient_penalty(netD, real, fperm[:bs], bs, l, dev, lam, 2)
                cost = netD(fperm).mean() - netD(real).view(-1).mean() + gp
                cost.backward(); opt.step()
            if it % critic == 0:
                netG.zero_grad()
                fake = netG(torch.randn(bs, nz, lz, lz, lz, device=dev))
                errG = -sum(netD(fake.permute(*perm).reshape(l * bs, 2, l, l)[torch.randperm(l * bs, device=dev)[:ns]]).mean()
                            for netD, perm in zip(netDs, [(0, 2, 1, 3, 4), (0, 3, 1, 2, 4), (0, 4, 1, 2, 3)]))
                errG.backward(); optG.step()
            if it % 50 == 0:
                torch.save(netG.state_dict(), OUT / "iro2_Gen.pt")
                print(f"it {it} ({(time.time()-t0)/60:.1f} min) wasserstein {(-cost).item():.3f}", flush=True)
            if time.time() - t0 > minutes * 60:
                break
    torch.save(netG.state_dict(), OUT / "iro2_Gen.pt")


def generate(n, seed=0):
    """n solid volumes (Z, Y, X) ~ (64, 160, 256) for the twin; saved to runs/slicegan/volumes.npz."""
    import torch
    (Disc, Gen), nz = nets()
    netG = Gen(); netG.load_state_dict(torch.load(OUT / "iro2_Gen.pt", map_location="cpu")); netG.eval()
    torch.manual_seed(seed)
    vols = []
    with torch.no_grad():
        for _ in range(n):
            out = netG(torch.randn(1, nz, 4, 7, 10))[0]           # output 32*(latent-2): 64 x 160 x 256
            solid = (out.argmax(0) == 1).numpy()                # channel order follows sorted phases: 0 black=pore, 1 white=solid
            vols.append(solid)
    np.savez_compressed(OUT / "volumes.npz", *vols)
    print("generated", len(vols), vols[0].shape, "porosity", round(float(1 - np.mean([v.mean() for v in vols])), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--n", type=int, default=14); a = ap.parse_args()
    {"prepare": prepare, "train": lambda: train(a.minutes), "generate": lambda: generate(a.n)}[a.cmd]()
