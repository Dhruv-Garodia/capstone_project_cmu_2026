"""
make_dataset.py — generate a calibrated, domain-randomised synthetic pFIB-SEM dataset.

    python make_dataset.py --out datasets/v2 --n 40 --shape 192 256 256 --families catalyst_layer
    python make_dataset.py --out datasets/felt --n 20 --shape 96 320 320 --families sheet_felt fiber_felt   # 250 nm voxels
    python make_dataset.py --out datasets/quick --n 2 --shape 64 96 96      # smoke test

Each volume -> <out>/stacks/<name>/{images.npy, masks.npy, material.npy, meta.json, preview.png}
Presets (see eponge_synth.microstructure.PRESETS) are drawn at random per volume; appearance parameters
are drawn from SEMParams.random() around the calibrated presets in samples/appearance_calibration.json.
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eponge_synth.microstructure import PRESETS, catalyst_layer
from eponge_synth.felt import sheet_felt
from eponge_synth.sem_render import SEMParams, SEMSimulator

CL_PRESETS = ["pristine", "uncomp", "comp", "mudcrack", "faceted"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--shape", type=int, nargs=3, default=[150, 150, 150]); ap.add_argument("--voxel_nm", type=float, default=6.0)
    ap.add_argument("--families", nargs="+", default=["catalyst_layer"], help="catalyst_layer sheet_felt fiber_felt"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib", default=os.path.join(os.path.dirname(__file__), "samples", "appearance_calibration.json"))
    ap.add_argument("--randomize_beam", action="store_true", help="random +/-y shine-through direction (enables y-flip aug)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    calib = json.load(open(a.calib)) if os.path.exists(a.calib) else {}
    os.makedirs(os.path.join(a.out, "stacks"), exist_ok=True)
    index = []
    for i in range(a.n):
        fam = a.families[i % len(a.families)]
        seed = int(rng.integers(1e9)); t = time.time()
        if fam == "catalyst_layer":
            preset = CL_PRESETS[int(rng.integers(len(CL_PRESETS)))]
            kw = dict(PRESETS[preset]); kw["porosity"] = float(np.clip(kw["porosity"] + rng.normal(0, 0.03), 0.3, 0.7))
            m = catalyst_layer(shape=a.shape, voxel_nm=a.voxel_nm, seed=seed, **kw)
            base = SEMParams(**calib.get("mudcrack" if preset == "mudcrack" else "faceted", {}).get("params", {})) if calib else SEMParams()
            vox = a.voxel_nm
        elif fam in ("sheet_felt", "fiber_felt"):
            preset = fam; vox = 250.0
            g = dict(calib.get("felt", {}).get("generator", {}))
            if fam == "fiber_felt":
                g.update(mode="fiber", length_um=(float(rng.uniform(15, 35)), 0.4), thickness_um=(float(rng.uniform(0.8, 2.0)), 0.2),
                         crumple=0.15, tilt_spread_deg=float(rng.uniform(15, 35)))
            else:
                g.update(length_um=(float(rng.uniform(6, 14)), 0.6), crumple=float(rng.uniform(0.25, 0.5)),
                         tilt_spread_deg=float(rng.uniform(35, 60)), compaction=float(rng.uniform(0.6, 0.85)))
            fc = calib.get("felt", {}); vox = float(fc.get("voxel_um", 0.25)) * 1000.0
            m = sheet_felt(shape=tuple(fc.get("shape", a.shape)), voxel_um=vox / 1000.0, seed=seed, n_sheets=40000, **g)
            base = SEMParams(**calib.get("felt", {}).get("params", {})) if "felt" in calib else SEMParams(tilt_deg=90, max_depth_nm=20000, atten_nm=3000)
        else:
            raise ValueError(f"unknown family {fam}; use catalyst_layer, sheet_felt, fiber_felt")
        p = SEMParams.random(rng, base)
        if not a.randomize_beam:
            p.beam_sign = 1
        sim = SEMSimulator(m.material, vox, p, seed=seed, edge_map=getattr(m, 'edges', None), normal_sigma=1.6 if hasattr(m, 'edges') else 1.0)
        imgs, masks = sim.stack()
        name = f"{i:04d}_{fam}_{preset}"
        d = os.path.join(a.out, "stacks", name); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "images.npy"), imgs); np.save(os.path.join(d, "masks.npy"), masks)
        np.save(os.path.join(d, "material.npy"), m.material)
        meta = dict(family=fam, preset=preset, voxel_nm=vox, sem_params=p.to_dict(), beam_sign_random=a.randomize_beam,
                    generator_meta=m.meta, realized_porosity=m.porosity)
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2, default=float)
        z = imgs.shape[0] // 2
        prev = np.concatenate([imgs[z], np.full((imgs.shape[1], 4), 255, np.uint8), masks[z].astype(np.uint8) * 255], 1)
        Image.fromarray(prev).save(os.path.join(d, "preview.png"))
        index.append(dict(name=name, family=fam, preset=preset, porosity=m.porosity, seconds=round(time.time() - t, 1)))
        print(f"[{i+1}/{a.n}] {name} porosity={m.porosity:.3f} ({time.time()-t:.1f}s)")
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
