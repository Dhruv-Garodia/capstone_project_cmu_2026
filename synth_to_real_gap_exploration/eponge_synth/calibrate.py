"""
Calibrate `catalyst_layer` parameters against the published real-data descriptors
(Ferner et al. 2024, Table 1 + Fig. 8).  Objective per region:

    J = |mu - mu*| + |sigma - sigma*| + |mean_pore - mean*| / mean*

This is a coarse-to-fine grid search (no gradients needed; each evaluation is a few seconds).
When real slices are available, `gap_report.py` re-measures the real stack with the SAME
descriptor code, so the targets can be replaced by measured values with one flag.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eponge_synth.descriptors import FERNER_2024, describe_volume  # noqa: E402
from eponge_synth.microstructure import catalyst_layer  # noqa: E402


def objective(d, tgt):
    return abs(d.lognormal_mu - tgt["lognormal_mu"]) + abs(d.lognormal_sigma - tgt["lognormal_sigma"]) \
        + abs(d.pore_mean_nm - tgt["mean_pore_nm"]) / tgt["mean_pore_nm"]


CLEAN_R = 2   # measurement operator = Ferner 2024 steps 5-6 (3-D open/close, spherical element)


def evaluate(region, shape, seed=0, **kw):
    """Generate, then measure through the paper's clean-up operator. One porosity-correction
    step compensates the porosity shift introduced by that operator (~+0.03)."""
    tgt = FERNER_2024[region]
    p_gen = tgt["porosity"] - 0.03
    m = catalyst_layer(shape=shape, voxel_nm=6.0, porosity=p_gen, dense_inclusions=0, seed=seed, **kw)
    d = describe_volume(m.solid, 6.0, clean_radius=CLEAN_R)
    p_gen += tgt["porosity"] - d.porosity
    m = catalyst_layer(shape=shape, voxel_nm=6.0, porosity=p_gen, dense_inclusions=0, seed=seed, **kw)
    d = describe_volume(m.solid, 6.0, clean_radius=CLEAN_R)
    kw["porosity_gen"] = p_gen
    return objective(d, tgt), d


def search(region, shape=(112, 112, 112), grid=None, log=print):
    tgt = FERNER_2024[region]
    if grid is None:
        base = dict(fine_scale_weight=[0.1], macro_scale_nm=[250], particle_radius_nm=[(26, 0.35)])
        if region == "pristine":
            grid = dict(agglomerate_nm=[85, 95], macro_weight=[0.15, 0.25], **base)
        elif region == "uncomp":
            grid = dict(agglomerate_nm=[110, 130, 150], macro_weight=[0.15, 0.3], **base)
        else:  # comp: PTL-fibre-compressed -> squeezed through-plane
            grid = dict(agglomerate_nm=[70, 90], compression_z=[0.55, 0.7], macro_weight=[0.15], **base)
    keys = list(grid)
    best = None
    for combo in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, combo))
        t = time.time()
        J, d = evaluate(region, shape, **kw)
        kw = dict(kw)
        log(f"[{region}] {kw} -> J={J:.3f} mu={d.lognormal_mu:.3f}({tgt['lognormal_mu']}) "
            f"sig={d.lognormal_sigma:.3f}({tgt['lognormal_sigma']}) mean={d.pore_mean_nm:.0f}({tgt['mean_pore_nm']}) "
            f"max={d.pore_max_nm:.0f} [{time.time()-t:.1f}s]")
        if best is None or J < best[0]:
            best = (J, kw, d)
    return best


if __name__ == "__main__":
    out = {}
    regions = sys.argv[1:] or ["pristine", "uncomp", "comp"]
    for region in regions:
        J, kw, d = search(region)
        out[region] = dict(objective=J, params=kw, realized={k: getattr(d, k) for k in
                           ["porosity", "pore_mean_nm", "pore_std_nm", "pore_max_nm", "lognormal_mu", "lognormal_sigma"]},
                           target=FERNER_2024[region])
        print("BEST", region, json.dumps(out[region], default=float))
    Path("calibration_result.json").write_text(json.dumps(out, indent=2, default=float))
