"""Layer-by-layer sheets: [clean structure render | mask | (material map) | calibrated SEM] per depth.
Volumes: generate with eponge_synth (make_preset / sheet_felt) or load saved .npy materials.
Produces gallery/13..16_layers_*.png used by docs/LAYER_BY_LAYER.md."""
import json, numpy as np
from PIL import Image, ImageDraw
from eponge_synth.sem_render import SEMSimulator, SEMParams

def clean(sem):
    s = dict(sem); s.update(poisson_scale=1e6, gauss_sigma=0.0, grain_amp=0.0, charge_amp=0.0, curtain_amp=0.0,
                            drift_gain=0.0, drift_offset=0.0, psf_sigma_px=0.5, gamma=1.0, gain=min(sem.get('gain', 0.6) * 1.35, 0.95))
    return s

def matmap(m):
    return np.array([[0, 0, 0], [190, 190, 190], [90, 90, 90], [255, 255, 255]], np.uint8)[m]

def sheet(out, mat, vox_nm, sem, depths, labels, edge_map=None, nsig=1.0, ds=1, with_material=False, tile=220):
    cols = ['structure (clean render of the cut)', 'ground-truth mask'] + (['material map'] if with_material else []) + ['simulated SEM (calibrated)']
    im = Image.new('L', (tile * len(cols) + 10 * (len(cols) - 1) + 70, (tile + 8) * len(depths) + 30), 255)
    d = ImageDraw.Draw(im)
    for c, t in enumerate(cols):
        d.text((70 + c * (tile + 10), 6), t, fill=0)
    sim_r = SEMSimulator(mat, vox_nm, SEMParams(**sem), seed=2, edge_map=edge_map, normal_sigma=nsig)
    sim_c = SEMSimulator(mat, vox_nm, SEMParams(**clean(sem)), seed=2, edge_map=edge_map, normal_sigma=nsig)
    for r, (z, lab) in enumerate(zip(depths, labels)):
        ir, _ = sim_r.image(z); ic, _ = sim_c.image(z)
        if ds > 1:
            ir = ir.reshape(ir.shape[0] // ds, ds, ir.shape[1] // ds, ds).mean((1, 3)).astype(np.uint8)
            ic = ic.reshape(ic.shape[0] // ds, ds, ic.shape[1] // ds, ds).mean((1, 3)).astype(np.uint8)
        tiles = [ic, (mat[z] > 0).astype(np.uint8) * 255] + ([matmap(mat[z])] if with_material else []) + [ir]
        y0 = 30 + r * (tile + 8); d.text((4, y0 + tile // 2 - 6), f"z={lab}", fill=0)
        for c, a in enumerate(tiles):
            im.paste(Image.fromarray(a).convert('L').resize((tile, tile), Image.BILINEAR), (70 + c * (tile + 10), y0))
    im.save(out); print('saved', out)

if __name__ == "__main__":
    calib = json.load(open('../results/appearance_calibration.json'))
    felt = np.load('felt_material_v12.npy'); fedges = np.load('felt_edges_v12.npy')
    sheet('13_layers_sheet_felt.png', felt, 125.0, calib['felt']['params'], [1, 8, 16, 26, 38, 52],
          ['0.1um', '1.0um', '2.0um', '3.3um', '4.8um', '6.5um'], edge_map=fedges, nsig=1.6, ds=2)
    sheet('15_layers_catalyst_mudcrack.png', np.load('mudcrack_material.npy'), 6.0, calib['mudcrack']['params'],
          [40, 55, 70, 85, 100, 115], ['240nm', '330nm', '420nm', '510nm', '600nm', '690nm'], with_material=True, tile=200)
