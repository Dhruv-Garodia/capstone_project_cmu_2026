"""
pFIB-SEM image formation simulator — replaces the Blender/Eevee renderer.

Why a simulator and not a renderer: an SE image of a milled cross-section is NOT a lit photograph.
Its defining artefacts are (i) shine-through: electrons image solid surfaces BELOW the cut plane
through open pores, foreshortened along the 52 deg beam direction (Ferner 2024 removed exactly these
"52 deg streaks"), (ii) edge brightening (SE yield rises at grazing incidence), (iii) material
contrast (IrO2 bright, ionomer dark), (iv) nanoparticle grain, (v) curtaining stripes from FIB
milling, (vi) charging/illumination drift, (vii) beam PSF + Poisson shot noise.
All of these are cheap to model directly on the voxel volume, are calibratable against real images
(see domain_adapt.calibrate_appearance), and produce an image that is pixel-aligned with the exact
cut-plane mask by construction — no camera, no resize, no perspective.

Frame convention (fixed, canonical): volume (Z, Y, X); slices are cut at increasing z; the SEM beam
makes `tilt_deg` with the cut plane and its in-plane component points along +y.  Shine-through
therefore appears displaced toward +y.  Augmentations that flip y or rotate in-plane break this
frame and must not be used unless the renderer is asked to randomise it (`beam_sign`).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi


@dataclass
class SEMParams:
    tilt_deg: float = 52.0          # angle between beam and cut plane (Helios: 52)
    beam_sign: int = 1              # +1: shine-through displaced toward +y, -1: toward -y
    max_depth_nm: float = 260.0     # ray length beyond which shine-through is negligible
    atten_nm: float = 55.0          # exponential attenuation length of the shine-through signal
    bright_catalyst: float = 1.00   # SE yield of the catalyst / fibre phase (relative)
    bright_ionomer: float = 0.45    # ionomer / binder phase (low-Z -> darker)
    bright_dense: float = 1.05      # dense inclusions
    pore_floor: float = 0.04        # signal from unhit pore (deep void)
    lambert_min: float = 0.35       # shading floor for below-plane surfaces facing away from the beam
    secant_weight: float = 0.5      # SE topographic contrast: yield ~ sec(theta) (grazing surfaces/edges BRIGHT)
    secant_max: float = 3.0         # cap on the secant term
    edge_enh: float = 0.35          # SE edge-brightening amplitude at cut-face solid boundaries
    edge_width_px: float = 1.5
    edge_blur_px: float = 0.7       # width of the SE edge highlight when a geometric edge map is supplied (electron escape depth / beam spread)
    grain_amp: float = 0.22         # nanoparticle brightness speckle amplitude (multiplicative)
    grain_sigma_px: float = 1.2     # correlation length of the speckle (~ primary particle radius)
    charge_amp: float = 0.15        # low-frequency multiplicative illumination/charging field
    charge_sigma_px: float = 30.0
    curtain_amp: float = 0.06       # vertical stripe (curtaining) amplitude
    curtain_sigma_px: float = 1.5
    psf_sigma_px: float = 0.8       # beam / detector blur
    poisson_scale: float = 90.0     # "photons" per unit brightness (shot noise)
    gauss_sigma: float = 0.02       # detector read noise (relative)
    gain: float = 0.78              # 8-bit mapping: out = ((I*gain + offset)**gamma)*255
    gamma: float = 1.0              # tone curve (<1 lifts mid-tones, >1 darkens faces relative to highlights)
    occlusion_k: float = 0.0        # ambient occlusion: signal *= exp(-k * amount of solid above the hit point) (plan views of piles)
    occlusion_sigma_px: float = 2.0
    transmittance: float = 0.0      # fraction of the beam signal continuing through a thin solid feature (low-kV thin flakes: ~0.3-0.5; IrO2 at 10 kV: 0)
    detector_elevation_deg: float = 90.0   # 90 = in-lens/TLD (along the beam); <90 = side-mounted Everhart-Thornley: directional shading
    detector_azimuth_deg: float = 45.0     # in-plane direction of the side detector (image frame: 0 = +x, 90 = +y)
    offset: float = 0.06
    drift_gain: float = 0.02        # per-slice random-walk of gain, RELATIVE to gain (slow brightness drift through the stack)
    drift_offset: float = 0.05      # relative to offset

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def random(rng: np.random.Generator, base: Optional["SEMParams"] = None) -> "SEMParams":
        """Domain randomisation: sample every appearance parameter in a plausible range."""
        b = base or SEMParams()
        u = rng.uniform
        return replace(
            b,
            beam_sign=int(rng.choice([-1, 1])),
            atten_nm=b.atten_nm * u(0.5, 2.0), max_depth_nm=b.max_depth_nm * u(0.6, 1.5),   # scale-aware: relative to the base
            bright_ionomer=u(0.3, 0.65), bright_dense=u(0.95, 1.15), pore_floor=u(0.0, 0.10),
            lambert_min=u(0.2, 0.5), secant_weight=u(0.2, 0.8), edge_enh=u(0.1, 0.8), edge_width_px=u(1.0, 2.5),
            grain_amp=u(0.08, 0.55), grain_sigma_px=u(0.8, 2.2),
            charge_amp=u(0.05, 0.3), charge_sigma_px=u(15, 60),
            curtain_amp=u(0.0, 0.12), curtain_sigma_px=u(1.0, 3.0),
            psf_sigma_px=u(0.5, 1.4), poisson_scale=u(40, 200), gauss_sigma=u(0.01, 0.05),
            gain=u(0.25, 1.0), offset=u(0.0, 0.15), gamma=u(0.8, 1.4), transmittance=b.transmittance * u(0.6, 1.4) if b.transmittance > 0 else 0.0,
            detector_elevation_deg=b.detector_elevation_deg if b.detector_elevation_deg >= 89 else u(25, 70), detector_azimuth_deg=u(0, 360),
        )


class SEMSimulator:
    """Simulates the SE image of every cut plane of a material volume."""

    def __init__(self, material: np.ndarray, voxel_nm: float, params: SEMParams = SEMParams(), seed: int = 0,
                 edge_map: Optional[np.ndarray] = None, normal_sigma: float = 1.0):
        self.material = np.asarray(material, dtype=np.uint8)
        self.solid = self.material > 0
        self.voxel_nm = float(voxel_nm)
        self.p = params
        self.rng = np.random.default_rng(seed)
        # outward surface normals (pointing from solid into pore) from a smoothed solid field
        g = ndi.gaussian_filter(self.solid.astype(np.float32), normal_sigma)
        gz, gy, gx = np.gradient(g)
        n = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2) + 1e-6
        self.nz, self.ny, self.nx = -gz / n, -gy / n, -gx / n
        # nanoparticle grain: one 3-D texture so consecutive slices are correlated like real particles
        self.grain = 1.0 + params.grain_amp * self._corr3d(params.grain_sigma_px)
        self.bright = np.array([params.pore_floor, params.bright_catalyst, params.bright_ionomer, params.bright_dense], np.float32)
        # thin-feature indicator (fraction of empty 3x3x3 neighbours): sheet edges, particle rims -> SE edge effect
        if edge_map is not None:   # geometric edges from the generator (no voxelisation artefacts)
            self.thin = ndi.gaussian_filter(edge_map.astype(np.float32), params.edge_blur_px)
            self.thin = (self.thin / max(float(self.thin.max()), 1e-3)).astype(np.float32)
        else:
            dens = ndi.uniform_filter(self.solid.astype(np.float32), 5)
            ref = float(np.median(dens[self.solid])) if self.solid.any() else 1.0
            self.thin = np.clip(1.0 - dens / max(ref, 1e-3), 0.0, 1.0).astype(np.float32)   # ~0 on faces/interior, ~0.5 at edges/rims
        self._gain, self._offset = params.gain, params.offset
        self.ao = None
        if params.occlusion_k > 0:
            above = np.cumsum(self.solid, axis=0, dtype=np.float32) - self.solid   # solid voxels between the top and this voxel
            above = ndi.gaussian_filter(above, (0, params.occlusion_sigma_px, params.occlusion_sigma_px))
            self.ao = np.exp(-params.occlusion_k * above).astype(np.float32)
            del above
        self._charge = None   # slowly evolving charging/illumination field (AR(1) across slices)

    def _corr3d(self, sigma):
        f = ndi.gaussian_filter(self.rng.standard_normal(self.solid.shape).astype(np.float32), sigma, mode="wrap")
        return f / (f.std() + 1e-8)

    def _corr2d(self, shape, sigma):
        """Unit-variance correlated field.  Normalised by the THEORETICAL std of Gaussian-filtered white
        noise (1/(2*sqrt(pi)*sigma) in 2-D), not the spatial std: for sigma comparable to the field size a
        single realisation is nearly constant and spatial normalisation would blow it up into a random
        global brightness factor (this was a real bug: adjacent slices differed by 3x in mean)."""
        f = ndi.gaussian_filter(self.rng.standard_normal(shape).astype(np.float32), sigma, mode="wrap")
        return np.clip(f * (2.0 * np.sqrt(np.pi) * sigma), -2.5, 2.5)

    # ------------------------------------------------------------------ core
    def signal(self, z0: int) -> np.ndarray:
        """Noise-free relative SE signal of the cut plane z0 (float32, ~[0, 1.4])."""
        p, mat, vox = self.p, self.material, self.voxel_nm
        Z, Y, X = mat.shape
        m0 = mat[z0]
        solid0 = m0 > 0
        grain0 = np.where(m0 == 3, 1.0, self.grain[z0])
        S = self.bright[m0].astype(np.float32) * grain0
        # SE edge effect on the cut face: boundaries of the solid phase brighten
        dist_in = ndi.distance_transform_edt(solid0)
        edge = np.exp(-(np.maximum(dist_in - 1.0, 0.0)) / p.edge_width_px) * solid0
        S *= (1.0 + p.edge_enh * edge)
        # shine-through: march rays from every pore pixel into the material along the beam
        a = np.deg2rad(p.tilt_deg)
        dz, dy = np.sin(a), p.beam_sign * np.cos(a)
        ys, xs = np.nonzero(~solid0)
        acc = np.zeros(len(ys), np.float32)          # accumulated signal along each ray
        wgt = np.ones(len(ys), np.float32)           # remaining beam weight (1 - absorbed)
        prev_solid = np.zeros(len(ys), bool)
        active = np.ones(len(ys), bool)
        T = int(p.max_depth_nm / vox)
        bz, by, bx = self.nz, self.ny, self.nx
        for t in range(1, T + 1):
            zi = z0 + int(round(t * dz))
            if zi >= Z:
                break
            yi = np.round(ys + t * dy).astype(int)
            valid = active & (yi >= 0) & (yi < Y)
            vidx = np.flatnonzero(valid)
            if len(vidx) == 0:
                break
            m = mat[zi, yi[vidx], xs[vidx]]
            solid_now = m > 0
            hit = solid_now & ~prev_solid[vidx]          # entering a solid feature
            prev_solid[vidx] = solid_now
            if hit.any():
                h = vidx[hit]
                yh, xh = yi[h], xs[h]
                mh = m[hit]
                # Lambertian-like shading w.r.t. the beam direction, plus material contrast and depth attenuation
                cosine = -(bz[zi, yh, xh] * dz + by[zi, yh, xh] * dy)          # n . (-d): facing the beam
                if p.detector_elevation_deg < 89.0:                                # side detector: n . d_det
                    el, az = np.deg2rad(p.detector_elevation_deg), np.deg2rad(p.detector_azimuth_deg)
                    ddet = (-np.sin(el), np.cos(el) * np.sin(az), np.cos(el) * np.cos(az))
                    cdet = bz[zi, yh, xh] * ddet[0] + by[zi, yh, xh] * ddet[1] + bx[zi, yh, xh] * ddet[2]
                    vis = 0.35 + 0.65 * np.clip(cdet, 0, 1)
                    edge_dir = 0.25 + 0.75 * np.clip(cdet, 0, 1)                   # edges facing the detector are bright, far side dark
                else:
                    vis = 0.5 + 0.5 * np.clip(cosine, 0, 1)                       # in-lens detector visibility
                    edge_dir = 1.0
                sec = np.minimum(1.0 / np.maximum(np.abs(cosine), 1e-2), p.secant_max) / p.secant_max  # SE yield ~ sec(theta)
                shade = p.lambert_min + (1 - p.lambert_min) * ((1 - p.secant_weight) * vis + p.secant_weight * sec)
                shade = shade * (1.0 + p.edge_enh * edge_dir * self.thin[zi, yh, xh] * np.clip(0.5 + 0.5 * (self.grain[zi, yh, xh] - 1.0) / max(p.grain_amp, 1e-3) + 0.5, 0.2, 1.5))   # thin edges bright, varying along the rim and with detector direction
                if self.ao is not None:
                    shade = shade * self.ao[zi, yh, xh]
                depth_nm = t * vox
                g = np.where(mh == 3, 1.0, self.grain[zi, yh, xh])
                acc[h] += wgt[h] * self.bright[mh] * g * shade * np.exp(-depth_nm / p.atten_nm)
                wgt[h] *= p.transmittance
                active[h] = wgt[h] > 0.05
            if not active.any():
                break
        S[ys, xs] = acc + p.pore_floor * wgt          # rays that never (fully) stopped see the pore floor
        return S

    def depth_map(self, z0: int) -> np.ndarray:
        """For every pixel of the cut plane: depth (nm below the plane) of the structure producing its
        signal. 0 = solid on the cut plane itself; pore pixels = first-hit depth along the beam;
        -1 = ray escaped (no solid within max_depth). Proof that the image at each layer contains the
        3-D structure BEHIND the plane, exactly like a pFIB-SEM scan."""
        p, mat, vox = self.p, self.material, self.voxel_nm
        Z, Y, X = mat.shape
        solid0 = mat[z0] > 0
        D = np.where(solid0, 0.0, -1.0).astype(np.float32)
        a = np.deg2rad(p.tilt_deg)
        dz, dy = np.sin(a), p.beam_sign * np.cos(a)
        ys, xs = np.nonzero(~solid0)
        active = np.ones(len(ys), bool)
        T = int(p.max_depth_nm / vox)
        for t in range(1, T + 1):
            zi = z0 + int(round(t * dz))
            if zi >= Z:
                break
            yi = np.round(ys + t * dy).astype(int)
            valid = active & (yi >= 0) & (yi < Y)
            vidx = np.flatnonzero(valid)
            if len(vidx) == 0:
                break
            hit = mat[zi, yi[vidx], xs[vidx]] > 0
            h = vidx[hit]
            D[ys[h], xs[h]] = t * vox
            active[h] = False
            if not active.any():
                break
        return D

    def image(self, z0: int, drift: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """(uint8 image, bool mask) for cut plane z0.  The mask is exactly material[z0] > 0."""
        p = self.p
        S = self.signal(z0)
        Y, X = S.shape
        new = self._corr2d((Y, X), p.charge_sigma_px)
        self._charge = new if self._charge is None else 0.9 * self._charge + np.sqrt(1 - 0.81) * new
        S = S * (1.0 + p.charge_amp * self._charge)
        curtain = ndi.gaussian_filter1d(self.rng.standard_normal(X).astype(np.float32), p.curtain_sigma_px)
        S = S * (1.0 + p.curtain_amp * curtain / (curtain.std() + 1e-8))[None, :]
        S = ndi.gaussian_filter(S, p.psf_sigma_px)
        S = np.maximum(S, 0)
        S = self.rng.poisson(S * p.poisson_scale) / p.poisson_scale + self.rng.normal(0, p.gauss_sigma, S.shape)
        if drift:
            self._gain += self.rng.normal(0, p.drift_gain * p.gain) - 0.2 * (self._gain - p.gain)
            self._offset += self.rng.normal(0, p.drift_offset * max(p.offset, 0.02)) - 0.2 * (self._offset - p.offset)
        v = np.clip(S * self._gain + self._offset, 0, None)
        img = np.clip((v ** p.gamma) * 255.0, 0, 255).astype(np.uint8)
        return img, self.solid[z0]

    def stack(self, z_start: int = 0, z_stop: Optional[int] = None, step: int = 1):
        z_stop = self.material.shape[0] if z_stop is None else z_stop
        imgs, masks = [], []
        for z in range(z_start, z_stop, step):
            i, m = self.image(z)
            imgs.append(i)
            masks.append(m)
        return np.stack(imgs), np.stack(masks)


def render_stack(material: np.ndarray, voxel_nm: float, params: Optional[SEMParams] = None,
                 seed: int = 0, **kw):
    sim = SEMSimulator(material, voxel_nm, params or SEMParams(), seed=seed)
    return sim.stack(**kw)
