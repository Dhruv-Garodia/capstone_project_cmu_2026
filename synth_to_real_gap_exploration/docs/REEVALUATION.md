# Éponge — project re-evaluation and synthetic-data pipeline v2

*Prepared 2026-08-30 from the GitHub repo, the Drive (design docs, progress logs, Ferner et al. 2024, dataset_v1, Blender scripts) and the two uploaded real micrographs. Everything in `eponge_synth/` was executed in a sandbox (1 CPU, no GPU); the PyTorch training code was written for your GPU box and compiled but not executed here.*

---

## 0. TL;DR

1. **The training signal of the current pipeline is broken before any modelling starts.** The Blender renderer produces an oblique perspective view (camera tilt 38°, azimuth 85–95°, 1080×720) that is resized onto an orthographic 150×150 z-slice mask. Labels and images are not pixel-aligned. This is consistent with every number you have: senior checkpoint val IoU 0.53 *on its own synthetic data*, your scratch baseline 0.51, fine-tune 0.50. A U-Net with exact labels should clear 0.9. Your progress log already asked "is the ground truth vs the output angled?" — it is.
2. **The target material is not fibrous.** The Drive stacks (`pristine/uncomp/comp_full.tif`) are the three regions of Ferner et al. 2024: blade-coated **IrO₂ anode catalyst layers** — nanoparticle/ionomer agglomerate networks, 6 nm voxels, porosity 42–56 %, pores 50–600 nm, published lognormal PSD fits. Your image 1 is exactly that morphology (speckled agglomerates, mud-crack domain boundaries, faceted crystallites). Your image 2 is a micron-scale flake felt — a *second morphology family*, which the library should support, not the target of the catalyst-layer work.
3. **PuMA is not the main gap; the image-formation model is.** The current synthetic slice is near-binary (75 % of pixels at 0/255, entropy 2.7 bits) while real SE images are mid-gray, granular and shine-through-textured (entropy 4.4–4.8 bits, 6–12 % extreme pixels).
4. **Delivered:** a calibrated, physics-motivated replacement for PuMA + Blender that generates a 150³ volume and simulates its full slice stack with exact pixel-aligned masks in ~20 s on one CPU; three catalyst-layer presets that reproduce the published porosity/PSD numbers within noise; two extra morphology families (faceted crystallites, flake felt) fitted to your uploaded images; the descriptor "ruler" that measures the synthetic–real gap; FDA/histogram-matching style transfer; and the 2.5D/3D training stack with boundary and inter-slice consistency losses.
5. **Architecture verdict:** drop the Sim(3)-equivariant e3nn U-Net as the core objective. pFIB-SEM has a canonical frame (slicing axis, beam direction, 52° streak direction); rotation-equivariance discards the strongest cue and costs ~10× compute. Train a 2.5D DeepLabV3+ or a plain 3D U-Net on *aligned, simulated* stacks first — that is where the IoU is.

---

## 1. What the project is (from the documents)

| Source | Content that matters |
|---|---|
| Capstone_Vision, Éponge deck | Deliverable = pip-installable library: stack → segmentation → 3D reconstruction → porosity / tortuosity. Targets: IoU > 0.9 synthetic, > 0.85 real, < 5 % porosity error. Mentor's "core objective": Sim(3)-equivariant 3D U-Net (e3nn), 10k+ synthetic volumes with varying porosity/morphology/anisotropy. |
| Ferner et al. 2024 (orig_paper.pdf) | Material: IrO₂ + Nafion (I/C 0.2). pFIB-SEM Helios, TLD, 10 keV, **52° beam-to-face angle**, 6 nm voxels, Pt cap above, Nafion below. Shine-through → "52° streak artifacts", removed by reslicing + forward-gradient threshold + 3D open/close + Gaussian. Results (Table 1 / Fig. 8): pristine 51.5 % / 123 nm / μ 4.657 σ 0.635; PTL-pore-area (uncomp) 55.8 % / 166 nm / μ 4.926 σ 0.740; PTL-fiber-compressed (comp) 41.7 % / 104 nm / μ 4.572 σ 0.613; max pore ≈ 600 nm (300 nm compressed); large dense low-porosity particles in every sample. |
| Progress log / weekly summary | Synthetic-on-synthetic IoU 0.50–0.67; CPU training; "confirm camera viewing angles"; real edges "irregular, non-linear"; plan to label real data. |
| dataset_v1 + Blender scripts | Eevee render, spotlight, oblique camera, progressive `bisect_plane`; your PSD-based volumes at 10 nm voxels, porosity ≈ 0.49. |
| Repo | 2D U-Net (1-ch, single slice), `train.py` already has CE/Dice/focal/Tversky; senior PuMA config: `generator: sphere`, bins 89–1647 nm, realized porosity 0.73; `model-recon-output` metrics unit-broken (volume_ratio 2.8e5). |

## 2. Audit — what is actually broken, ranked by leverage

| # | Problem | Evidence | Consequence |
|---|---|---|---|
| 1 | Image/mask misalignment | oblique perspective render vs orthographic mask; IoU ≤ 0.67 with exact labels | every downstream experiment measures noise |
| 2 | Wrong image-formation model | Eevee lighting; no depth-attenuated shine-through, curtaining, charging, shot noise; near-binary output | model learns a lighting model, not SEM physics; zero transfer |
| 3 | Wrong geometry class & scale | sphere packing, 10 nm voxels, porosity 0.73 (senior) / 0.49 (yours) vs 0.42–0.56 real; pores 89–1647 nm vs 50–600 nm | descriptors of training data never overlap the real distribution |
| 4 | Per-slice 2D model | single-slice U-Net; shine-through makes a pore pixel look like solid from below the plane | fundamentally ambiguous per slice — the paper needed 3D context (reslice) too |
| 5 | Metrics | Chamfer/Hausdorff on unit-mismatched meshes; no PSD/tortuosity acceptance test | success is undefined |
| 6 | Equivariance as the core objective | data has a fixed frame; augmentation with rotations even *hurts* (breaks the streak direction) | wasted quarter |
| 7 | No unlabeled-real usage | 3 real stacks (0.4–1.5 GB) untouched by training | the cheapest sim-to-real levers unused |

## 3. Flaws in thinking → corrections

* **"We need a life-like fibrous generator."** → You need a *catalyst-layer* generator (agglomerate hierarchy) for the funded target, plus morphology *families* for the library's generality. Fiber and flake families are included; the CL family is primary.
* **"PuMA looks bad, replace the library."** → Any generator looks bad through a wrong imaging model. Fix image formation first (done), then geometry (done), then verify with descriptors, not eyes.
* **"Craft a prompt and generate images with an image model."** → Text-to-image gives no masks, no 3D consistency, no physics. The right "prompt" is a *quantitative generation spec* (§7) executed by a procedural generator with labels for free. Generative models belong at the *appearance* stage (FDA now, CUT/CycleGAN later if the descriptor table still shows a gap).
* **"Compare synthetic to the paper's numbers directly."** → The published PSD/porosity are properties of their *measurement operator* (3D open/close r≈2 + Gaussian). Measure synthetic volumes through the same operator (`describe_volume(..., clean_radius=2)`), otherwise you chase a σ that their pipeline removed.
* **"More rotation augmentation = more robust."** → Only x-flips are frame-safe. y-flips only if the simulator randomised the beam sign (`--randomize_beam`). No in-plane rotations.
* **"Fine-tune on real data."** → You have no real labels; the "paper-seg-output" is a pseudo-label from a thresholding pipeline. Use real data unlabeled (appearance calibration, FDA, self-training) and hand-label 5–10 slices for a *test* set only.

## 4. What was built (`eponge_synth/`)

| Module | What it does | Validated here |
|---|---|---|
| `microstructure.py` | `catalyst_layer()`: 3-scale correlated field → Voronoi mud-crack domains → lognormal primary-particle decoration → partial ionomer films → dense inclusions (ellipsoid / faceted polyhedra) → through-plane compression → bisection to exact porosity. `flake_network()` (curled platelets), `fiber_network()`. Presets: `pristine, uncomp, comp, mudcrack, faceted`. | yes (13–17 s per 150³) |
| `sem_render.py` | SE image simulator: 52° ray-cast shine-through with depth attenuation, normal shading, SE edge brightening, material contrast (catalyst/ionomer/dense), 3D nanoparticle grain, charging field, curtaining, PSF, Poisson + Gaussian noise, per-slice drift; `SEMParams.random()` for domain randomisation. Masks are exactly `material[z] > 0`. | yes (0.05 s per 150² slice) |
| `descriptors.py` | porosity, local-thickness PSD + lognormal MLE (Ferner's definition), chord lengths, S₂ correlation length, connectivity, specific surface, `paper_cleanup`, image texture statistics, `gap_table`. | yes |
| `calibrate.py` | grid search of generator parameters against Ferner Table 1 through the paper's operator. | yes |
| `domain_adapt.py` | `calibrate_appearance` (random search of `SEMParams` on texture statistics of real images), `histogram_match`, `fda`, `random_real_style` augmentation. | yes |
| `training.py` | `SliceDataset` (2.5D, frame-safe aug, real-style aug), `PatchDataset3D`, `UNet` 2D/3D, DeepLabV3+ (smp), CE+Dice, boundary loss, inter-slice consistency loss, metrics, `predict_stack`, `evaluate_volume` (3D IoU + descriptor errors). | compiled only — run `smoke_test.py` on the GPU box |
| `make_dataset.py` | CLI: N volumes, random presets, domain-randomised appearance around the calibrated params → `stacks/<name>/{images,masks,material}.npy` + preview. | yes (end-to-end) |

## 5. Structural calibration — synthetic vs published real (150³, 6 nm, measured through the paper's operator)

| region | porosity | mean pore (nm) | μ | σ | max pore (nm) | pore connectivity |
|---|---|---|---|---|---|---|
| pristine | **0.512** / 0.515 | **128** / 123 | **4.666** / 4.657 | **0.683** / 0.635 | 276 / 600 | 0.999 |
| uncomp (PTL-pore-area) | **0.556** / 0.558 | **185** / 166 | **5.016** / 4.926 | **0.758** / 0.740 | 300 / 600 | 0.999 |
| comp (PTL-fiber-compressed) | **0.414** / 0.417 | **117** / 104 | **4.564** / 4.572 | **0.709** / 0.613 | 216 / 300 | 0.996 |

(synthetic / target; at 112³ during calibration uncomp was μ 4.930 / mean 168 — the difference is seed + box size). The max-pore shortfall is a box-size effect: 150 voxels at 6 nm = 0.9 µm cannot contain a 600 nm pore population. **Generate at ≥ 256³ (1.5 µm) for training.** The old 150³ @ 10 nm was the right field size at the wrong voxel size. See `samples/psd_vs_ferner.png`.

## 6. Appearance calibration — simulator fitted to your two real images

Distance = RMS of 7 normalised texture statistics (mean, std, entropy, extreme-pixel fraction, gradient magnitude, spectrum slope, local contrast). Old pipeline for reference.

| statistic | real image 1 (L) | old synthetic | new mud-crack | real image 1 (R) | new faceted | real image 2 | new flake |
|---|---|---|---|---|---|---|---|
| mean | 71 | 166 | 58 | 101 | 99 | 55 | 61 |
| std | 59 | 108 | 40 | 64 | 62 | 25 | 35 |
| entropy (bits) | 4.44 | 2.71 | 4.15 | 4.77 | 4.52 | 3.50 | 3.94 |
| extreme fraction | 0.12 | 0.75 | 0.13 | 0.06 | 0.03 | 0.00 | 0.01 |
| gradient mean | 11.0 | 37.8 | 12.1 | 14.4 | 17.0 | 7.9 | 7.7 |
| spectrum slope | −3.49 | −2.62 | −3.24 | −3.59 | −3.10 | −2.94 | −2.86 |
| local contrast | 17.3 | 68.7 | 17.7 | 24.5 | 27.5 | 12.7 | 12.0 |
| **distance** | – | ≫ 3 | **0.48** | – | **0.59** | – | **0.35** |

Caveats: image 1 is a low-resolution screenshot with no scale bar (matched at 2× downsampling; its domains are ~4 per field, which a 0.9 µm slice cannot show — a native crop with a scale bar will let the domain size be calibrated); image 2 is a surface micrograph, so it was matched against the simulator's surface view (z = 0, long attenuation), not a cut plane. Montages: `samples/montage_v3_calibrated.png`, `samples/old_vs_new.png`.

## 7. The generation spec (the "prompt", made executable)

> Generate a 3-D binary/material volume of a proton-exchange-membrane electrolyser IrO₂ anode catalyst layer at 6 nm voxels, ≥ 1.5 µm on a side. Build it hierarchically: (a) mud-crack domains 0.5–1.2 µm across separated by meandering 40–100 nm cracks that are 30–40 % filled with ionomer; (b) inside domains, agglomerates of lognormal primary particles (median radius 22–26 nm, σ 0.35) arranged on a correlated skeleton with 90–130 nm correlation length plus a 250 nm macro-void component, so that the pore-phase local-thickness distribution, measured after 3-D opening/closing with a 2-voxel sphere, is lognormal with (μ, σ) = (4.66, 0.64) at 51.5 % porosity [pristine], (4.93, 0.74) at 55.8 % [uncompressed], (4.57, 0.61) at 41.7 % with 0.85× through-plane compression [compressed]; (c) 6–10 nm ionomer films on 55–85 % of the solid surface; (d) 0–7 dense low-porosity inclusions, ellipsoidal or faceted polyhedra, 90–320 nm radius. Then image every cut plane as a 10 keV TLD secondary-electron micrograph viewed at 52° to the face: shine-through of solids below the plane displaced along the beam with 30–110 nm attenuation, Lambertian shading, edge brightening, ionomer at 0.3–0.65 of the catalyst yield, 1–2 px nanoparticle grain, 5–30 % charging field, ≤ 12 % curtaining, 0.5–1.4 px PSF, Poisson noise at 40–200 counts, 8-bit gain 0.25–1.0 with slow drift; label = solid on the cut plane.

That paragraph is `make_preset(...)` + `SEMParams.random(...)`. Two more families implement image 2 (`flake_network`) and PTL/GDL-like fibres.

## 8. Model and training plan

* **Stage A — prove the labels.** Train the repo's own 2D U-Net on `datasets/v2` (aligned masks). Expect val IoU > 0.9 within a few epochs. If not, the bug is in the loader, not the data.
* **Stage B — context.** 2.5D DeepLabV3+ (prev/cur/next, `--context 1`, later 2) vs 3D U-Net on (32, 128, 128) patches. Losses: 0.5 CE + 0.5 Dice, then + 0.1 boundary (`--w_bd`) + 0.1 consistency (`--w_cons`). Report 2D IoU, Dice, boundary F1 **and** `evaluate_volume`: 3D IoU, porosity error, PSD (μ, σ) error.
* **Stage C — sim-to-real (unlabeled real).** (1) appearance calibration on native-resolution real crops (`calibrate_appearance`); (2) FDA + histogram-matching augmentation with real references (`--real_refs`); (3) self-training with confidence-filtered pseudo-labels on real slices; (4) CUT/CycleGAN only if the descriptor table still shows a gap.
* **Stage D — real test set.** Hand-label 5–10 real slices per region (use the paper's 52° reslice trick to help annotators separate cut-face solid from shine-through). Never train on them. Also report porosity/PSD of the predicted real volumes against Ferner's values — that is the mentor's acceptance test.
* **Not now:** e3nn equivariance, diffusion segmentation, adversarial refinement. Revisit only if Stages A–D are done and time remains.

## 9. Metrics and acceptance tests

| level | metric | target |
|---|---|---|
| pixel | IoU / Dice / boundary F1 on synthetic val | > 0.9 IoU |
| volume | 3D IoU, porosity |error| | > 0.9; < 2 % |
| physics | PSD (μ, σ) error vs GT; porosity/PSD of real predictions vs Ferner | |Δμ| < 0.1, |Δσ| < 0.1; within the paper's error bars |
| stability | slice-to-slice porosity std on real stacks | comparable to synthetic GT |
| gap | descriptor distance real ↔ synthetic (structure + appearance) | monotone decrease per iteration |

## 10. Roadmap (Fall, 14 weeks)

* **W1–2** alignment test; generate `datasets/v2` (≥ 200 volumes, 256³, all presets, `--randomize_beam`); Stage A.
* **W3–5** Stage B experiments (the Fall Plan's questions 1–4 answered on aligned data).
* **W6–8** native real crops → appearance re-calibration; FDA/self-training; label the real test set.
* **W9–11** real evaluation, failure analysis, descriptor-level reporting; tortuosity module (porespy/taufactor).
* **W12–14** library packaging (CLI: `eponge segment stack.tif --model best.pt --report`), docs, paper figures.

## 11. Tools

`numpy/scipy` (everything here), `porespy` (independent local-thickness cross-check; the paper used it), `segmentation_models_pytorch` (DeepLabV3+), `MONAI` (3D U-Net + patch sampling, optional), `taufactor` or porespy for tortuosity, `hydra`/`wandb` for experiment tracking, `Dragonfly` (already in use) for the real test-set labels. Not needed: Blender, PuMA, e3nn.

## 12. Status and limitations

* Validated: generator, simulator, descriptors, calibration, dataset CLI, end-to-end generation.
* Compiled but not executed: `training.py` (no torch in the sandbox). Run `python smoke_test.py` first.
* Open items: (i) native-resolution real crops with scale bars (both morphologies) to tighten appearance and domain-size calibration; (ii) 256³ generation for the 600 nm pore tail; (iii) the ionomer/catalyst contrast is a modelling assumption (the paper did not separate the phases); (iv) tortuosity not yet implemented.

## 13. How to run

```bash
pip install -r requirements.txt
python make_dataset.py --out datasets/v2 --n 200 --shape 192 256 256 --families catalyst_layer --randomize_beam
python make_dataset.py --out datasets/felt --n 40 --shape 96 320 320 --families sheet_felt fiber_felt
python -m eponge_synth.training --data datasets/v2 --arch deeplabv3plus --mode 2.5d --epochs 40 --w_bd 0.1 --w_cons 0.1 --real_refs samples/real_uploads
python -m eponge_synth.calibrate            # structural calibration (re-run after changing the generator)
python smoke_test.py                        # 1-epoch check of every model on a tiny dataset
```
