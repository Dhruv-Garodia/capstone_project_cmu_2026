# Findings, choices, caveats — and how to transfer this to the real pFIB-SEM stacks

*Everything learned in the synth-to-real exploration, in one place, written for the moment the team
turns from the two reference micrographs to the full stacks. Each item says what was found, what was
decided and why, and what it would take to make it better.*

---

## 1. The principle behind every appearance choice: signal vs nuisance

**The ground truth is never touched.** The label of every layer is exactly `material[z] > 0`; no
appearance effect filters, blurs or perturbs it. Every effect is applied to the *image* only.

"Every weird detail in a real picture is likely useful" — correct, and it is the reason the simulator
is built the way it is. Each effect is one of two kinds:

| effect | computed from | kind | why the model must see it | how its magnitude is set today |
|---|---|---|---|---|
| shine-through (pore pixels show solids below the plane, displaced along the beam) | the 3-D volume, ray-cast at the beam angle | **signal** | bright ≠ solid; the displacement direction identifies pore pixels | attenuation length **fitted** (30–110 nm CL; ~3 µm felt) — measure it (§5.2) |
| SE edge brightening (geometric edge map, finite width) | sheet outlines / particle rims in the generator | **signal** | edges are where the label changes | width **fitted** — measure from edge profiles |
| secant-law SE contrast, detector direction | surface normals | **signal** | orientation of the surface below | weight **fitted** |
| material contrast (catalyst / ionomer / dense) | material map | **signal** | ionomer counts as solid but is darker | ratios **assumed** (0.3–0.65) |
| thin-feature transmission (5 kV flakes ghost through) | ray continues with weight T | **signal** (felt only; 0 for IrO2 at 10 keV) | overlapping sheets | **assumed** 0.35 |
| depth attenuation, occlusion, deep-void scatter floor | ray depth, solid above the hit | **signal** (depth) | depth ordering of layers | **fitted** |
| shot noise (Poisson + read noise) | random | nuisance | exists in every real frame; the model must be invariant to it | **fitted**, then *reduced* to match the real image; randomised |
| charging / illumination field | random low-frequency field, slowly evolving across slices | nuisance | real micrographs are non-stationary | **fitted**; randomised |
| curtaining (vertical stripes) | random per column | nuisance | FIB milling artefact | **assumed** range |
| brightness/contrast drift across slices | random walk | nuisance | the paper normalises histograms per image, i.e. drift exists | **assumed** range |
| PSF | Gaussian | instrument | – | **fitted** — measure from edge spread |

Two rules that follow:

1. **Never add an effect to win a metric without naming its mechanism** and testing it against a
   real-image control. Every accepted pass in the felt loop names its mechanism (§2, F11–F16); the
   rejected ones were rejected by the test, not by taste.
2. **A distribution-matching loop can be satisfied by the wrong mechanism** (a bigger noise term can
   stand in for a missing physical cue). The defence used here — per-feature diagnostics tied to a
   mechanism, real-vs-real controls, and labels that appearance fitting cannot reach — is necessary
   but not sufficient. On the real stacks, measure each artefact *separately* (§5.2) and let the
   classifier only *verify*, never *fit*.

## 2. Core findings

**About the existing pipeline**

- **F1 — Labels and images were not aligned.** Blender's oblique perspective render (camera tilt 38°,
  azimuth 85–95°) was resized onto an orthographic z-slice mask. Consistent with IoU 0.50–0.67 on
  synthetic data with exact labels. Still untested on the old renders — one hour, do it first.
- **F2 — The target material is IrO2/ionomer agglomerates, not fibres.** Ferner et al. 2024: blade-coated
  anode CLs, 6 nm voxels, 52° beam, porosity 42–56 %, pores 50–600 nm, lognormal PSDs. Your image 1 is
  that morphology; your image 2 (flake felt) is a second family.
- **F3 — Image formation, not the geometry library, was the main gap.** The old slice is 75 % pure
  black/white; real SE images are mid-gray, granular and full of shine-through.
- **F4 — Published numbers are properties of the paper's measurement operator** (3-D open/close +
  Gaussian). Match through the same operator or you chase a σ that their pipeline removed.
- **F5 — The frame is canonical.** Slicing axis, beam direction and streak direction are fixed;
  rotation equivariance discards the strongest cue. Augment with x-flips only; y-flips only when the
  renderer randomised the beam sign.

**About the physics of the images**

- **F6 — Per-slice 2-D segmentation is nearly ill-posed for high-porosity materials.** Felt at 94 %
  porosity, 2 µm deep: 10 % of pixels are solid on the plane, 82 % show structure from below, 8 % show
  nothing; median depth of what a pore pixel shows is 1.25 µm. The image is full, the label is sparse.
  Context (2.5-D/3-D) is mandatory, not an option.
- **F7 — SE contrast follows the secant law, not Lambert.** Grazing surfaces and thin edges are
  bright; a Lambertian model gets sheet edges dark and was visibly wrong.
- **F8 — Low-frequency random fields must be normalised by their theoretical variance.** Normalising a
  nearly-constant charging field by its spatial std turned it into a random global gain (3× swings
  between adjacent slices). Found by looking at consecutive slices, not by any metric.
- **F9 — Felts are deposition processes.** Randomly placed curved ellipsoids never look like a felt;
  ballistic deposition with quantile settling and compaction does. Rigid draping (v13) and slabs (v14)
  were worse — sheets are semi-rigid.
- **F10 — Detect nothing you can generate.** A density-based edge detector lit up voxel terraces on
  curved thin sheets ("bullseyes"); a geometric edge map emitted by the generator fixed it exactly.

**About measuring "indistinguishable"**

- **F11 — Define it operationally: classifier two-sample test on held-out real data, with a real-vs-real
  control.** Left half of the reference vs its own right half scores 0.66 (texture) / 0.83
  (structure) because the micrograph is non-stationary; 0.5 is unreachable by definition. Test
  repeatability ±0.005; felt realisation spread ±0.01 (texture) / ±0.03 (structure).
- **F12 — Iterate on the top separator, not on intuition.** Rank per-patch features by KS excess over
  the control; fix the mechanism behind the top one; retest. Summary statistics (mean, std, entropy…)
  are *not* what the classifier uses — a random search on them made things worse.
- **F13 — Shot noise was the dominant texture cue** (Laplacian std 19 vs 11, KS 0.55 vs control 0.06),
  not edge sharpness. Blurring structure to "soften" the image had failed for that reason.
- **F14 — Deep voids and fine crevices have different floors.** Real large voids are lit by scatter
  (never black); real fine crevices are darker than mine. Scatter floor + crevice-scale occlusion:
  structure RF 0.931 → 0.899.
- **F15 — Brightness belongs in the highlight tail, not in the gain**, and the real image's
  non-stationarity is part of its texture. Right-skewed histogram (skew 1.28) + a large-scale
  illumination field: structure RF → 0.863, texture RF → 0.730.
- **F16 — SE edge highlights have a physical width.** Rims narrower and spikier than real (grad_max 33
  vs 25, kurtosis +0.4 vs −0.4); widening them with the peak compensated took texture kNN to the floor.
- **F17 — What did not work, by the test**: draping, slab geometry, bimodal populations, Laplacian-
  pyramid matching (invents ripples), GPNN patch harmonisation (converges to a blur fixed point),
  statistics-driven random search, JPEG-pipeline matching (no effect — confirmed on the real image).
- **F18 — Final felt (v24b)** over two realisations, floor in brackets: texture kNN 0.59–0.61 [0.584],
  texture RF 0.70–0.72 [0.657], structure kNN 0.83–0.85 [0.812], structure RF 0.84–0.89 [0.826]. At the
  floor on the non-parametric test; the RF residual is joint patch structure.
- **F19 — With one real image only non-parametric methods are honest; with many real slices train a
  structure-preserving translator** (CUT / SinCUT: PatchNCE keeps geometry, so masks stay valid; add a
  mask-consistency term; acceptance = within the control interval at both scales *and* > 0.95 IoU
  mask reproduction on translated images).

## 3. Choices made, and why

| choice | why | alternative rejected |
|---|---|---|
| numpy ray-cast SE simulator on the voxel volume | pixel-aligned labels by construction; every artefact is a parameter that can be measured on real data; 0.05 s per slice | Blender/Eevee (lighting model, camera frame, no SEM physics) |
| procedural structure generation with statistical calibration | labels and 3-D consistency for free; descriptors match published targets | text-to-image / diffusion for structure (no masks, no 3-D, no physics) |
| hierarchical CL generator (skeleton field → domains/cracks → particles → films → inclusions) | reproduces the agglomerate hierarchy seen in image 1 and described in the paper | sphere packing (wrong geometry class) |
| measure synthetic volumes through the paper's clean-up operator | the published PSD is a property of that operator | raw comparison |
| deposition felt with geometric edge maps | F9, F10 | random placement, edge detection |
| 0.125 µm felt voxels, rendered then box-downsampled to the reference's 0.25 µm/px | thin sheets need ≥ 2 voxels; edge width sub-pixel accurate | 0.25 µm voxels (puffy sheets) |
| plan view (tilt 90°) for the felt reference | image 2 is a surface micrograph, not a cut | forcing a 52° cut model onto it |
| held-out split of the reference (left = source, right = test) | the only honest way to score with one image | scoring on the same pixels the fit saw |
| two classifiers (kNN, random forest) at two scales (11 px, 33 px) | kNN is non-parametric and stable; RF finds interactions; scales separate texture from layout | a single FID-like number with an off-the-shelf CNN (no SEM prior, no control) |
| domain randomisation ranges *relative* to the calibrated base | a 6 nm CL and a 0.25 µm felt cannot share absolute ranges | fixed absolute ranges (silently wrecked the felt) |
| training code written but not executed | no torch/GPU in the authoring environment; honesty over pretence | claiming numbers |
| branch, not `main` | team review; nothing destructive | direct push |

## 4. Caveats

- Two reference images only, both low-resolution screenshots; image 1 without a scale bar; image 2 of
  an unidentified material at ×1000 — every geometric prior for the felt (thickness 0.22 µm, aspect,
  tilt spread 55°, compaction 0.88, transmission 0.35) is by eye.
- Every simulator magnitude is fitted or assumed, not measured (table in §1; full list in
  `EVIDENCE_AUDIT.md`). The three region presets match the paper's numbers, not the raw stacks.
- The RF residual (texture 0.70 vs floor 0.66) may still hide a wrong mechanism compensating a missing
  one; §5.2 is the cure.
- Felt generation costs 270 s per 512² volume on one CPU and 1.5 GB; the fibre volume predates the
  final recipe (0.25 µm voxels).
- `training.py` and `cut_train.py` compile but have never run; `smoke_test.py` first.
- 150³ boxes cap the largest pores at ~300 nm; use ≥ 256³.

## 5. Transfer playbook: replicating the real pFIB-SEM stacks

**5.0 Data hygiene first.** Split the real stacks into disjoint sets *by region and by z-block*:
calibration source, control, test. The appearance source must never overlap the test blocks; the
control (real vs real) must come from blocks as far apart as the test is from the source, so its score
is a fair floor. Report every number with the floor next to it.

**5.1 Structure.** Run `describe_volume` (through the paper's operator, r = 2) on the raw stacks; replace
`FERNER_2024` with measured (μ, σ, porosity, connectivity, S₂ length); refit presets with
`calibrate.py`. Expect σ to move; note the operator radius is a guess — try r = 1, 2, 3 and keep the one
that reproduces the paper's own numbers on the same stack.

**5.2 Measure each imaging artefact separately, then set the simulator to measured values** (this is the
step that removes "noise for fun" as a possibility):

| artefact | how to measure it on real slices | simulator parameter |
|---|---|---|
| shot noise | variance vs mean in locally flat regions (cut-face solid interiors): slope → Poisson scale, intercept → Gaussian read noise | `poisson_scale`, `gauss_sigma` |
| PSF | edge spread function at the Pt-cap/CL interface or at faceted crystallite edges; fit a Gaussian σ | `psf_sigma_px` |
| shine-through length and direction | in pore regions, brightness decay vs distance from a solid edge along +y (52° projection); autocorrelation anisotropy gives the direction and the streak length | `atten_nm`, `max_depth_nm`, `beam_sign` |
| edge highlight width and amplitude | profile across cut-face solid boundaries (from the paper's own segmentation as a proxy label) | `edge_width_px`, `edge_blur_px`, `edge_enh` |
| material contrast | histogram modes of cut-face pixels inside vs outside ionomer-rich regions (proxy labels) | `bright_ionomer`, `bright_dense` |
| charging / illumination field | low-pass each slice (σ ≈ field/10); amplitude, correlation length, and the slice-to-slice AR coefficient | `charge_amp`, `charge_sigma_px`, the 0.9 AR factor |
| curtaining | spectrum of column means; stripe amplitude and width | `curtain_amp`, `curtain_sigma_px` |
| drift | per-slice mean/std series before any normalisation | `drift_gain`, `drift_offset` |
| detector geometry | TLD (in-lens): `detector_elevation_deg = 90`, small `secant_weight`; ETD: measure the shading asymmetry across pores | `detector_*`, `secant_weight` |

Set `SEMParams` to the measured values; set the *domain-randomisation ranges* to the measured spread across
slices and stacks — not wider. Transmission is 0 for IrO2 at 10 keV.

**5.3 The floor and the test.** C2ST at 11 px and 33 px (add ~100 px for domain-scale structure) on the
held-out blocks; the control from disjoint real blocks. Expect non-stationarity with depth (the
compressed region), so compute the floor per region.

**5.4 The loop.** Same as here: rank per-patch features by KS excess over the control → name the
mechanism behind the top one → fix it in the physics → retest. If the top separator has no mechanism
you can measure (§5.2), do not fit it — leave it to the translator.

**5.5 The translator.** `cut_train.py` with the whole real set as target; mask-consistency on; acceptance
gates: C2ST within the control interval at all scales **and** frozen-segmenter mask reproduction > 0.95
IoU on translated images. Only then Stage A/B training, and a hand-labelled real test set (5–10 slices
per region, never trained on). Report physics descriptors of predicted real volumes against the paper.

**5.6 What carries over from the felt work to the CL stacks, and what does not.** Carries over: the
metric and its control, the diagnostic loop, theoretical-variance normalisation of fields, the
edge-width parameter, the deep-void floor, the illumination-field model, seed-deterministic materials.
Does not: the deposition model, plan-view geometry, transmission (off), and every felt-specific size.

**5.7 If the real target is ever a felt-like material under pFIB-SEM**: expect sparse cross-section
masks (ribbons), shine-through dominating, and a hard requirement for 3-D context; the layer sheets in
`LAYER_BY_LAYER.md` are what those stacks will look like.

## 6. Learnings, if we started again

1. Write the metric and its control before the first generator.
2. Measure artefacts before fitting them; fit only what cannot be measured, and randomise it.
3. Physics first, non-parametric second, learned translation last — each step justified by the test.
4. Look at consecutive slices; two of the bugs found here were invisible to every statistic.
5. Report over ≥ 3 realisations; the felt's own randomness is larger than most tone knobs.
6. Keep generation and rendering separate and seed-deterministic; cache volumes; keep a volume under a
   minute on one CPU or budget a GPU.
7. Keep an evidence audit next to the results, so nothing fitted is ever quoted as measured.
