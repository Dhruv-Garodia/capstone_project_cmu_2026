# Closing the appearance gap: what "indistinguishable" means, what worked, what didn't, what's next

## 1. Make "indistinguishable" measurable first

A human judging one image pair is not a metric. The operational definition used here is the
**classifier two-sample test (C2ST)**: sample thousands of patches from held-out real data and
from the synthetic image, train a classifier to tell them apart with cross-validation, and report
its AUC. 0.5 would be perfectly indistinguishable. Two patch scales: 11 px (texture) and 33 px
downsampled 3× (structure). Two classifiers: k-nearest-neighbour vote (non-parametric, no fitting)
and a random forest (learns feature interactions).

Crucially, the test needs a **control**: the left half of your felt micrograph against its own
right half. That control scores **0.66 (texture) / 0.83 (structure)** — the two halves of the
same image are themselves "distinguishable", because the micrograph is non-stationary (the right
half is brighter and denser, mean 59 vs 52). So the target is the control's score, not 0.5;
anything at the control level is indistinguishable *for this test*.

| candidate | texture kNN | texture RF-AUC | structure kNN | structure RF-AUC |
|---|---|---|---|---|
| real vs real (control = floor) | 0.584 | 0.657 | 0.812 | 0.826 |
| v6 (previous turn's felt) | 0.677 | 0.834 | 0.829 | 0.958 |
| v12a simulated | 0.630 | 0.814 | 0.855 | 0.934 |
| **v12c simulated (best)** | **0.627** | **0.786** | 0.856 | 0.946 |
| v12d / v12e (tone variants) | 0.648 / 0.657 | 0.795 / 0.796 | 0.879 / 0.892 | 0.943 / 0.946 |
| v12a + patch-NN harmonisation | 0.710 | 0.903 | 0.785 | 0.944 |
| v12c + patch-NN v3 (centre-aggregated) | 0.695 | 0.876 | 0.796 | 0.929 |

Reading: on texture, v12c is 0.04 (kNN) / 0.13 (RF) above the floor; on structure ~0.04 / 0.12.
The v6 → v12c progress is real (0.834 → 0.786 texture RF). The remaining structure gap is the
size/curvature distribution of sheets and the ghosting of thin flakes, i.e. generator geometry,
not rendering.

## 2. What was tried this turn

**Physics passes (v7 → v12), each one motivated by a specific visible fault**

| pass | change | why | result |
|---|---|---|---|
| v7 | 0.125 µm voxels + roll/fold/torn-outline sheets + ambient occlusion + gamma | thin curled ribbons, black voids | step change; too crisp, shadows too black |
| v8 | heavier PSF + background floor, tighter length clip | softness | regression (haze, confetti) — rejected |
| v9 | v7 geometry, intermediate tone | – | close on tone; **bullseye rings** on rolled sheets (voxel terraces lit by the edge detector) |
| v10 | geometric edge map from the generator instead of edge detection; smoother normals | remove terrace artefacts | rings gone; but uniform rims around flat faces |
| v11 | denser, more upright pile; face shading restored; rim brightness modulated | 3-D look | tone matched; but compaction flattened curvature → tiles |
| v12 | **thin-feature transmission** (5 kV flakes are partially electron-transparent), less compaction, larger sheets | see-through ghosting, curvature back | best statistics; shadows lifted in v12c |

**Non-parametric "learn from the one real image" (patch_transfer.py)**

* Laplacian-pyramid histogram matching (Heeger–Bergen style): invents ripple structure — rejected.
* GPNN-style multi-scale patch nearest-neighbour harmonisation (Granot et al. 2022): structure from the simulation, patches from the real image. Three variants (plain, centre-weighted, centre-only + more iterations). All blur: a smooth query finds smooth real patches and the loop converges on that fixed point; the test confirms it (texture AUC 0.88–0.90 vs 0.79 for the plain simulator). Kept in the package as tooling, **not** as a recommended step.

## 3. Why the remaining gap needs a trained translator, and which one

With one real image, the appearance model has to be learned from crops of that image with a
structure-preserving objective — otherwise the label of the simulated input is lost. The right
family is **contrastive unpaired translation (CUT)** and its single-image variant **SinCUT**:
PatchNCE maximises mutual information between input and output patches at the same location, so
geometry (and therefore the mask) is preserved, while a patch discriminator pushes texture, tone,
noise and edge rendering onto the real distribution. `cut_train.py` implements it (ResNet
generator with exposed encoder features, 70×70 PatchGAN, NCE heads, identity NCE, optional
mask-consistency through a frozen segmenter) and finishes with the C2ST acceptance test on the
held-out strip that training never sees. Expected on a single GPU: ~1 h for SinCUT on the felt.

Why not the alternatives:
* diffusion/SDEdit: no label preservation guarantee, needs a trained SEM prior, expensive;
* CycleGAN: works but weaker structure preservation than PatchNCE and 2× the networks;
* neural style transfer (Gram matrices): fine for one image, no batch generation, blurs edges;
* pure procedural tuning: it got within 0.04 kNN of the floor on texture; the last stretch is a
  learned map, and once it is learned it applies to every generated image for free.

For the catalyst-layer stacks the situation is better: thousands of real slices ⇒ normal CUT
with the whole real set as target, the simulated stacks (with masks) as source, and the C2ST
control computed between two disjoint sets of real slices.

## 4. Acceptance criterion going forward

A translated (or simulated) dataset is "indistinguishable enough" when its C2ST scores fall
within the bootstrap interval of the real-vs-real control at both patch scales, evaluated on
held-out real data, **and** the frozen segmenter reproduces the input masks on the translated
images to > 0.95 IoU (label preservation). Both numbers are printed by `cut_train.py`.

## 5. Where the felt still differs from your image (visible, honest)

* real sheets are larger and more varied, with more edge-on ribbons; mine are slightly small and uniform (structure C2ST);
* real faces carry faint streak texture; mine are smooth;
* real thin flakes ghost through each other more strongly; transmission 0.35 is a guess;
* the reference is a ×1000, downsampled, JPEG-softened image of an unidentified material; a native micrograph with a scale bar would change every geometric prior above.


---

## 6. Loop 2 (v13 -> v25): diagnostic-driven passes

Each pass was accepted or rejected by the two-sample test on the held-out half, and the direction of each
pass came from a **separator diagnostic**: per-patch features (mean, std, min, max, skew, kurtosis, gradient
mean/max, Laplacian std, lag-1/lag-3 autocorrelation) scored by how much they separate real from synthetic
*beyond* the real-vs-real control (KS excess). Findings, in the order they were discovered:

1. **Draping / slabs / bimodal populations (v13-v16): rejected** - worse on both scales; and a second random
   realisation of the same recipe moves the score by ~0.03, so small structural knobs are below the noise floor.
2. **Compression is not the residual**: JPEG-matching the reference changes nothing (0.790 vs 0.786), confirmed by
   applying the same operation to the real image.
3. **Pixel noise was the dominant texture cue** (Laplacian std 19 vs 11, KS 0.55 vs control 0.06): shot noise
   survived the downsample. Earlier "softening" had blurred structure instead. Removing the noise: 0.816 -> 0.793.
4. **Void floor and crevice occlusion (v19)**: at 33 px my large voids were darker than real and my fine crevices
   lighter; a scatter floor for deep voids plus crevice-scale occlusion took structure RF 0.931 -> 0.899.
5. **Right-skewed histogram (v21-v22)**: real skew 1.28 vs mine 0.80 - brightness belongs in the highlight tail,
   not the gain. Together with a **large-scale illumination field** (the real micrograph is non-stationary,
   which is why the control floor is high) structure RF fell to 0.863 and skew matched (1.29 / 1.28).
6. **Edge-highlight width (v24)**: my SE rims were narrower and spikier than real (grad_max 33 vs 25, kurtosis
   +0.4 vs -0.4). Widening the highlight (electron escape depth) with the peak compensated: texture RF 0.704,
   texture kNN 0.592 vs floor 0.584.

Final recipe (v24b) over two random realisations, floor in brackets:
texture kNN 0.59-0.61 [0.584], texture RF 0.70-0.72 [0.657], structure kNN 0.83-0.85 [0.812],
structure RF 0.84-0.89 [0.826]. Test repeatability +/-0.005; realisation spread +/-0.01 (texture) / +/-0.03 (structure).

What is left is joint patch structure the marginals no longer show (RF texture 0.70 vs 0.66), plus the visible
facts that real sheets are more ribbon-like with finer fluff than my plates. The translator (`cut_train.py`)
addresses the first; a native micrograph addresses the second.
