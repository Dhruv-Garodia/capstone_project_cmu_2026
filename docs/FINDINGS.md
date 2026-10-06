# Findings

What we learned about the data, the labels and the earlier pipelines, in order of impact.
Each item lists the evidence and what changed because of it.

## 1. The napari "manual" masks are a grey-level threshold

Inside the hand-drawn catalyst-layer outline, the pore/solid labels equal `raw intensity < t`
for 100.00 % of pixels in 115 of 116 frames. The threshold changes between annotation sessions
(142 for frames 0–29, 147–150 for 30–60, 140 for 61–117), and porosity jumps with it.
At 0.18 the labelled porosity is half of the published values for this region (0.383 Otsu,
0.417 Ferner et al. 2024).

* Why it mattered: a network trained on these labels learns detector noise and session-dependent
  thresholds. The earlier error analysis (98.7 % of pore errors ≤ 16 px, 84.5 % within 3 px of a
  boundary) is that noise.
* What changed: corrected labels (v2). The hand-drawn outline is kept, the Pt-cap/membrane rim is
  trimmed, and pore is defined by one denoised Otsu rule on every frame: porosity 0.354, continuous
  across frames. Details in [ANNOTATIONS.md](ANNOTATIONS.md).
* Cross-check: D. Garodia's independent classical pipeline (`pore_pipeline/`) brackets porosity
  between 0.322 (uncertain pixels as solid) and 0.508 (uncertain as pore). Its Otsu alone gives
  0.354, the same as v2.

## 2. The stack does not drift; the ~1 px/slice shift is shine-through

Correlating neighbouring slices inside the catalyst layer gives a best y-offset of about 1 px
per slice. That offset is not stage drift:

| region | dy per slice at gap 1 / 2 / 5 / 10 slices |
|---|---|
| catalyst layer (rows 150–350) | −0.30 / −1.27 / −0.01 / 0.00 |
| PTL fibres and membrane (rows 450–630, cut-face only) | −0.10 / −0.08 / −0.01 / 0.00 |

Cut-face structures are stationary, and the in-layer offset is incoherent across gaps. What moves
is sub-surface material seen through pore openings, which slides as the mill approaches it
(52° geometry). D. Garodia's pipeline notes this and deliberately does not register; an early
version of `eponge` did register, which shifted the cut face by up to tens of pixels. Registration
is now off by default and documented in `eponge/preprocess.py`.

## 3. The earlier real-data training skipped a third of every frame

`experiments/train_eval_real_multiclass.py` tiled frames with strides larger than the patch
(374 × 674 for 256 × 512 patches). 32.9 % of every frame was never trained on, including rows
256–373 through the middle of the catalyst layer. It also used the same six crops in every epoch.
`eponge.train` samples random crops with scale, gamma, noise, blur and curtaining augmentation.

## 4. Model selection must not use the test frames

`pore_pipeline/unet/train.py` validates on frames 100–117 and keeps the best checkpoint there, so
its reported scores are on the selection set. Here, frames 79–90 select the checkpoint and frames
100–117 are evaluated once, at the end, for every model.

## 5. 2D sections understate connectivity

At 35–45 % porosity the pore phase rarely spans the layer in a single slice (2D site percolation
needs about 59 %), while in 3D most pore volume does. The website therefore reports solid-phase
tortuosity and the largest pore cluster for single images, and pore-phase tortuosity and
percolation only for stacks.

## Limits that remain

* All labels are rule-based. Model scores measure agreement with a rule, so the physical checks
  (porosity against the literature, z-continuity, isotropy, cross-pipeline agreement) carry the
  scientific weight.
* Otsu-type rules still call bright shine-through solid, so v2 porosity is a lower bound.
* One stack (comp_full) trains and tests everything. The pristine and uncompressed stacks are the
  natural next test of generalisation.
