# Self-review against a top-venue standard

Written as a referee would read the work after all experiments in this repository. It separates what
the evidence supports from what it does not, and lists the one missing dataset that would change the
verdict.

## Verdict

**Not yet ready for an A\*-level venue; publishable at a strong specialist venue** (e.g. *Journal of
Microscopy*, *Ultramicroscopy*, *npj Computational Materials*, a CVPR/MICCAI workshop) with the claims
restricted to the solid ones below. One dataset, about 20 hand-labelled IrO₂ patches drawn with ±3-slice
context by two annotators, would let the real-data claims be tested directly, and is the main thing
between this work and a top-venue submission.

## Claims the evidence supports

| claim | evidence | strength |
|---|---|---|
| The original napari labels are a session-dependent raw threshold and under-read porosity by half | 100.00 % agreement with I < 140–150 on 115/116 frames; [ANNOTATIONS.md](ANNOTATIONS.md) | decisive |
| The ~1 px/slice offset in comp_full is sub-surface sliding, not drift | cut-face regions static at every slice gap; [FINDINGS.md](FINDINGS.md) | decisive |
| Shine-through biases thresholding by ≈ −0.10 porosity (−23 %) | exact truth on three twin geometries | strong (simulation) |
| Twin-trained networks beat the Ferner et al. (2024) method even with best-case parameters | pore IoU 0.93 vs 0.85, 3 seeds, sd ≤ 0.005; holds with SliceGAN training geometry vs real test geometry | strong (simulation) |
| Slice context and the streak bank make MillNet robust to structure shift | +0.026 / +0.009 IoU when training structures differ from test (1 seed) | moderate |
| On real hand-labelled FIB-SEM data, our 7-slice models match the published state of the art | MillNet 7-slice 0.800 ± 0.003, ResUNet 7-slice 0.797 ± 0.004 vs Skärberg et al. 0.798; every paired bootstrap CI of the difference includes 0; both beat Röding et al. (0.762) | strong |
| Pretraining on the IrO₂ twin helps when labels are scarce, even across material and microscope | 10 % labels: 0.760 ± 0.001 vs 0.743 ± 0.008 from scratch (2 seeds); no gain with all labels | moderate |
| The compressed region is less porous than the uncompressed one | every method and every twin variant agree with Ferner et al. on comp < uncomp | strong |

## Claims the evidence does not support (and should not be made)

| claim | why not |
|---|---|
| "MillNet is the best architecture" | on the public benchmark it ties ResUNet and the published CNN; on matched twin geometry it ties ResUNet |
| "Our porosity values for comp / uncomp / pristine are correct" | twin-trained models agree with each other on the twin but spread 0.46–0.58 on real comp_full and 0.23–0.43 on pristine (published 0.515); no method reproduces the published ordering comp < pristine < uncomp. Absolute real-data porosity is not pinned down by any method here, including the published one |
| "Self-training on unlabelled slices closes the sim-to-real gap" | mean-teacher self-training lowered accuracy (10 % labels: 0.719 vs 0.743); with twin pretraining it was neutral (0.759 vs 0.760) |
| "The twin is realistic" beyond grey-level statistics | calibration matches percentiles (≤ 7 grey levels) but texture, wall roughness and SliceGAN feature size differ measurably |

## What a referee would ask for, and the status

| request | status |
|---|---|
| Seeds and confidence intervals | done: 2–3 seeds on every headline number, paired bootstrap on the public test set |
| External benchmark with real labels | done: Röding/Skärberg data, official split, published baselines reproduced |
| Strongest classical baseline | done: Ferner et al. reimplemented from the paper and given best-case parameters on the twin |
| Independent training geometry | done: SliceGAN structures for training, real-derived for testing |
| Cross-sample validation | done: three stacks; comp < uncomp robust; pristine and absolute values method-dependent |
| **Real IrO₂ ground truth** | **missing: needs ~20 patches × 2 annotators with ±3-slice context (Röding protocol); the napari review tool in `eponge/scripts/napari_review.py` can be used** |
| Independent physical measurement | missing: mercury porosimetry or N₂ adsorption on the same electrodes would anchor absolute porosity |
| nnU-Net baseline | not run; on the public benchmark the ResUNet already ties the published CNN, so a large difference is unlikely, but referees expect it |
| Ferner et al.'s original code and parameters | not public; ask the corresponding author (S. Litster) |

## Recommended model

* **Single slices / the website:** MillNet 2D (trained on the corrected labels).
* **Stacks:** MillNet 7-slice. Train on real hand labels when available; when labels are scarce, start
  from the IrO₂ twin weights (`runs/twin_slicegan/millnet_k3/best.pt`) and fine-tune.
* **Porosity reporting:** give a range across methods and state the label definition. Thresholding
  gives a lower bound; twin-trained networks and the z-gradient method give the shine-through-corrected
  estimate.

## Paper outline that the evidence supports

1. Problem: shine-through makes FIB-SEM labels and segmentations of porous electrodes biased; label
   audits of real annotations show the bias directly.
2. Method: a calibrated digital twin of pFIB-SEM serial sectioning (real-derived and SliceGAN geometry,
   physics of shine-through) giving exact truth.
3. Results 1: quantified bias of thresholding; classical correction (Ferner et al.) plateaus even at
   best-case parameters; twin-trained networks recover the cut plane (IoU 0.93) and true pore shape.
4. Results 2: on real hand-labelled data, the same networks match the state of the art; twin
   pretraining reduces the labels needed.
5. Discussion: sim-to-real gap for absolute porosity; what real ground truth is needed; negative results
   (self-training; per-acquisition calibration, which needs a physical constraint and still overshoots).
