# pfib-synth

**Synthetic pFIB-SEM stacks of porous catalyst layers, with exact ground truth for every pixel.**

> **This folder is the generator's code on the team repo.** The showcase GIFs, the bundled viewer stacks and the live pages are in the standalone repo [beetroottree/pfib-synth](https://github.com/beetroottree/pfib-synth). Viewer: <https://beetroottree.github.io/pfib-synth/viewer/>. Everything here runs from this folder: `cd pfib-synth && python studio.py`. The two synthetic-trained U-Net checkpoints are not committed here (this repo ignores `*.pt`); get them from [pfibsynth/checkpoints](https://github.com/beetroottree/pfib-synth/tree/main/pfibsynth/checkpoints) if you need the experiment rows.

**[Open the 3-D viewer in your browser](https://beetroottree.github.io/pfib-synth/viewer/)** · [showcase gallery](https://beetroottree.github.io/pfib-synth/) · [make your own](#make-your-own)

![the viewer: milling and rotating a fibre mat](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/media/viewer_demo.gif)

In a pFIB-SEM stack, the beam images the cut face at 52°, so a pore does not look black: you see its lit floor or far wall a few slices down. This is *shine-through*. A pore whose floor is one slice down looks almost like solid. Hand labels and threshold rules call it solid, and every model trained on those labels learns to do the same. On real data nobody can measure how often this happens, because nobody knows the answer.

This generator builds a 3-D material, mills it slice by slice the way the microscope does, and renders each cut face with the beam geometry, the depth response and the detector measured on the team's three real acquisitions (comp, uncomp, pristine). The truth is known exactly, so the shine-through error can be measured instead of argued about.

## Showcase

Each material has five GIFs:

* **milling**: the ion beam takes one slice at a time. The front of the block is the frame the electron beam images.
* **rotate**: the solid phase in 3-D, built from the exact truth with the frame drift removed.
* **slices**: the stack with its truth, slice by slice.
* **reslice**: an x–z cut sweeping through the block.
* **shine**: where the threshold rule behind our labels misses pores, coloured by how far down the lit floor is.

The first three are calibrated to the real stacks. The last four are held out of training.

| | milling | in 3-D | more |
|---|---|---|---|
| **comp-like**<br>catalyst foam, sparse large voids | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/comp_like_milling.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/comp_like_rotate.gif) | [slices](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/comp_like_slices.gif) · [reslice](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/comp_like_reslice.gif) · [shine](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/comp_like_shine.gif) |
| **uncomp-like**<br>coarser open foam | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/uncomp_like_milling.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/uncomp_like_rotate.gif) | [slices](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/uncomp_like_slices.gif) · [reslice](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/uncomp_like_reslice.gif) · [shine](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/uncomp_like_shine.gif) |
| **pristine-like**<br>finer foam, pristine's detector | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/pristine_like_milling.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/pristine_like_rotate.gif) | [slices](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/pristine_like_slices.gif) · [reslice](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/pristine_like_reslice.gif) · [shine](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/pristine_like_shine.gif) |
| **mudcrack** (held out)<br>3-D drying-crack network | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/mudcrack_milling.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/mudcrack_rotate.gif) | [slices](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/mudcrack_slices.gif) · [reslice](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/mudcrack_reslice.gif) · [shine](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/mudcrack_shine.gif) |
| **fibre mat** (held out)<br>stacked fibres, bright rims | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/fibre_mat_milling.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/fibre_mat_rotate.gif) | [slices](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/fibre_mat_slices.gif) · [reslice](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/fibre_mat_reslice.gif) · [shine](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/fibre_mat_shine.gif) |
| **coated fibre** (held out)<br>fibres with particles and binder | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/fibre_coated_milling.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/fibre_coated_rotate.gif) | [slices](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/fibre_coated_slices.gif) · [reslice](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/fibre_coated_reslice.gif) · [shine](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/fibre_coated_shine.gif) |
| **faceted** (held out)<br>sintered grains, pores at junctions | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/faceted_milling.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/faceted_rotate.gif) | [slices](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/faceted_slices.gif) · [reslice](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/faceted_reslice.gif) · [shine](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/faceted_shine.gif) |

The stack with its truth, slice by slice (comp-like):

![comp-like: image, truth, truth on image](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/comp_like_slices.gif)

Where the threshold rule misses pores (red: floor 1 slice down, orange 2–3, yellow 4–8; blue outlines: pores it found):

![shine-through misses on comp-like](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/comp_like_shine.gif)

![all seven materials, paged slice by slice](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/media/all_materials.gif)

**Other material families** the generator has (`--family`). They are rendered in the calibrated optics but not validated against real stacks; their GIFs are in [`docs/showcase/families/`](https://github.com/beetroottree/pfib-synth/tree/main/docs/showcase/families).

| spinodal | clustered spheres | agglomerate | three-scale catalyst | fibre felt | sheet felt | compressed |
|---|---|---|---|---|---|---|
| ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/families/spinodal_rotate.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/families/spheres_rotate.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/families/agglomerate_rotate.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/families/catalyst_rotate.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/families/fiber_felt_rotate.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/families/sheet_felt_rotate.gif) | ![](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/families/compressed_rotate.gif) |

## Look through a stack: the viewer

`docs/viewer/index.html` is a single page with no install. Open it on [GitHub Pages](https://beetroottree.github.io/pfib-synth/viewer/), where all 14 stacks above are bundled, or open it locally and drop in a `.pfibv` file.

* **3-D block**: drag to rotate, scroll to zoom. There are four views:
  * the milled block, with the image on every face;
  * the solid surface from the truth;
  * the pore space;
  * an x-ray view.
* **Mill with slice**: the 3-D block is cut back to the current slice, as the beam does. **Corner cut** opens the block at the crosshair to show the inside. **z stretch** sets the slice thickness.
* **Cut face**: play or step through slices (space, arrow keys). Overlays show pore outlines, tinted pores, truth only, or a swipe between image and truth. Click to move the crosshair.
* **x–z and y–z cuts** through the crosshair, with depth running down. Click them to jump to a slice.
* **Links**: `?mode=0..3&z=40&yaw=2.4&ov=tint#fibre_mat` opens a given stack and view.

To pack your own stack: `python tools/export_viewer.py out/<name>` writes `out/<name>/<name>.pfibv`. Use `--bin 2` for a smaller file.

## Make your own

```bash
pip install -r requirements.txt
python studio.py                                          # browser form: generate, then view (below)
python generate.py --list                                 # presets, material families, render knobs
python generate.py comp_like --frames 40 --width 512      # ~40 s, writes out/comp_like/
python generate.py fibre_mat --view                       # full 118-slice package + viewer file + showcase GIFs
python generate.py --style comp --family spinodal --porosity 0.5 --name my_spinodal
python generate.py faceted --mat crystal_px=40 --set 'jitter=[0.3,0.2]' --set 'halo=[0.5,2]'
python generate.py --config examples/thick_fibres.json    # everything in a file; out/<name>/job.json reruns a stack
```

**Studio.** `python studio.py` opens a local page (127.0.0.1 only). It has:

* a form with every parameter: preset, microscope style, material family and its builder arguments, porosity, slices, width, seed, and every render knob, each shown with its calibrated value;
* the equivalent command line for whatever you have set;
* a live log while generating;
* a card for each finished stack, with its five GIFs and a link that opens it in the viewer.

![the studio](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/media/studio.png)

Two settings do most of the work:

* **Microscope style** (`--style comp | uncomp | pristine`): which real acquisition the beam, depth response, detector and surroundings of the layer are calibrated to.
* **Material family** (`--family`): how the 3-D material is built.

Any pairing works. **Render knobs** (`--set`) override single calibrated settings: slice thickness `zlen`, beam-angle edge asymmetry `edge_dy`, edge blooming `halo`, stage `jitter`, curtaining, micropores, large voids, detector saturation and noise correlation. `generate.py --list` describes each, and `pfibsynth/slicing.json` holds the calibrated values with the reasons behind them.

Each package matches the team's review packages:

* `synth_<name>_full.tif`: the image stack.
* `truth/masks/slice_NNNN.png`: the truth masks (0 = pore, 255 = solid, 128 = outside the layer, the encoding `eponge.io.export_to_classes` reads).
* `start_masks/`: what the threshold rule behind our labels would have drawn.
* `truth.json`: every setting and the true porosity.

No real data is needed to generate. The calibrated constants (`optics_v5.json`, `geometry.json`, `slicing.json`, ...) were measured on the real stacks once. The one thing the generator used to read from them at run time, the layer's grey-level histogram, ships as `pfibsynth/real_tone.json`. A stack from this repo is byte-identical to one from the original working copy (tested with the real stacks hidden).

## How a stack is made

The generator mimics the process rather than the look. The look follows from the process.

1. **One material, milled.** The material is a 3-D volume. Each frame is the face left after removing one slice, and a slice is thinner than a pixel: about 0.8–0.9 px of material per slice, fitted to how fast real structure decorrelates between frames. Coarse structure stretches further along the milling axis than fine structure, as in the real stacks. Cutting through an x–z plane gives one continuous object, not a stack of unrelated pictures:

   ![x-z cut sweeping through comp-like: image and truth](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/showcase/comp_like_reslice.gif)

2. **The beam comes in at 52°.** For every pore pixel, the renderer traces down the beam into the volume until it hits solid. The brightness falls off with that depth (fitted shine-through curve), with the secant of the wall, and with edge escape. Edge escape is directional: a pore's top edge overhangs and blooms, its bottom edge does not, so the top/bottom contrast matches real frames (1.22 vs 1.24 on comp). The frame drifts about 1.1 px per slice, and buried surfaces move a further 0.2–0.3 px per slice (parallax). In real stacks you can watch a pore's floor come forward from the back over several slices; here it is the same geometry.

3. **One-sided change.** As cover thins, a surface brightens gradually; when the beam cuts through, a pore opens all at once. So sudden darkenings outnumber sudden brightenings (11:1 on real comp), and the renderer reproduces this, although more weakly than real (below).

4. **Outside the layer.** The support, cap, caverns and voids under the layer are 3-D ellipsoids, ray-traced to their far wall, with the glow under an overhanging top edge. Curtaining stripes evolve regionally between slices. The stage jitters about 0.2 px per frame (measured). Detector noise is scan-correlated, and the detector saturates after the noise, not before.

5. **Noise or pores.** Speckle that does not survive one slice matches the noise on the featureless cap: that is noise. Specks of 1–3 px that persist across slices are micropores, and they go in the truth, not in a painted texture.

Every constant has a measurement behind it, and the reasoning for each is in [`pfibsynth/slicing.json`](pfibsynth/slicing.json) (`_why`, `_v7*` notes) and the module docstrings.

## How real does it look

Each test is a classifier trained to tell synthetic from real, scored against a real-vs-real control: a second classifier trained to tell two halves of the same real stack apart. AUC 0.5 means it cannot tell them apart; 1.0 means it always can.

| full-length stack | single frame (control) | 9-slice clips (control) | x–z reslices | arrow of time: synthetic / held-out real |
|---|---|---|---|---|
| comp | 0.86 (0.55) | 0.97 (0.51) | 0.72 | 0.99 / 1.00 |
| uncomp | 0.93 (0.58) | 0.88 (0.62) | 0.83 | 0.97 / 0.97 |
| pristine | 0.97 (0.56) | 0.90 (0.57) | 0.98 | 0.94 / 0.98 |

It is not indistinguishable yet. A classifier still separates every stack, most easily through the slices. The arrow-of-time model was trained on real clips to tell forward from reversed, and it recognises synthetic clips as forward almost as reliably as real ones, so the cutting physics is right. Two known gaps remain:

* The one-sided darkening is weaker than real (skew −1.1 vs −1.9 on comp).
* Pristine reslices are easy to spot.

Coherence along the beam does match. Take a pore pixel today; how strongly does it correlate with the image 4, 8 and 12 slices later? Comp gives 0.84/0.74/0.65 against real 0.86/0.75/0.66, and pristine gives 0.87/0.79/0.72 against real 0.86/0.77/0.72. The scorecards (`pfibsynth/scorecard.py`, `temporal_scorecard.py`) need the real stacks; point `PFIB_REAL_DIR` and `PFIB_LABELS_DIR` at them.

## Experiment: the team's best model, scored on exact truth

MillNet-SA2 is the team's best model: 0.935 mIoU against hand labels on the real test frames, 3 seeds. I ran it exactly as reported (team code at commit `7892e9e`, `segment_stack`, 1024 px tiles, k = 3, flip TTA) on 16 frames of each of the seven packages above. Next to it are the threshold rule our labels came from and two U-Nets trained only on synthetic stacks. All are scored against the exact truth, in the team's shared metrics.

| (mean over 7 materials) | recall, floor 1 / 2–3 / 4–8 / >8 slices down | solid correct | in-layer pore IoU | \|porosity error\|, all / 4 held out | 3-class mIoU | layer IoU |
|---|---|---|---|---|---|---|
| threshold rule (our labels) | 0.26 / 0.40 / 0.62 / 0.92 | 0.83 | 0.62 | 9.2 / 14.0 pp | 0.76 | (from truth) |
| **MillNet-SA2**, real labels, 3 seeds | 0.29 / 0.45 / 0.66 / 0.89 | 0.77 | 0.59 | 11.1 / 15.6 pp | 0.71 | 0.95 |
| 1-slice U-Net, synthetic | 0.54 / 0.71 / 0.87 / 0.98 | 0.88 | 0.78 | 2.8 / 3.1 pp | – | – |
| 9-slice U-Net, synthetic | **0.66 / 0.83 / 0.93** / 0.97 | **0.93** | **0.85** | **1.6 / 1.3 pp** | – | – |

![recall by depth and porosity error per material](https://raw.githubusercontent.com/beetroottree/pfib-synth/main/docs/media/millnet_on_truth.png)

**On real frames MillNet is the rule; on exact truth the rule misses most shallow pores.** On the real test frames (101–117), MillNet's pores agree with the threshold rule's at pore IoU 0.91–0.96 inside the layer (3 seeds, `experiments/real_control.py`). A model that close to the rule inherits the rule's misses. On synthetic stacks, where the truth is known, the same rule misses 74% of pores whose floor is one slice down and 60% at two to three slices. The rule's miss rate on real frames cannot be measured, but nothing suggests real shine-through is easier, so 0.935 mIoU most likely measures agreement with labels that are blind to shallow shine-through.

On the synthetic stacks themselves, MillNet behaves like the rule: it misses the shallowest pores at the rule's rate (0.29 vs 0.26 recall), and its porosity errors follow the rule's on fibres (−12.7 pp) and faceted grains (+35 pp). The three seeds agree within 1 pp, so this is systematic. But its agreement with the rule falls from 0.91–0.96 on real frames to 0.69–0.84 on synthetic, and it calls more solid as pore than the rule does (0.77 vs 0.83 solid correct). So part of what it does on synthetic is domain shift, and these synthetic numbers are not an estimate of its error on real data.

Two caveats limit how far this goes:

* **Matched renderer.** The synthetic U-Nets were trained on this renderer and MillNet was not, so part of their lead is a matched-renderer advantage. The same effect showed up as 2.9 vs 15.6 pp in the earlier cross-renderer test.
* **Synthetic is still distinguishable from real** (table above). This is the same domain shift. Part of it is specific to MillNet: it conditions on whole-frame grey-level statistics (FiLM), and here it saw whole synthetic frames, not synthetic crops pasted into a real frame (the team's `twin_context_check.py` approach). That likely explains part of the drop in agreement from 0.94 to 0.76.

The rule's layer IoU is 1.0 only because start masks take the layer outline from the truth.

On the real stacks, the 9-slice synthetic model gives porosity 0.382 / 0.479 / 0.511 against labels of 0.397 / 0.512 / 0.501. So the shine-through correction does not simply raise real porosity, and real porosity remains unsettled.

Reproduce with `experiments/score_millnet.py` (needs the team repo at `eponge-v2` and the checkpoints; see its docstring), then `experiments/plot_millnet.py`. Results are in `experiments/millnet_on_synth_v7d.json`.

### Does calibrating the decoder to labels move porosity toward the truth?

The team kept MillNet-SA2's argmax decoder and reports Yunzhi's shared, validation-selected decoder as an optional ablation: on the real test split it lowered label-based porosity MAE from 1.614 to 1.492 pp and made pristine worse. `experiments/decoder_on_truth.py` repeats her selection on these packages, where the truth is known. It uses her decoder (inside if P(inside) > layer threshold, pore if P(pore | inside) > pore threshold), her grid (pore 0.35–0.65 by 0.01, layer 0.3–0.7 by 0.1) and her guard (mIoU and pore IoU within 0.002 of argmax on every source). One shared setting per seed is chosen by porosity MAE against the threshold-rule labels on 8 validation frames per material, then everything is scored on the 16 test frames above. Her final-SA2 thresholds were not in the shared files and the four-annotator labels are not used, so this reproduces her procedure, not her numbers.

| porosity MAE, pp (3 seeds) | argmax | label-calibrated | best shared setting chosen with truth |
|---|---|---|---|
| against the truth, mean of 7 | 11.48 ± 0.18 | 11.45 ± 0.17 | 11.39 ± 0.08 |
| against the rule's labels, mean of 7 | 5.64 ± 0.35 | 5.55 ± 0.31 | 5.30 ± 0.26 |

Two seeds chose pore threshold 0.55; seed 1 kept argmax. Calibration helps comp-like, pristine-like, mudcrack and faceted against the truth by 0.08–0.31 pp, and hurts uncomp-like and both fibre mats by 0.15–0.27 pp. It hardly matters, because the true error is not a threshold problem: MillNet's bias against the truth runs from −13 pp (fibres) to +35 pp (faceted, where the rule itself is +28 pp), so even the setting picked with the truth leaves 11.4 pp. In the team's four metrics against the truth (3 seeds, mean of 7 materials), MillNet-SA2 with argmax scores 3-class mIoU 0.708 ± 0.002, true-layer pore IoU 0.575 ± 0.001, layer IoU 0.947 ± 0.001 and porosity MAE 11.5 ± 0.2 pp. Its argmax mIoU matches `score_millnet.py` to three decimals for every material. This supports the team's choice to keep argmax. All of it is on this renderer only. Results are in `experiments/decoder_on_truth_summary.json`, and `experiments/tabulate_decoder.py` prints the full per-material table.

## Training on it

`pfibsynth/shine_data.py` builds training volumes with randomised style, porosity and ±25% jittered settings. `pfibsynth/train_shine.py` trains the team's U-Net on them (`--config zwin_random_v7` for 9-slice input, `2d_random_v7` for 1 slice). It needs `torch`; the team U-Net it trains is vendored as `pfibsynth/team_unet.py`. The two checkpoints used above are in `pfibsynth/checkpoints/`.

## Layout

```
generate.py              command line: presets, families, --set / --mat / --config, --view
studio.py                local browser app: form -> generate -> viewer + GIFs
pfibsynth/
  make_full.py           build + render a full stack, write the package
  materials.py           material builders (foam, mudcrack, fibre, faceted, spinodal, spheres, ...)
  context.py             what is around the layer: support, cap, voids, caverns
  microscope.py, v4.py   beam / depth response / edge escape / detector
  make_package.py        package writer and the threshold-rule start masks
  slicing.json           per-acquisition slicing and realism settings, with the reasons
  optics_v5.json, geometry.json, real_tone.json, ...   calibrated constants
  scorecard.py, temporal_scorecard.py    realism discriminators (need real data)
  shine_data.py, train_shine.py, shine_depth_eval.py  training and evaluation
tools/
  showcase.py            the five showcase GIFs for any package
  export_viewer.py       pack a package into a .pfibv file for the viewer
  demo_media.py          gallery.png, all_materials.gif, reslice PNG
  passgif_t.py, forward.py   real-vs-synthetic GIFs and coming-forward coherence (need a real stack)
docs/                    GitHub Pages: index.html (gallery), viewer/, studio/ (served by studio.py), showcase/, media/
experiments/             MillNet-SA2 on exact truth: driver, real-frame control, decoder calibration test, results, plot
examples/                config files and a porosity sweep
```

| variable | what | needed for |
|---|---|---|
| `PFIB_OUT` | where packages are written and read back (default `out/`) | evaluation |
| `PFIB_REAL_DIR` | `comp_full.tif`, `uncomp_full.tif`, `pristine_full.tif` | realism scoring, recalibration |
| `PFIB_LABELS_DIR` | the team's labelled review packages | realism scoring, recalibration |

## Where this sits in the team's work

The Éponge team compares four lines of work on the same dataset and split:

* Aaditya: architecture (MillNet-SA2).
* Dhruv: unseen acquisitions.
* Yunzhi: porosity calibration.
* Hou Kin: this generator, the experiments where the answer is known.

The team's findings call for scoring MillNet-SA2 on these packages and testing Yunzhi's decoder calibration against truth. The two experiments above do that. Still open, and listed as future work in the team's freeze:

* Score models trained on Aaditya's learned-appearance digital twin on these packages, and ours on the twin.
* Train MillNet-SA2 on synthetic data and repeat this test.
