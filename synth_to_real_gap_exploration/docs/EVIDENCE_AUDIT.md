# Evidence audit — grounded vs assumed, and what the old synthetic lacked

*Companion to `REEVALUATION.md`. Written so nothing here can be mistaken for measured fact when it is not.*

## A. What I actually looked at

| source | used for | limits |
|---|---|---|
| Ferner et al. 2024 (text only; figures not extracted) | material, imaging geometry (52°, 6 nm, TLD 10 keV, Pt cap), porosity / PSD / lognormal fits, "dense low-porosity particles", segmentation operator | numbers are properties of *their* segmentation pipeline; I never saw their Fig. 5/6 pixels |
| your image 1 (751×265 screenshot, two panels, no scale bar) | agglomerate hierarchy, speckle, cracks, faceted crystallites; appearance statistics at ~2× downsampling | unknown pixel size → every *size* fitted from it can be off by ~2× |
| your image 2 (SEM banner: 5 kV, ×1000, 10 µm bar ≈ 0.25 µm/px after downsampling) | the felt family; plan-view appearance statistics | surface micrograph, not a slice; material unknown |
| team docs (vision, HLD, logs, weekly summary), repo, `dataset_v1` scripts, one `png_slices` PNG | pipeline facts, IoU numbers, camera geometry | – |
| **not opened**: the three real stacks (0.4–1.5 GB), the Blender renders (750 KB each), the senior checkpoint | – | the three region presets are calibrated to the paper, **not** to those stacks |

## B. What the real data has that the OLD synthetic (PuMA spheres + Blender) does not

| # | feature of the real data | evidence | old synthetic | new synthetic |
|---|---|---|---|---|
| 1 | agglomerate hierarchy: domains separated by cracks | image 1 L; Ferner "wide range of length scales" | isolated spheres | modelled (Voronoi cracks + correlated skeleton) |
| 2 | nanoparticle speckle inside agglomerates (primary particles tens of nm) | image 1; Ferner §3.4 | smooth spheres | particle decoration (geometry) + 3-D grain (appearance) |
| 3 | faceted crystallites and large dense low-porosity particles | image 1 R; Ferner §3.4 ("found across all samples") | none | polyhedra / ellipsoids with ~5 % micro-porosity |
| 4 | ionomer as a second, darker solid phase (I/C 0.2) | Ferner §2.1, §3.6 ("did not distinguish phases") | none | second material with lower SE yield (level **assumed**) |
| 5 | porosity 41.7–55.8 % | Ferner Fig. 8 | 0.73 (senior), 0.49 (yours) | 0.414 / 0.512 / 0.556 measured through the paper's operator |
| 6 | pores 50–600 nm, lognormal PSD (μ, σ) per region | Ferner Table 1 | sphere bins 89–1647 nm | μ within 0.09, σ within 0.1 (150³) |
| 7 | shine-through: solids below the cut plane imaged through pores, displaced along the 52° beam | Ferner §2.4 ("52° streak artifacts"); log: "edges irregular, non-linear" | Eevee spotlight render of a cut mesh (oblique perspective camera) | ray-cast model with depth attenuation, normal shading |
| 8 | SE topographic contrast: grazing surfaces and thin edges bright | SEM physics (secant law); image 2 edges | none | secant term + thin-feature boost |
| 9 | mid-gray textured histogram (entropy 4.4–4.8 bits, 6–12 % extreme pixels) | measured on image 1 | 75 % pixels at 0/255, entropy 2.7 | within noise of image 1 |
| 10 | PSF, shot noise, charging field, curtaining, slow brightness drift | Ferner (Pt cap "to mitigate curtaining"; per-image histogram normalisation to a reference ⇒ drift) | none | modelled; **magnitudes fitted, not measured** |
| 11 | image and label in the same pixel frame | geometric argument; IoU ≤ 0.67 with exact labels | no (oblique render → orthographic mask) | by construction |
| 12 | compressed region: thinner, lower porosity, smaller pores | Ferner §3.4–3.6 | none | through-plane squeeze (**form assumed**) |
| 13 | felt: thousands of thin crumpled sheets resting in layers, bright edges, deep gaps | image 2 | (old `flake_network`) random curved ellipsoids in space | ballistic deposition of crumpled sheets, compaction; fibre mode |
| 14 | inter-slice correlation of texture (same particles across slices) | physics of serial sectioning | independent renders | one 3-D grain field; slowly evolving charging field |

## C. What in the NEW synthetic is assumed — the "made-up" list

Nothing below was measured. Each is either a physically motivated form or a number fitted to a low-resolution screenshot. Treat them as knobs, not facts.

| item | value used | status | effect if wrong |
|---|---|---|---|
| ionomer film thickness / surface coverage | 6–10 nm / 55–85 % | assumed | appearance only; mask unaffected (ionomer counted as solid, as in the paper) |
| ionomer SE brightness vs catalyst | 0.30–0.65 | assumed (low-Z ⇒ darker) | appearance |
| crack width / ionomer fill of cracks | 40 nm / 40 % | from image 1 without scale bar | geometry of the `mudcrack` preset (not the three paper presets) |
| domain size | 720 nm | from image 1 without scale bar; could be 2× off | `mudcrack` preset only |
| dense inclusion count and radius | 0–7, 90–320 nm | paper says "some"; counts assumed | minor |
| primary particle radius | lognormal (26 nm, 0.35) | consistent with "tens of nm"; not measured | roughness of agglomerate edges |
| shine-through attenuation length | fitted 30–110 nm | fitted to appearance, not a measured electron-optics quantity | how far below the plane solids remain visible — the core ambiguity the model must learn |
| grain / charging / curtaining / PSF / noise | fitted | fitted to a downsampled screenshot | must be refit on native crops |
| compression as a pure z-scaling | 0.85 | assumed mechanism | `comp` preset geometry |
| agglomerate skeleton = correlated Gaussian field | – | statistical proxy, not a drying/cracking model | fine for PSD; not a process model |
| measurement operator = 3-D open/close, r = 2 voxels | – | paper says spherical element; radius not stated | σ of the PSD (±0.05) |
| felt sheet thickness, tilt spread, compaction, settling | 0.3 µm, 50°, 0.75, 0.15 | by eye against a ×1000 image | felt geometry; material of image 2 unknown |
| felt plan view = beam normal to the pile top | tilt 90° | reasonable for a surface SEM | – |

## D. What the old pipeline had that was not carried over

* PuMA's effective-property solvers (tortuosity, conductivity). Not needed for segmentation, but the vision doc promises tortuosity → add porespy/taufactor on the predicted volumes.
* OBJ meshes as the primary artefact. The new pipeline is voxel-native; `utils/convert_mesh.py` / marching cubes still produce meshes when needed.
* The oblique-camera perspective. It would only be a feature if labels lived in the same camera frame; they did not.

## E. Verdict per component

| component | status | what would falsify it |
|---|---|---|
| labels aligned with images | by construction (new); **old pipeline misalignment still not tested by anyone** | overlay mask edges on an old Blender render; re-render `--flat` and watch IoU |
| structure of the three paper regions | validated against published numbers through the paper's operator | measuring the real stacks with `describe_volume` and getting different (μ, σ) — likely for σ, since the paper's operator radius is a guess |
| mud-crack / faceted appearance | statistics within noise of a screenshot; distance 0.75 / 0.29 | native-resolution crops with scale bars |
| felt structure + appearance | family-level match (distance 0.28; feature scale 10 vs 6–8 px) | native crop; sheet thickness measurement; a 0.125 µm-voxel run |
| SE image formation | physically motivated, fitted; **two bugs found and fixed during visual inspection** (independent per-slice charging field; spatial-std normalisation of a nearly constant field) | native slices: compare slice-to-slice mean drift and the shine-through streak length |
| training / losses / metrics | compiled, never executed | `smoke_test.py` on the GPU box |
| dataset CLI | executed end-to-end (small shapes) | 256³ runs for memory/time |

## F. Remaining gaps, ranked by how much they would change conclusions

1. **Native real pFIB-SEM slices** (3–5 per region). Changes: appearance fit, shine-through length, domain size, and replaces every paper-derived target with a measured one via the same code.
2. Alignment test on the old renders (one hour; decides whether the 0.5 IoU numbers were ever about the model).
3. Felt at 0.125 µm voxels with 0.2 µm sheets (8× cost) to get crisper edges; measure sheet thickness from a tilted view if the sample matters.
4. Mud-crack residual (std 35 vs 59 at the screenshot's scale) — probably domain size / crack contrast; needs item 1.
5. `training.py` execution; then Stage A (repo U-Net on aligned data) as the label sanity check.
6. 256³ generation for the 600 nm pore tail; the box currently caps max pore at ~300 nm.
7. Tortuosity module.

## G. Rule for keeping this honest

The descriptor table (`describe_volume` for structure, `image_stats` for appearance) is the contract. Any change to a generator or the renderer ships with the table recomputed against the same references; "looks better" without a number is not a result. When the real stacks are measured, the paper-derived targets in `FERNER_2024` are replaced by measured values and the presets are re-fitted with `calibrate.py`.
