# pFIB-SEM Porous Electrode Reconstruction: Literature Review and Proposed Method

Prepared 2026-09-07 for the Éponge capstone team (Aaditya, Dhruv, Hou Kin, Yunzhi).
Scope: how the microscopy, electrode, and ML communities segment FIB-SEM stacks of highly porous materials, what industry actually uses, how people evaluate without ground truth, and what this team should propose. Every citation below was located and at least abstract-checked this session; nothing is inferred beyond what was retrievable.

---

## 1. The physical problem, stated precisely

A FIB-SEM slice is not a planar cross-section. Because the SEM has a very large depth of field, solid material behind the milling plane is visible through pore openings and is nearly as bright as the cut face. The literature calls this **shine-through** (also pore-back, sub-surface, depth-of-field artifact). Consequences:

- Gray value alone cannot classify a voxel. Per-voxel thresholding, including Otsu and Ferner's gradient pipeline, systematically overestimates the solid fraction and dislocates structures along z.
- The artifact has a fixed geometry set by the 52° stage tilt and the slicing direction. z is physically distinguished from x and y.
- The cut face is identifiable only from **how a feature's intensity evolves across consecutive slices**: pore-back material appears early, brightens as the mill approaches, and persists until milled; cut-face material is stationary and disappears in one slice.

This last point is the consensus finding of a 12-year literature (Section 2) and the single most important design constraint for any method. The seniors' paper did not cite this literature, and their slice-independent 2D U-Net cannot exploit it.

Additional real-image characteristics observed in our own stack (comp_full.tif, 354 × 630 × 1860, 6 nm voxels, PTL-fiber-compressed IrO2 region): platinum cap, catalyst layer, membrane, and a delamination gap in every frame; vertical curtaining stripes; 0.25 to 2 px residual per-slice misalignment; mean pore chord 13 px (78 nm) versus 28 to 53 px in the seniors' synthetic masks.

---

## 2. How the field segments FIB-SEM of porous media

### 2.1 Rule-based, z-aware algorithms (Ulm / Fraunhofer ITWM / Freiburg)

| Paper | Material | Method | Validation |
|---|---|---|---|
| Salzer et al. 2012, Mater. Charact. 69:115 | Porous silica | Local-threshold backpropagation: a bright region is assigned to the slice of its *last* occurrence in z | Visual vs global threshold |
| Salzer, Thiele, Zengerle, Schmidt 2014, Mater. Charact. 95:36, DOI 10.1016/j.matchar.2014.05.014 | PEMFC Pt/C catalyst layer | Same principle on a catalyst layer; the closest structural analog to IrO2 CL | Manual segmentation |
| Prill et al. 2013, J. Microsc. 250:77, DOI 10.1111/jmi.12021 | Nanoporous electrodes | Morphological: edges mark where a bright region *first* appears, then constrained watershed | Simulated stacks with known truth |
| **Salzer et al. 2015, J. Microsc. 257:23, DOI 10.1111/jmi.12182** | Simulated | **Benchmark of three FIB-SEM-specific algorithms**; no universal winner; decision table by porosity and feature size | Misclassified-voxel fraction plus porosity and chord-length fidelity |
| Prill & Schladitz 2013, Scanning 35:189, DOI 10.1002/sca.21047 | Simulator | Fast Monte Carlo SEM simulator; shine-through emerges physically | n/a |
| Moroni & Thiele 2020, Ultramicroscopy 219:113090 | PEFC CL, Li-ion | Optical flow between slices: pore-back structures appear to move, cut-face material does not | Manual; CL accuracy 86.6% vs threshold much lower |
| Roldán et al. 2021, Ultramicroscopy 226:113291; 2024, J. Microsc., DOI 10.1111/jmi.13254 | Simulated + real | Anisotropic-sampling bias on descriptors; quality indices for noise, curtaining, charging | Synthetic twins |

### 2.2 Supervised ML on real hand-labeled slices

| Paper | Material | Method | Result |
|---|---|---|---|
| Röding et al. 2021, J. Microsc. 281:76, DOI 10.1111/jmi.12950 | Porous polymer films | Scale-space features + random forest; **labels on 100 random 256² patches (~0.5% of voxels) with ±3-slice context**; open data and code | Porosity 21.7 / 29.5 / 44.9% vs known leached fraction 22 / 30 / 45% |
| Skärberg et al. 2021, DOI 10.1111/jmi.13007 | Same data | Multi-slice-input CNN | Beat the random forest |
| **Osenberg et al. 2023, J. Power Sources 570:233030, arXiv 2207.14114** | NMC cathode | ilastik random forest with tailored "milling-evolution" features; one densely labeled slice as truth | Weighted error: double threshold 60.8%, z-gradient 23.2%, RF single detector 12.3%, RF two detectors 10.3%, RF milling-evolution features **6.1%** |
| Sun et al. 2024, APL 125:173902 | LiCoO2 | Swin transformer on fused SE + BSE | Reduces z-elongation artifact |
| Otic & Kinefuchi 2025, J. Power Sources, DOI 10.1016/j.jpowsour.2025.237556 | PEMFC CL | DeepLabv3+ 2D → 3D U-Net for z-continuity → super-resolution net | Closest electrode analog; same two-stage design as the seniors' paper, so the architecture alone is not novel |

### 2.3 ML trained only on simulated stacks (the seniors' family)

| Paper | Method | Outcome |
|---|---|---|
| **Fend, Moghiseh, Redenbach, Schladitz 2021, J. Microsc. 281:16, DOI 10.1111/jmi.12944** | 3D U-Net trained solely on Prill-simulated stacks of Boolean sphere models | "Usable", qualitative only on real data |
| **Schladitz et al. 2024 review, arXiv 2501.18313** | Retrospective on the above | Verbatim: synthetic data "turned out to be too clean"; fixed by "gray value histogram matching"; this "does not shed a good light on generalizability"; "it is not clear at all, which characteristics of the real image data have to be met to which degree of fidelity" |
| **Sardhara et al. 2022, Front. Mater. 9:837006** | Monte Carlo (MCXray) rendered synthetic nanoporous gold; compared 2D U-Net with 3/5/7/9 adjacent slices against 3D U-Net, V-Net, ResUNet3D | On real data the **2D net with adjacent slices won**; misplaced-pixel fraction 0.10 vs Otsu 0.12 vs k-means 0.39; lineal-path anisotropy 0.004 vs 0.24 to 0.37 |
| Sardhara et al. 2024, Nanomanuf. Metrol., DOI 10.1007/s41871-024-00223-y | Multi-kV acquisition turns shine-through depth into a signal channel | Better than single-kV |
| Sardhara et al. 2025, MLST, DOI 10.1088/2632-2153/adc870 | ANN BSE predictor + CycleGAN sim-to-real | ~20% real-data improvement over plain simulation |
| Müller et al. 2021, Nat. Commun., DOI 10.1038/s41467-021-26480-9 | Synthetic XCT + CycleGAN refinement; gap measured on 400 manually labeled slices | Dice pore 0.63 → 0.69, Si 0.69 → 0.82 with synthetic augmentation |

**Reading of this family:** the group that invented synthetic-trained FIB-SEM segmentation hit exactly the wall the seniors describe as "harsher noise on real data", four years earlier, and still lists it as open. Photorealistic rendering was never the fix; what helped was (a) physically simulating shine-through, (b) feeding adjacent slices, (c) style transfer to real texture, and (d) label-free physics metrics to measure the gap.

### 2.4 Experimental workarounds

Groups with instrument access sidestep the problem: silicone or epoxy infiltration of pores (Hegge et al. 2018, J. Power Sources 393:62, IrRuOx PEMWE anode, porosity 55%), e-beam Pt infilling (Cunha et al. 2026, Energy Storage Mater.), low-kV imaging to shorten shine-through depth (Beran et al. 2026; Fager et al. 2020), dual-detector SE + BSE or InLens + SE2 acquisition (Osenberg 2023; Sun 2024). Our data is unfilled, single-detector, 10 keV. We cannot change acquisition, so the burden falls entirely on the algorithm.

---

## 3. What industry and established labs actually use

- **Acquisition suites** (Thermo Auto Slice & View 5, ZEISS Atlas 5, TESCAN) do drift correction, Y-shift correction, and curtain-reducing rocking or spin milling. They do not solve shine-through.
- **Post-processing** is still done by hand: alignment (AMST, Sci. Rep. 2020; Yuan et al., Ultramicroscopy 2021) and wavelet-FFT de-curtaining (Münch et al. 2009, in ImageJ).
- **Segmentation** is almost always supervised with painted labels: ORS Dragonfly Segmentation Wizard (random forest or U-Net; 86.7% reported on artifact-laden FIB-SEM), Thermo Amira-Avizo Segmentation+ (U-Net with ResNet/VGG backbones), ZEISS Intellesis (scikit-learn random forest), or open-source ilastik and Trainable Weka. None are label-free.
- **Metrics** come from PoreSpy (porosity, local-thickness PSD, REV), TauFactor 2 (tortuosity factor by diffusion solve, GPU), OpenPNM, PuMA, or GeoDict.
- **Validation** in the best groups is by independent measurement: MIP (IJHE 2015 resin-filled CL; Odungat 2025), Archimedes density (Shukla 2019, JES), Wicke-Kallenbach diffusivity (Prokop 2022), N2 adsorption (Bierling 2026, Small, IrO2 CL porosity ~61%), and REV by sub-volume statistics (De Angelis 2023, Sci. Rep., IrO2/TiO2 PEMWE anode: 50 sub-volumes, SD under 1% for pores; notes ~25% internal IrO2 nanoporosity unresolved at 16 nm).
- **No public IrO2 catalyst-layer FIB-SEM benchmark exists.** No surveyed paper publishes an end-to-end open pipeline with uncertainty bars.

Practical implication: a materials scientist with a Dragonfly license and two hours of painting would today get a supervised segmentation of comp_full.tif that is probably better than any of our current models. Our library has to beat that on either label cost, reproducibility, or validated uncertainty, or it has no user.

---

## 4. Evaluation without ground truth: what the literature supports

No single metric is defensible. The field triangulates:

1. **Sparse expert labels on random patches.** Röding: ~0.5% of voxels; Machireddy et al. 2023 (Front. Bioinformatics): 10 to 24 labeled slices gave Dice 0.76 to 0.99. Label with ±3-slice context so the annotator can see z-evolution. Use two annotators and report inter-annotator agreement as the accuracy ceiling. No materials paper found does the two-annotator step.
2. **Label-free physics proxies** (Sardhara 2022): relative difference of the two-point correlation and lineal-path functions between the xy plane and the z direction. Shine-through inflates z-anisotropy, so a correct segmentation of an isotropic material is isotropic. Also per-plane pore diameter and detector consistency where available.
3. **Physical cross-checks**: porosity from loading, thickness, and density (Shukla 2019 style); Ferner's published values for the same three regions; REV by sub-volume convergence.
4. **Uncertainty bands**: Norris et al. (Sandia, SAND2024-02784J, OSTI 2320335) run MC-dropout Bayesian CNN → per-voxel solid probability → percentile segmentations → CDFs of porosity, conductivity, and tortuosity. Tortuosity uncertainty was ~6× larger than conductivity uncertainty. No FIB-SEM paper does this.

Things the literature says to avoid: zero-shot SAM/SAM2 for interconnected pore/solid (works only on isolated object-like pores; false positives documented); z super-resolution that assumes isotropy; SE(3)-equivariant networks (no serial-section evidence; z is physically distinguished, so at most in-plane dihedral symmetry is justified); Dice on self-generated synthetic data as a headline result.

---

## 5. Where the current project stands against this

| Aspect | Seniors / current plan | Literature |
|---|---|---|
| Handling shine-through | None; slice-independent 2D U-Net | Every method uses z-evolution: backpropagation, optical flow, milling-evolution features, adjacent-slice channels |
| Synthetic imaging | Blender photoreal render of a bisected mesh, trapezoid not aligned to mask | Monte Carlo or physics-based depth shading; domain randomization; CycleGAN to real texture |
| Synthetic geometry | PuMA sphere packing, pores 2 to 4× coarser than real in pixels | Geometry matched to measured chord lengths; SliceGAN/MicroLib from real slices |
| Real-image phases | Two (pore/solid) | Real frame has Pt cap, CL, membrane, gap |
| Baselines | None | Otsu, Salzer, Prill, Ferner; Otsu alone gives 0.383 ± 0.024 vs Ferner 0.417 on our stack |
| Evaluation | Pixel accuracy and IoU on synthetic; one qualitative real figure; reconstruction JSON has unit bugs and 0.000 porosity error everywhere | Sparse labels + anisotropy proxies + physical cross-checks + UQ |
| Architecture ambition | SE(3)/Sim(3)-equivariant 3D U-Net (e3nn) | No evidence; wrong symmetry for fixed 52° geometry |
| Deliverable | pip library with mesh-export options | Align → de-curtain → crop → segment → REV → metrics → uncertainty report |

---

## 6. Proposed method: Éponge v2

**Thesis.** The scientific contribution is not a new network. It is the first open, z-aware, uncertainty-quantified porosity and tortuosity pipeline for unfilled IrO2 PEMWE catalyst layers imaged by Xe-pFIB-SEM, validated by triangulation on real data, released with the first public IrO2 CL benchmark.

### 6.1 Pipeline

1. **Preprocess (deterministic, no knobs).** Rigid slice registration by chained phase correlation on the CL band (report residual). Wavelet-FFT de-curtaining (Münch 2009) with a fixed, published parameter set. Automatic CL band detection from row-wise intensity statistics; crop Pt cap and membrane. Per-stack histogram normalization to a reference.
2. **Segmentation: z-aware 2.5D U-Net.** Input channel stack = slices i−k … i+k (k = 2 or 3, chosen by ablation, following Sardhara 2022 and Skärberg 2021). Optional extra channels: forward z-difference and backward z-difference maps, which hand the network Osenberg's "milling-evolution" features and Ferner's gradient idea explicitly. In-plane flips and 90° rotations only as augmentation; no z flips, because z is not symmetric. Output = solid probability, not a hard mask.
3. **Synthetic data v2 (the model's only teacher).**
   - Geometry: PuMA packings whose sphere-size distribution is fit to the real Otsu chord-length statistics, plus SliceGAN volumes generated from real slices (MicroLib route), plus bimodal and dense-agglomerate variants to cover Ferner's "large dense particles".
   - Imaging: replace the Blender bisected-mesh render with a **first-hit depth-shading model**: for each pixel, cast a ray along the beam direction at 52°, find the first solid voxel, and set intensity as a decaying function of depth below the cut plane, plus edge brightening. This is a cheap approximation of the Prill/MCXray physics and produces shine-through by construction. Render at 6 nm, pixel-aligned to the mask by construction (no perspective camera).
   - Domain randomization: Poisson and Gaussian noise, vertical curtaining stripes, charging gradients, brightness drift along z, sub-pixel slice misalignment, and synthetic Pt cap and membrane bands at the frame edges.
   - Optional stage: CycleGAN or lighter histogram/style transfer from synthetic to real texture, validated by t-SNE overlap and by the anisotropy proxy, not by eye.
4. **Sub-surface ablation.** Render each synthetic geometry twice, with and without shine-through. Train and test crosswise. This measures directly whether the model learned to find the cut face, which is the actual research question and has never been reported for an electrode CL.
5. **Uncertainty.** MC-dropout or a 5-seed ensemble → probability map → percentile segmentations (10th to 90th) → distributions of porosity, PSD, and tortuosity factor (TauFactor 2). Report every number as median with 10 to 90 band.
6. **Metrics.** Porosity, local-thickness PSD (PoreSpy, matching Ferner), tortuosity factor in three directions (TauFactor 2), REV curve by sub-volume.
7. **Baselines built into the library.** Otsu, Salzer backpropagation, Ferner's six-step method, and the seniors' 2D U-Net, all run on identical crops so the comparison table is one command.

### 6.2 Evaluation protocol

- **Real sparse labels**: 100 random 256² patches per stack with ±3-slice context, two annotators (team plus one domain expert from the Litster group if possible), report agreement. Hold out entirely from any tuning.
- **Anisotropy proxy** on every real segmentation: xy-vs-z two-point correlation and lineal-path relative difference.
- **Physical cross-checks**: Ferner's three region values; loading-thickness-density porosity bound; REV convergence; run-to-run stability across seeds and across 5 registration perturbations.
- **Synthetic held-out**: IoU, porosity MAE, and the sub-surface ablation gap, reported alongside the classical baselines, never alone.

### 6.3 Hypotheses to replace H1 to H4 in the planning report

- **H1 (z-awareness).** A 2.5D U-Net with adjacent-slice channels reduces z-anisotropy of the real segmentation and sparse-label error relative to the slice-independent 2D U-Net and Otsu. Target: anisotropy below 0.05 and sparse-patch error below the inter-annotator disagreement.
- **H2 (physics over photorealism).** Depth-shaded synthetic images with domain randomization transfer better to real data than Blender renders, measured on the same sparse labels and the anisotropy proxy.
- **H3 (uncertainty is informative).** Percentile-segmentation bands on porosity are narrower than the classical parameter-sensitivity range (Ferner's ±10 points) while containing Ferner's values, and tortuosity bands are materially wider than porosity bands.
- **H4 (reproducibility).** Same stack, same config, five independent runs and five registration perturbations: porosity spread below 1 percentage point.

### 6.4 What to cut

- SE(3)/Sim(3)-equivariant 3D U-Net and e3nn.
- The seven mesh-export "reconstruction methods" and Chamfer distance evaluation.
- Marching-cubes mesh output as a scientific result. Keep STL export as a convenience only.
- Graphite flake and fuel-cell cathode framing in the report; the data is IrO2 PEMWE anode.

### 6.5 Timeline sketch (14 weeks)

| Weeks | Work |
|---|---|
| 1–2 | Preprocess module; baselines table on comp_full and real_input2; verify seniors' code for render/mask alignment and evaluation units |
| 3–5 | Synthetic v2: depth-shading renderer, geometry matched to real chord statistics, domain randomization; sub-surface ablation harness |
| 4–6 | Sparse labeling protocol and 200 patches labeled by two annotators |
| 6–9 | 2.5D U-Net training, k ablation, z-difference channels; anisotropy proxy; UQ via ensemble |
| 9–11 | Full triangulated evaluation on both real stacks; REV; tortuosity via TauFactor |
| 11–13 | Library hardening, one-command report, docs, benchmark release (stack + patches + labels) |
| 14 | Final report and paper draft |

---

## 7. Read-first list

1. Salzer et al. 2015, J. Microsc. 257:23 (the benchmark and decision table).
2. Osenberg et al. 2023, J. Power Sources 570:233030 (best-documented electrode pipeline with numbers; open tools).
3. Sardhara et al. 2022, Front. Mater. 9:837006 (full synthetic-data recipe, adjacent-slice U-Net, label-free validation).
4. Schladitz et al. 2024, arXiv 2501.18313 (honest retrospective on synthetic-trained FIB-SEM).
5. Röding et al. 2021, J. Microsc. 281:76 (sparse patch labeling protocol; open data).
6. Norris et al. 2024, Sandia SAND2024-02784J (uncertainty propagation to porosity and tortuosity).
7. Moroni & Thiele 2020, Ultramicroscopy 219:113090 (optical-flow use of shine-through on a catalyst layer).
8. De Angelis et al. 2023, Sci. Rep. (IrO2 PEMWE anode 3D, REV protocol, tortuosity values to compare against).
9. Ferner et al. 2024, IJHE 59 (our data's provenance and target values).
10. Kench & Cooper 2021, Nat. Mach. Intell. (SliceGAN, for real-derived synthetic geometry).

---

## Sources

Salzer 2015 https://pubmed.ncbi.nlm.nih.gov/25231671/ · Prill 2013 https://pubmed.ncbi.nlm.nih.gov/23550597/ · Prill & Schladitz 2013 https://onlinelibrary.wiley.com/doi/abs/10.1002/sca.21047 · Salzer 2014 https://www.osti.gov/biblio/22403536 · Salzer 2012 https://www.sciencedirect.com/science/article/abs/pii/S1044580312001003 · Moroni & Thiele 2020 https://www.sciencedirect.com/science/article/pii/S0304399120302412 · Roldán 2021 https://www.sciencedirect.com/science/article/abs/pii/S0304399121000796 · Roldán 2024 https://onlinelibrary.wiley.com/doi/10.1111/jmi.13254 · Röding 2021 https://pmc.ncbi.nlm.nih.gov/articles/PMC7754501/ · Skärberg 2021 https://onlinelibrary.wiley.com/doi/10.1111/jmi.13007 · Fager 2020 https://academic.oup.com/mam/article/26/4/837/6887639 · Osenberg 2023 https://arxiv.org/abs/2207.14114 · Fend 2021 https://pubmed.ncbi.nlm.nih.gov/32681535/ · Schladitz 2024 https://arxiv.org/html/2501.18313v1 · Sardhara 2022 https://www.frontiersin.org/journals/materials/articles/10.3389/fmats.2022.837006/full · Sardhara 2024 https://link.springer.com/article/10.1007/s41871-024-00223-y · Sardhara 2025 https://iopscience.iop.org/article/10.1088/2632-2153/adc870 · Sun 2024 https://pubs.aip.org/aip/apl/article-abstract/125/17/173902/3318260/ · Otic & Kinefuchi 2025 https://www.sciencedirect.com/science/article/abs/pii/S0378775325013928 · Müller 2021 https://www.nature.com/articles/s41467-021-26480-9 · Hegge 2018 https://doi.org/10.1016/j.jpowsour.2018.04.089 · De Angelis 2023 https://www.nature.com/articles/s41598-023-30960-x · Bierling 2026 https://onlinelibrary.wiley.com/doi/10.1002/smll.202513029 · Shukla 2019 https://iopscience.iop.org/article/10.1149/2.0251915jes · Norris 2024 https://www.osti.gov/biblio/2320335 · Kench & Cooper 2021 https://www.nature.com/articles/s42256-021-00322-1 · MicroLib 2022 https://www.nature.com/articles/s41597-022-01744-1 · TauFactor 2 https://joss.theoj.org/papers/10.21105/joss.05358 · PoreSpy https://joss.theoj.org/papers/10.21105/joss.01296 · Machireddy 2023 https://doi.org/10.3389/fbinf.2023.1308708 · Dragonfly wizard https://theobjects.com/dragonfly/dfhelp/Content/Artificial%20Intelligence/Segmentation%20Wizard/Segmentation%20Wizard.htm · Auto Slice & View https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/auto-slice-view-datasheet.pdf · AMST https://www.nature.com/articles/s41598-020-58736-7 · Ferner 2024 https://www.sciencedirect.com/science/article/pii/S0360319924004312
