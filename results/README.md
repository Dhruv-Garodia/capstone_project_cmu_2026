# results

Small, text-only outputs of `pore_pipeline` (masks, models and caches are git-ignored; they live in
`MyDrive/Eponge/outputs/` and in `Capstone/outputs/` on Dhruv's laptop).

    <stack>/summary.json, stats.csv        classical pipeline: calibration, per-slice porosity
    <stack>/evaluation*.json               label-free checks (z-profile, x-z anisotropy)
    comp/borders.csv                       hand-drawn borders + pipeline phases, per slice
    comp/unet_<labels>_k<k>_s<seed>/       U-Net training config and validation log
    <stack>/pred_unet_*/porosity.csv       per-slice porosity of the U-Net predictions
    colab_run.log                          full log of the A100 experiment grid (2026-10-02)

Headline numbers (comp-trained 7-slice U-Net, seed 0; "high" = grey pixels read as pore,
"low" = grey as solid):

| stack    | U-Net high | U-Net low | Ferner-style | Ferner 2024 | mass balance |
|----------|-----------:|----------:|-------------:|------------:|-------------:|
| comp     | 0.57       | 0.35      | 0.55         | 0.417       | 0.60-0.65    |
| uncomp   | 0.67       | 0.46      | 0.66         | 0.558       | 0.64-0.68    |
| pristine | 0.77       | 0.52      | 0.66         | 0.515       | 0.76-0.79    |

Validation pore IoU on comp slices 100-117 (3 seeds): high k=3 0.948, high k=0 0.931,
low k=3 0.957, low k=0 0.955. The networks reproduce their labels; the high/low bracket is the
error bar on the grey-pixel assumption and cannot be narrowed by training on these labels.
