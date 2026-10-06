# The website

A single static page (`webapp/index.html`). The model runs in the visitor's browser with
TensorFlow.js on the GPU (WebGL), so there is no server, and uploaded images never leave the
visitor's computer.

## What a visitor can do

1. **Upload** a PNG, JPG or TIFF (8/16/32-bit, multi-page stacks too), or pick a sample.
2. Set the **pixel size**. With "Match model scale" on, the image is resampled to 6 nm/px, the scale the model was trained at.
3. Read the **summary**: porosity with its uncertainty range, catalyst-layer thickness, median pore
   diameter, interface density, tortuosity factor, pore connectivity.
4. Explore the three visualisations:
   * **Segmentation map**: overlay, mask, pore-diameter heat map, connectivity, model confidence, raw.
   * **Pore statistics**: pore size distribution, porosity through the layer (with uncertainty band), chord lengths.
   * **3D pore network**: three.js viewer with orbit, zoom, a sliding cut along the milling direction,
     and colouring by pore diameter or by connectivity. It shows a 128-slice comp_full reconstruction
     by default; a multi-page TIFF upload is meshed in the browser and shown instead, with 3D tortuosity.
5. **Copy results as JSON**.

Typical time for one 1860 × 630 frame on a laptop GPU: 3–8 s.

## Files

| path | what |
|---|---|
| `index.html` | the page body (styles, markup, scripts); `build_static.py` wraps it into a full HTML document |
| `build_static.py` | wraps the page into a full document and assembles `dist/` for any static host |
| `model/manifest.json`, `model/weights.bin` | MillNet 2D weights (BatchNorm folded, float16, 5.1 MB) from `eponge export-web` |
| `model/results.json` | model-card table, written by `eponge/scripts/compare_models.py` |
| `samples/` | sample frames and the 3D mesh (`pore_mesh.bin/json`, plus `pore_mesh.glb` for Blender/ParaView) |

## Deploy to Vercel

```bash
python webapp/build_static.py         # creates webapp/dist
cd webapp/dist
npx vercel login                      # once
npx vercel deploy --prod              # prints the public URL
```

Without logging in, `npx vercel deploy --temporary` creates a deployment that lasts 60 minutes
and can be claimed into an account from the printed link.

## Run locally

```bash
python webapp/build_static.py && cd webapp/dist && python3 -m http.server 8000
# open http://localhost:8000
```

## Update the model

```bash
eponge export-web runs/<run>/best.pt webapp/model
python eponge/scripts/make_demo_mesh.py --ckpt runs/<run>/best.pt
python webapp/build_static.py
```
The page implements MillNet 2D (histogram FiLM, axial attention, factorised head) and the ResUNet in
TensorFlow.js; its output matches PyTorch to 5e-4 in probability. The larger zoo models run in the Python library.
