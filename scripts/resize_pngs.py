#!/usr/bin/env python3
from pathlib import Path
from PIL import Image

# === Config ===
INPUT_DIR = "data/synthetic_pfibs"   # your source directory
OUTPUT_DIR = "data/synthetic_pfibs_300"    # where resized images go
SIZE = (300, 300)
OVERWRITE = False  # set True if you want to replace original files

# === Script ===
input_path = Path(INPUT_DIR)
output_path = Path(OUTPUT_DIR)

if not OVERWRITE:
    output_path.mkdir(parents=True, exist_ok=True)

count = 0
for p in input_path.rglob("*.png"):
    try:
        img = Image.open(p).convert("L")  # grayscale; change to "RGB" if needed
        img_resized = img.resize(SIZE, Image.BILINEAR)

        if OVERWRITE:
            save_path = p
        else:
            # Mirror the original folder structure
            rel = p.relative_to(input_path)
            save_path = output_path / rel
            save_path.parent.mkdir(parents=True, exist_ok=True)

        img_resized.save(save_path)
        count += 1

    except Exception as e:
        print(f"❌ Failed to process {p}: {e}")

print(f"✅ Done! Resized {count} .png files to {SIZE}.")
