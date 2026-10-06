# Corrected annotations (v2)

The 116 masks (`slice_XXXX.png`: `0` pore, `255` solid, `128` outside the catalyst layer) are not
committed, like the other image data in this repository. Rebuild them in a few seconds:

```bash
eponge fix-annotations --stack data/raw/comp_full.tif --masks data/manual_annotation --out data/annotations_v2
```

`correction_log.json` (per frame) and `audit_summary.json` are the record of what changed.
The full explanation is in [docs/ANNOTATIONS.md](../../docs/ANNOTATIONS.md).
