#!/usr/bin/env python3
"""Generate a synthetic pFIB-SEM stack with exact ground truth.

    python generate.py comp_like                          # a preset, 118 frames
    python generate.py fibre_mat --frames 40 --width 512  # quick look
    python generate.py --style comp --family spinodal --porosity 0.5 --name my_spinodal
    python generate.py faceted --mat crystal_px=40 --set jitter=[0.3,0.2] --set halo=[0.5,2]
    python generate.py --config examples/coarse_foam.json
    python generate.py --list                             # presets, families, render knobs
    python generate.py faceted --frames 60 --view         # then pack it for the viewer + showcase GIFs

Or run `python studio.py` for a browser form that does all of this.

A package holds the image stack (<name>_full.tif), the exact truth masks
(truth/masks/slice_NNNN.png: 0 pore, 255 solid, 128 outside the layer), start
masks from the threshold rule our labels came from, and stack.json / truth.json.
The masks use the team's 0/128/255 encoding (eponge.io.export_to_classes).

--style picks the microscope calibration (comp / uncomp / pristine: beam,
depth response, detector, layout of what is around the layer). --family picks
how the 3-D material is built. Any pair works; the presets are the seven
validated ones.
"""
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRESETS = {
    "comp_like":     dict(style="comp",     family="v5",       porosity=0.417, seed=20261040),
    "uncomp_like":   dict(style="uncomp",   family="v5",       porosity=0.558, seed=20261057),
    "pristine_like": dict(style="pristine", family="v5",       porosity=0.515, seed=20261074),
    "mudcrack":      dict(style="comp",     family="mudcrack", porosity=0.45,  seed=20261091),
    "fibre_mat":     dict(style="uncomp",   family="fibre",    porosity=0.62,  seed=20261108,
                          mat_kw={"r_med": 5.0, "rim": 1.5, "thick_frac": 0.12}),
    "fibre_coated":  dict(style="uncomp",   family="fibre",    porosity=0.60,  seed=20261125,
                          mat_kw={"r_med": 4.0, "rim": 1.2, "decorate": 2.0, "binder": 0.6}),
    "faceted":       dict(style="pristine", family="faceted",  porosity=0.22,  seed=20261142),
}
STYLES = ["comp", "uncomp", "pristine"]
# render settings worth turning (keys of slicing.json "v7"); anything else can still be --set
KNOBS = {
    "zlen":           "slices per pixel-width of material (slice thickness); higher = structure persists longer",
    "edge_dy":        "how much more a pore's top edge glows than its bottom edge (the 52 deg beam), px",
    "halo":           "[strength, width px] edge blooming just inside pore rims",
    "jitter":         "[y, x] per-frame stage jitter, px",
    "curtain_illum":  "curtaining stripe strength",
    "curtain_regional": "[strength, change per slice] curtaining that evolves regionally",
    "micropore":      "fraction of solid turned into 1-3 px micropores (catalyst families)",
    "macro_per_kpx":  "large voids per 1000 px of frame width (catalyst families)",
    "macro_r":        "[min, max] radius of those voids, px",
    "sat_spread":     "how far the detector saturates past 255",
    "noise_sig":      "[y, x] correlation of detector noise",
    "approach":       "[gain, slices] how a buried surface brightens as the cut approaches it",
    "ctx3d":          "3-D voids and caverns around the layer (true/false)",
}


def families():
    sys.path.insert(0, str(HERE / "pfibsynth"))
    import inspect, materials
    out = {"v5": "Voronoi foam catalyst layer, calibrated per style (voro settings in slicing.json)"}
    for k, f in materials.FAMILIES.items():
        ps = [f"{p.name}={p.default}" for p in list(inspect.signature(f).parameters.values())[3:] if p.default is not p.empty]
        out[k] = (f.__doc__ or "").strip().split("\n")[0] + ("  [" + ", ".join(ps) + "]" if ps else "")
    return out


def kv(items):
    out = {}
    for s in items or []:
        k, _, v = s.partition("=")
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def resolve(a):
    """Merge preset < config file < command line into one job description."""
    cfg = {}
    if a.config:
        cfg = json.loads(Path(a.config).read_text())
    preset = a.material or cfg.get("material")
    job = dict(PRESETS.get(preset, {}))
    for k, v in cfg.items():
        if k in ("mat_kw", "override"):
            job[k] = {**job.get(k, {}), **v}           # config adds to the preset's settings
        elif k != "material" and not k.startswith("_"):
            job[k] = v
    for k in ("style", "family", "porosity", "seed", "frames", "width", "name", "out"):
        v = getattr(a, k)
        if v is not None:
            job[k] = v
    job["mat_kw"] = {**job.get("mat_kw", {}), **kv(a.mat)}
    job["override"] = {**job.get("override", {}), **kv(a.set)}
    job.setdefault("frames", 118)
    job.setdefault("seed", 20261027)
    job["name"] = job.get("name") or preset or f"{job.get('family', 'v5')}_{job.get('style', 'comp')}"
    if "style" not in job:
        raise SystemExit("give a preset (python generate.py --list) or --style")
    job.setdefault("family", "v5")
    job.setdefault("porosity", {"comp": 0.417, "uncomp": 0.558, "pristine": 0.515}[job["style"]])
    job["out"] = str(job.get("out") or HERE / "out" / job["name"])
    return job


def command(job):
    cmd = [sys.executable, str(HERE / "pfibsynth" / "make_full.py"), "--style", job["style"], "--family", job["family"],
           "--porosity", str(job["porosity"]), "--seed", str(job["seed"]), "--frames", str(job["frames"]),
           "--name", f"synth_{job['name']}", "--out", job["out"]]
    if job.get("width"):
        cmd += ["--width", str(job["width"])]
    if job.get("mat_kw"):
        cmd += ["--mat-kw", json.dumps(job["mat_kw"])]
    if job.get("override"):
        cmd += ["--override", json.dumps(job["override"])]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("material", nargs="?", choices=list(PRESETS), help="a preset")
    ap.add_argument("--list", action="store_true", help="show presets, material families and render knobs")
    ap.add_argument("--config", help="JSON file with any of: material, style, family, porosity, seed, frames, width, name, mat_kw, override")
    ap.add_argument("--style", choices=STYLES, default=None, help="microscope calibration")
    ap.add_argument("--family", default=None, help="material builder (see --list)")
    ap.add_argument("--porosity", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None, help="slices (real stacks: comp 354, uncomp 410, pristine 665)")
    ap.add_argument("--width", type=int, default=None, help="narrower frames for a quick look")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None, help="output folder (default out/<name>)")
    ap.add_argument("--mat", action="append", metavar="KEY=VALUE", help="material builder argument, repeatable")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="render setting override, repeatable (see --list)")
    ap.add_argument("--view", action="store_true", help="afterwards: pack for the viewer and render the showcase GIFs")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved job and command only")
    a = ap.parse_args()
    if a.list or not (a.material or a.config or a.style):
        print("presets:")
        for k, v in PRESETS.items():
            print(f"  {k:<14} style {v['style']:<9} family {v['family']:<9} porosity {v['porosity']}" + (f"  mat {v['mat_kw']}" if "mat_kw" in v else ""))
        print("\nfamilies (--family, builder arguments via --mat):")
        for k, v in families().items():
            print(f"  {k:<12} {v}")
        print("\nrender knobs (--set KEY=VALUE; current values per style in pfibsynth/slicing.json):")
        for k, v in KNOBS.items():
            print(f"  {k:<17} {v}")
        return
    job = resolve(a)
    cmd = command(job)
    print(json.dumps(job, indent=1))
    if a.dry_run:
        print(" ".join(cmd)); return
    rc = subprocess.call(cmd)
    if rc == 0:
        Path(job["out"], "job.json").write_text(json.dumps(job, indent=1))   # rerun with --config out/<name>/job.json
    if rc == 0 and a.view:
        rc = subprocess.call([sys.executable, str(HERE / "tools" / "export_viewer.py"), job["out"]])
        rc = rc or subprocess.call([sys.executable, str(HERE / "tools" / "showcase.py"), job["out"], str(Path(job["out"]) / "gifs"), "--label", job["name"]])
    sys.exit(rc)


if __name__ == "__main__":
    main()
