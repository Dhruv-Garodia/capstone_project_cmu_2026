"""Print decoder_on_truth.json as per-material tables (mean and sample std over seeds) and write the summary.
    python experiments/tabulate_decoder.py experiments/decoder_on_truth.json experiments/millnet_on_synth_v7d.json experiments/decoder_on_truth_summary.json
"""
import json, sys, numpy as np
d = json.load(open(sys.argv[1])); old = json.load(open(sys.argv[2]))
P = ["comp_like", "uncomp_like", "pristine_like", "mudcrack", "fibre_mat", "fibre_coated", "faceted"]
M = ["miou", "pore_iou_true_layer", "layer_iou", "mae", "bias"]
seeds = ["0", "1", "2"]
for ref in ("truth", "label"):
    print(f"\n=== reference: {ref}")
    for dec in ("argmax", "calibrated", "truth_oracle"):
        print(f"-- {dec}")
        allm = {m: [] for m in M}
        for p in P + ["MEAN"]:
            cells = []
            for m in M:
                if p == "MEAN":
                    v = [np.mean([d["seeds"][s]["materials"][q][ref][dec][m] for q in P]) for s in seeds]
                else:
                    v = [d["seeds"][s]["materials"][p][ref][dec][m] for s in seeds]
                cells.append(f"{np.mean(v):.3f}±{np.std(v, ddof=1):.3f}" if m not in ("mae","bias") else f"{np.mean(v):+.2f}±{np.std(v, ddof=1):.2f}")
            print(f"  {p:<14} " + "  ".join(f"{m}={c}" for m, c in zip(M, cells)))
print("\nold argmax mIoU (score_millnet):", {p: round(np.mean([old[p][f'MillNet-SA2 s{s}']['miou_3class'] for s in range(3)]),3) for p in P})
for s in seeds: print(s, d["seeds"][s]["label_selected"], d["seeds"][s]["truth_oracle"], d["seeds"][s]["val_label_mae"])
summ = {"note": "MillNet-SA2 seeds 0-2 on the 7 v7d synthetic stacks (own renderer), test frames " + str(d["test_frames"]) + "; mean and sample std over seeds; mae/bias in pp, porosity in the predicted layer vs the reference layer",
        "decoders": {s: {"label_selected": d["seeds"][s]["label_selected"], "truth_oracle": d["seeds"][s]["truth_oracle"]} for s in seeds}, "table": {}}
for ref in ("truth", "label"):
    for dec in ("argmax", "calibrated", "truth_oracle"):
        for p in P + ["MEAN"]:
            for m in M:
                v = [np.mean([d["seeds"][s]["materials"][q][ref][dec][m] for q in P]) for s in seeds] if p == "MEAN" else [d["seeds"][s]["materials"][p][ref][dec][m] for s in seeds]
                summ["table"].setdefault(ref, {}).setdefault(dec, {}).setdefault(p, {})[m] = [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)]
json.dump(summ, open(sys.argv[3], "w"), indent=1)
