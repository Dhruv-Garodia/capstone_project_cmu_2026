"""fig: MillNet-SA2 (trained on real labels) vs the threshold rule and synthetic-trained U-Nets, scored on exact truth.
    python experiments/plot_millnet.py experiments/millnet_on_synth_v7d.json docs/media/millnet_on_truth.png
"""
import sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

r = json.load(open(sys.argv[1])); pk = list(r)
M = {"threshold rule\n(our labels)": ["rule (start masks)"], "MillNet-SA2\n(real labels, 3 seeds)": [f"MillNet-SA2 s{s}" for s in range(3)],
     "1-slice U-Net\n(synthetic)": ["2d_random_v7_s0"], "9-slice U-Net\n(synthetic)": ["zwin_random_v7_s0"]}
C = ["#9a9a9a", "#c0392b", "#7fa7d6", "#1f4e8c"]
B = [("1", "1"), ("2-3", "2-3"), ("4-8", "4-8"), ("none<=8", ">8")]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
for j, (lab, ks) in enumerate(M.items()):
    v = [np.mean([r[p][k][f"recall_{b}"] for p in pk for k in ks]) for b, _ in B]
    ax[0].bar(np.arange(4) + (j - 1.5) * 0.2, v, 0.2, color=C[j], label=lab)
ax[0].set_xticks(range(4), [t for _, t in B]); ax[0].set_xlabel("slices down to the pore's lit floor (1 = strongest shine-through)", fontsize=8)
ax[0].set_ylabel("recall of true pore pixels"); ax[0].set_ylim(0, 1.32); ax[0].set_title("Shine-through: pores whose lit floor is close", fontsize=10)
ax[0].legend(fontsize=7, loc="upper left", frameon=False)
for j, (lab, ks) in enumerate(M.items()):
    v = [np.mean([r[p][k]["porosity_error_pp"] for k in ks]) for p in pk]
    e = [np.std([r[p][k]["porosity_error_pp"] for k in ks]) for p in pk]
    ax[1].bar(np.arange(len(pk)) + (j - 1.5) * 0.2, v, 0.2, yerr=e if len(ks) > 1 else None, color=C[j], capsize=2)
ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xticks(range(len(pk)), [p.replace("_", " ") for p in pk], fontsize=8)
ax[1].set_ylabel("porosity error vs truth, pp"); ax[1].set_title("Porosity error on 7 synthetic materials (last 4 never seen in training)", fontsize=10)
ax[1].set_ylim(-16, 38)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130); print(sys.argv[2])
