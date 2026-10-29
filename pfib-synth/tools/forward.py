"""How visibly does buried material 'come forward'? For pore pixels at slice z,
correlate what they show now with the image k slices later at the best row
shift (the material's path). High = the thing seen through the pore is a
coherent preview of what gets cut later."""
import sys, numpy as np, tifffile
from scipy import ndimage as ndi
lab, p, y0, y1, z0, smax = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
with tifffile.TiffFile(p) as t: F = [ndi.gaussian_filter(t.pages[z0 + i].asarray().astype(np.float32), 1.5) for i in range(0, 30)]
res = {}
for k in (1, 2, 4, 8, 12):
    pc, sc = [], []
    for z in range(0, 30 - k, 3):
        A = F[z][y0:y1, 300:-300]; thr = np.percentile(A, 40); pore = A < thr; solid = A > np.percentile(A, 60)
        best_p, best_s = -1, -1
        for s in range(0, int(np.ceil(smax * k)) + 4):
            B = F[z + k][y0 + s:y1 + s, 300:-300]
            if B.shape != A.shape: continue
            best_p = max(best_p, np.corrcoef(A[pore], B[pore])[0, 1]); best_s = max(best_s, np.corrcoef(A[solid], B[solid])[0, 1])
        pc.append(best_p); sc.append(best_s)
    res[k] = (round(float(np.mean(pc)), 2), round(float(np.mean(sc)), 2))
print(f"{lab:10s} pore/solid corr with the image k slices later:", "  ".join(f"k{k}: {v[0]}/{v[1]}" for k, v in res.items()))
