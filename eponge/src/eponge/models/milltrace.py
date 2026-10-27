"""MillTrace: serial sectioning read as a video, with slide-warped recurrent memory.

MillNet engineers features from a fixed window. MillTrace instead follows the milling sequence:

1. A shared 2D encoder turns every slice of a 2k+1 window (default k = 7, 15 slices) into features at
   1/2, 1/4, 1/8 and 1/16 resolution.
2. A learned slide field s(x, y) (px/slice, the same SlideField as MillNet-SA) gives the rate at which
   sub-surface material moves in the image as the blade advances.
3. At 1/4, 1/8 and 1/16 resolution two ConvGRUs run along z: one from the oldest slice up to the centre,
   one from the newest slice down to the centre. Before every step the hidden state is warped by the
   slide field, so the memory of a piece of material stays on that material while it slides towards
   the cut face. The centre slice then knows what was seen before it was cut and what is still to come
   (shine-through: material visible through a pore is cut later).
4. A U-Net decoder over the fused centre features, a full-resolution path from the centre slice and its
   two neighbours, and the factorised head (P(outside), P(pore | inside)).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import ResBlock
from .zoo import SlideField


def warp_feat(h: torch.Tensor, s: torch.Tensor, d: float) -> torch.Tensor:
    """Sample (B, C, H, W) features at rows y + s*d (sub-pixel, edge-clamped); s is (B, 1, H, W)."""
    b, c, hh, ww = h.shape
    ys = (torch.arange(hh, device=h.device, dtype=h.dtype).view(1, 1, hh, 1) + s * d).clamp(0, hh - 1)
    y0 = ys.floor()
    t = ys - y0
    y0 = y0.long()
    y1 = (y0 + 1).clamp(max=hh - 1)
    y0, y1, t = y0.expand(b, c, hh, ww), y1.expand(b, c, hh, ww), t.expand(b, c, hh, ww)
    return torch.gather(h, 2, y0) * (1 - t) + torch.gather(h, 2, y1) * t


class ConvGRU(nn.Module):
    def __init__(self, cin: int, ch: int):
        super().__init__()
        self.zr = nn.Conv2d(cin + ch, 2 * ch, 3, padding=1)
        self.n = nn.Conv2d(cin + ch, ch, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.n(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class TraceFusion(nn.Module):
    """Bidirectional slide-warped ConvGRU along z, fused into the centre slice's features."""

    def __init__(self, ch: int, hid: int):
        super().__init__()
        self.inp = nn.Conv2d(ch, hid, 1)                  # compact memory: GRUs run at hid channels
        self.fwd, self.bwd = ConvGRU(hid, hid), ConvGRU(hid, hid)
        self.hid = hid
        self.out = nn.Sequential(nn.Conv2d(2 * hid + ch, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

    def forward(self, f, s):
        """f: (B, T, C, h, w) per-slice features, s: (B, 1, h, w) slide in px/slice at this resolution."""
        b, t, c, hh, ww = f.shape
        k = t // 2
        g = self.inp(f.reshape(b * t, c, hh, ww)).reshape(b, t, self.hid, hh, ww)
        hf = f.new_zeros(b, self.hid, hh, ww)
        for i in range(k + 1):                        # oldest -> centre; move memory one slice forward first
            hf = self.fwd(g[:, i], warp_feat(hf, s, -1.0) if i else hf)
        hb = f.new_zeros(b, self.hid, hh, ww)
        for i in range(t - 1, k - 1, -1):             # newest -> centre
            hb = self.bwd(g[:, i], warp_feat(hb, s, 1.0) if i < t - 1 else hb)
        return f[:, k] + self.out(torch.cat([hf, hb, f[:, k]], 1))


class MillTrace(nn.Module):
    needs_stats = False

    def __init__(self, in_channels: int = 15, num_classes: int = 3, chs=(16, 24, 32, 64, 128), hid=(16, 32, 64),
                 slide_max: float = 1.5, dropout: float = 0.1, trace_levels=(2, 3, 4), aligned_full: bool = False):
        super().__init__()
        assert num_classes == 3 and in_channels >= 3
        self.k = (in_channels - 1) // 2
        self.multiple, self.infer_tile = 16, 512
        self.learned_slide = True
        self.slide = SlideField(in_channels, slide_max)
        c0, c1, c2, c3, c4 = chs
        # full-resolution path. v1: centre +- 1 raw. v2 (aligned_full): centre, both neighbours warped by the learned
        # slide, forward/backward differences and the future maximum, so the slide field gets a full-resolution signal
        self.aligned_full = aligned_full
        self.full = nn.Sequential(ResBlock(6, c0), ResBlock(c0, c0)) if aligned_full else ResBlock(3, c0)
        self.e1 = nn.Sequential(nn.Conv2d(1, c1, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c1),
                                nn.ReLU(inplace=True), ResBlock(c1, c1))       # 1/2
        self.e2, self.e3, self.e4 = ResBlock(c1, c2), ResBlock(c2, c3), ResBlock(c3, c4)
        self.trace_levels = tuple(trace_levels)
        self.trace = nn.ModuleDict({str(l): TraceFusion(c, h) for l, c, h in zip((2, 3, 4), (c2, c3, c4), hid) if l in trace_levels})
        self.d3 = ResBlock(c4 + c3, c3, dropout)
        self.d2 = ResBlock(c3 + c2, c2, dropout)
        self.d1 = ResBlock(c2 + c1, c1)
        self.d0 = ResBlock(c1 + c0, c0)
        self.head_roi, self.head_pore = nn.Conv2d(c0, 1, 1), nn.Conv2d(c0, 1, 1)

    def forward(self, x):
        b, t, hh, ww = x.shape
        k = self.k
        s = self.slide(x)                                                      # (B, 1, H, W) px/slice
        self.last_slide = s
        f1 = self.e1(x.reshape(b * t, 1, hh, ww))
        f2 = self.e2(F.max_pool2d(f1, 2))
        f3 = self.e3(F.max_pool2d(f2, 2))
        f4 = self.e4(F.max_pool2d(f3, 2))
        feats = {}
        for lvl, f in ((2, f2), (3, f3), (4, f4)):
            f = f.reshape(b, t, *f.shape[1:])
            if str(lvl) in self.trace:
                sc = 2 ** lvl
                sl = F.avg_pool2d(s, sc) / sc                                  # px/slice at this resolution
                feats[lvl] = self.trace[str(lvl)](f, sl)
            else:
                feats[lvl] = f[:, k]
        c1 = f1.reshape(b, t, *f1.shape[1:])[:, k]
        y = feats[4]
        for blk, skip in ((self.d3, feats[3]), (self.d2, feats[2]), (self.d1, c1)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, skip], 1))
        y = F.interpolate(y, size=(hh, ww), mode="bilinear", align_corners=False)
        if self.aligned_full:
            c = x[:, k:k + 1]
            fut = [warp_feat(x[:, k + d:k + d + 1], s, float(d)) for d in range(1, k + 1)]
            past = warp_feat(x[:, k - 1:k], s, -1.0)
            fi = torch.cat([c, past, fut[0], fut[0] - c, c - past, torch.cat(fut, 1).amax(1, keepdim=True) - c], 1)
        else:
            fi = x[:, k - 1:k + 2]
        y = self.d0(torch.cat([y, self.full(fi)], 1))
        r, u = self.head_roi(y), self.head_pore(y)
        log_in = F.logsigmoid(-r)
        return torch.cat([log_in + F.logsigmoid(-u), log_in + F.logsigmoid(u), F.logsigmoid(r)], 1)
