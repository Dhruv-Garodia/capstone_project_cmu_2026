"""Model zoo: every architecture compared in this project, behind one ``build_model(config)`` call.

Each model maps (B, C, H, W) with C = 2k + 1 adjacent slices to (B, 3, H, W) logits
(solid, pore, outside the catalyst layer) and exposes ``multiple``: H and W must be
padded to a multiple of it at inference (``inference.predict_probs`` does this).

=================  ==========================================================================
``resunet``        Éponge residual U-Net (this project), 2.4 M params, browser-deployable.
``unet_garodia``   D. Garodia's 2.5D U-Net (pore_pipeline/unet/model.py), 3-class head.
``transunet``      TransUNet (Chen et al. 2021, arXiv:2102.04306): R50 + ViT-B/16 hybrid
                   encoder pretrained on ImageNet-21k, cascaded upsampler (CUP) with skips.
``segformer``      SegFormer (Xie et al. 2021) with a MiT-B2 encoder (ImageNet), all-MLP decoder.
``unetpp_r34``     UNet++ (Zhou et al. 2018) with an ImageNet ResNet-34 encoder.
``millnet``        MillNet (this project): physics-informed 2.5D network for serial sections; see the class.
``milltrace``      MillTrace (this project): the stack read as a video, slide-warped bidirectional ConvGRU memory.
=================  ==========================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet as ResUNet


# ------------------------------------------------------------------------- Garodia U-Net
def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class GarodiaUNet(nn.Module):
    """Architecture copied from pore_pipeline/unet/model.py; only the head is 3-class here."""

    def __init__(self, in_channels: int, num_classes: int = 3, base: int = 32, depth: int = 4):
        super().__init__()
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList(_block(cin, cout) for cin, cout in zip([in_channels] + widths[:-1], widths))
        self.up = nn.ModuleList(_block(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(base, num_classes, 1)
        self.multiple = 2 ** depth

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()
        for block in self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skips.pop()], 1))
        return self.head(x)


# ----------------------------------------------------------------------------- TransUNet
class _ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _CUPBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(_ConvBNReLU(cin + cskip, cout), _ConvBNReLU(cout, cout))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class TransUNet(nn.Module):
    """TransUNet with the R50-ViT-B/16 hybrid encoder (the configuration of the paper).

    Encoder: ResNet-50 (v2, GN + weight-standardised convs) root and first three stages give
    skips at 1/2 (64 ch), 1/4 (256 ch) and 1/8 (512 ch); stage-3 features (1/16) are projected
    to 768-d tokens and passed through 12 ViT-B transformer layers. Decoder (CUP): reshape the
    tokens to a 1/16 map, 3x3 conv to 512, then four (upsample x2 -> concat skip -> 2 convs)
    blocks with 256, 128, 64, 16 channels, and a 1x1 head. Weights: ImageNet-21k
    (timm ``vit_base_r50_s16_224.orig_in21k``), positional embeddings resized to the input.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, pretrained: bool = True, img_size: int = 256):
        super().__init__()
        import timm
        self.vit = timm.create_model("vit_base_r50_s16_224.orig_in21k", pretrained=pretrained, in_chans=in_channels,
                                     img_size=img_size, dynamic_img_size=True, num_classes=0)
        for prm in self.vit.parameters():  # resized pretrained tensors (pos-embed, 1-channel stem) can be
            prm.data = prm.data.contiguous()  # non-contiguous views, which break MPS backward
        self.multiple = 16
        self.infer_tile = img_size  # ViT position embeddings were learned at this size: infer on tiles of it
        self.conv_more = _ConvBNReLU(768, 512)
        self.dec = nn.ModuleList([_CUPBlock(512, 512, 256), _CUPBlock(256, 256, 128),
                                  _CUPBlock(128, 64, 64), _CUPBlock(64, 0, 16)])
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        bb = self.vit.patch_embed.backbone
        stem = bb.stem
        s2 = stem.norm(stem.conv(x)) if hasattr(stem, "norm") else stem.conv(x)  # 1/2, 64 ch
        y = stem.pool(s2)
        skips = [s2]
        for i, stage in enumerate(bb.stages):
            y = stage(y)
            if i < len(bb.stages) - 1:
                skips.append(y)            # 1/4 (256 ch), 1/8 (512 ch)
        if hasattr(bb, "norm"):
            y = bb.norm(y)
        tok = self.vit.patch_embed.proj(y)  # (B, 768, H/16, W/16)
        gh, gw = tok.shape[-2:]
        tok = tok.permute(0, 2, 3, 1).contiguous()  # (B, gh, gw, 768) for dynamic-size pos-embed
        tok = self.vit._pos_embed(tok)
        tok = self.vit.norm_pre(tok)
        tok = self.vit.blocks(tok)
        tok = self.vit.norm(tok)
        tok = tok[:, self.vit.num_prefix_tokens:]
        y = tok.transpose(1, 2).contiguous().reshape(tok.shape[0], 768, gh, gw)
        y = self.conv_more(y)
        y = self.dec[0](y, skips[2])
        y = self.dec[1](y, skips[1])
        y = self.dec[2](y, skips[0])
        y = self.dec[3](y)
        y = self.head(y)
        return y if y.shape[-2:] == (H, W) else y[..., :H, :W].contiguous()


# ------------------------------------------------------------------------------ smp models
class _SMPWrap(nn.Module):
    def __init__(self, net, multiple):
        super().__init__()
        self.net, self.multiple = net, multiple
        for prm in self.net.parameters():  # 1-channel adapted stems can be non-contiguous (breaks MPS backward)
            prm.data = prm.data.contiguous()

    def forward(self, x):
        return self.net(x)


def _patch_segformer_mlp():
    """smp's SegFormer MLP returns a non-contiguous view; F.interpolate's MPS backward rejects it."""
    from segmentation_models_pytorch.decoders.segformer import decoder as d
    if getattr(d.MLP, "_eponge_patched", False):
        return

    def forward(self, x):
        b, _, h, w = x.shape
        x = self.linear(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).contiguous().reshape(b, -1, h, w)

    d.MLP.forward, d.MLP._eponge_patched = forward, True


def _smp(kind: str, in_channels: int, num_classes: int, pretrained: bool):
    import segmentation_models_pytorch as smp
    w = "imagenet" if pretrained else None
    if kind == "segformer":
        _patch_segformer_mlp()
        net = smp.Segformer(encoder_name="mit_b2", encoder_weights=w, in_channels=in_channels, classes=num_classes)
    elif kind == "unetpp_r34":
        net = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=w, in_channels=in_channels, classes=num_classes)
    else:
        raise ValueError(kind)
    return _SMPWrap(net, 32)


# ------------------------------------------------------------------------------- factory
ARCHS = ("resunet", "unet_garodia", "transunet", "segformer", "unetpp_r34", "millnet", "zafunet", "unet_r34", "milltrace")


def build_model(config: dict) -> nn.Module:
    """``config``: {"arch": ..., "in_channels": ..., "num_classes": 3, ...arch kwargs}.

    Configs without "arch" (checkpoints from before the zoo) build the Éponge ResUNet.
    """
    cfg = dict(config)
    arch = cfg.pop("arch", "resunet")
    pretrained = cfg.pop("pretrained", True)
    cin, nc = cfg.pop("in_channels", 1), cfg.pop("num_classes", 3)
    if arch == "resunet":
        m = ResUNet(cin, nc, **cfg)
    elif arch == "unet_garodia":
        m = GarodiaUNet(cin, nc, **cfg)
    elif arch == "transunet":
        m = TransUNet(cin, nc, pretrained=pretrained, **cfg)
    elif arch in ("segformer", "unetpp_r34"):
        m = _smp(arch, cin, nc, pretrained)
    elif arch == "millnet":
        m = MillNet(cin, nc, **cfg)
    elif arch == "zafunet":
        m = ZAFUNet(cin, nc, pretrained=pretrained, **cfg)
    elif arch == "milltrace":
        from .milltrace import MillTrace
        m = MillTrace(cin, nc, **cfg)
    elif arch == "unet_r34":               # standard 2.5D baseline: slices as input channels, ImageNet ResNet-34
        import segmentation_models_pytorch as smp
        m = _SMPWrap(smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=cin, classes=nc), 32)
    else:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
    m.config = {"arch": arch, "in_channels": cin, "num_classes": nc, **cfg}
    return m


# ------------------------------------------------------------------------------- MillNet
class _AxialAttention(nn.Module):
    """Row attention then column attention (Ho et al. 2019 / Wang et al. 2020): global context
    along the catalyst layer (x) and through it (y) at O(HW(H+W)) cost."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.row = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.col = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.permute(0, 2, 3, 1).reshape(b * h, w, c).contiguous()
        q = self.n1(t)
        t = t + self.row(q, q, q, need_weights=False)[0]
        t = t.reshape(b, h, w, c).permute(0, 2, 1, 3).reshape(b * w, h, c).contiguous()
        q = self.n2(t)
        t = t + self.col(q, q, q, need_weights=False)[0]
        return t.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous()


def _shift_y(img: torch.Tensor, s: int) -> torch.Tensor:
    """Shift (B, H, W) by s rows (positive = down), replicating the edge row."""
    if s == 0:
        return img
    if s > 0:
        return torch.cat([img[:, :1].expand(-1, s, -1), img[:, :-s]], 1)
    return torch.cat([img[:, -s:], img[:, -1:].expand(-1, -s, -1)], 1)


class MillNet(nn.Module):
    """Physics-informed 2.5D segmentation network for FIB-SEM serial sections.

    1. Streak-hypothesis bank: neighbour slice z+d is shifted by s*d rows for s in ``streaks``
       (default -1, 0, +1 px/slice), the ways sub-surface material can slide behind a stationary cut face.
       Hypothesis s compensates material that moves by -s rows per slice (s = -1: sliding down).
    2. Milling-evolution channels per hypothesis: forward difference (future - now), backward
       difference (now - past) and future maximum - now (Salzer's last-occurrence idea).
    3. Global histogram conditioning: FiLM from ``n_quant`` quantiles of the whole frame, so a
       crop knows its frame's grey-level distribution (the labels use a per-frame threshold).
    4. Axial attention at the bottleneck for long-range context along and through the layer.
    5. Factorised head: P(outside) and P(pore | inside), composed into exact 3-class log-probabilities.
    """

    needs_stats = True

    def __init__(self, in_channels: int = 7, num_classes: int = 3, base: int = 24, depth: int = 5,
                 max_ch: int = 128, dropout: float = 0.1, streaks=(-1, 0, 1), n_quant: int = 16,
                 use_bank: bool = True, use_stats: bool = True, use_axial: bool = True,
                 learned_slide: bool = False, slide_max: float = 2.5, factor_head: bool = True,
                 use_milling: bool = True):
        super().__init__()
        assert num_classes == 3
        self.k = (in_channels - 1) // 2
        self.streaks = tuple(streaks) if (use_bank and self.k > 0) else (0,)
        self.use_stats, self.use_axial, self.n_quant = use_stats, use_axial, n_quant
        self.learned_slide = learned_slide and self.k > 0
        self.use_milling = use_milling
        per = (2 * self.k + (3 if use_milling else 0)) if self.k > 0 else 0
        cin = 1 + per * len(self.streaks) + ((per + 1) if self.learned_slide else 0)
        if self.learned_slide:
            self.slide = SlideField(2 * self.k + 1, slide_max)
        self.depth, self.multiple = depth, 2 ** depth
        chs = [min(base * 2 ** i, max_ch) for i in range(depth + 1)]
        from .unet import ResBlock
        self.stem = ResBlock(cin, chs[0])
        self.down = nn.ModuleList([ResBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([ResBlock(chs[i + 1] + chs[i], chs[i], dropout=dropout if i >= 1 else 0.0)
                                 for i in reversed(range(depth))])
        self.axial = _AxialAttention(chs[-1]) if use_axial else nn.Identity()
        if use_stats:
            self.film = nn.Sequential(nn.Linear(n_quant, 64), nn.GELU(), nn.Linear(64, 2 * (chs[0] + chs[-1])))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        self.chs = chs
        self.factor_head = factor_head
        if factor_head:
            self.head_roi = nn.Conv2d(chs[0], 1, 1)
            self.head_pore = nn.Conv2d(chs[0], 1, 1)
        else:                                               # ablation: plain 3-class softmax head
            self.head = nn.Conv2d(chs[0], 3, 1)

    def features_in(self, x):
        k = self.k
        c = x[:, k]
        feats = [c[:, None]]
        if k == 0:
            return torch.cat(feats, 1)
        for s in self.streaks:
            past = [_shift_y(x[:, k - d], -s * d) for d in range(1, k + 1)]
            fut = [_shift_y(x[:, k + d], s * d) for d in range(1, k + 1)]
            fwd = fut[0] - c
            bwd = c - past[0]
            fmax = torch.stack(fut, 1).amax(1) - c
            feats += ([torch.stack(past + fut, 1), fwd[:, None], bwd[:, None], fmax[:, None]] if self.use_milling
                      else [torch.stack(past + fut, 1)])                 # ablation: raw neighbour slices only
        if self.learned_slide:
            s = self.slide(x)                                   # (B, 1, H, W) px per slice
            self.last_slide = s
            past = [warp_rows(x[:, k - d], s, -d) for d in range(1, k + 1)]
            fut = [warp_rows(x[:, k + d], s, d) for d in range(1, k + 1)]
            fmax = torch.stack(fut, 1).amax(1) - c
            feats += [torch.stack(past + fut, 1), (fut[0] - c)[:, None], (c - past[0])[:, None], fmax[:, None],
                      s / self.slide.slide_max]
        return torch.cat(feats, 1)

    def forward(self, x, stats=None):
        h = self.stem(self.features_in(x))
        if self.use_stats:
            if stats is None:
                stats = frame_quantiles(x[:, self.k], self.n_quant)
            g = self.film(stats)
            c0, cl = self.chs[0], self.chs[-1]
            g0, b0, gl, bl = torch.split(g, [c0, c0, cl, cl], 1)
            h = h * (1 + g0[:, :, None, None]) + b0[:, :, None, None]
        skips = [h]
        for blk in self.down:
            skips.append(blk(F.max_pool2d(skips[-1], 2)))
        y = skips.pop()
        y = self.axial(y)
        if self.use_stats:
            y = y * (1 + gl[:, :, None, None]) + bl[:, :, None, None]
        for blk in self.up:
            s = skips.pop()
            y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, s], 1))
        if not self.factor_head:
            return F.log_softmax(self.head(y), 1)
        r, u = self.head_roi(y), self.head_pore(y)
        log_in = F.logsigmoid(-r)
        # class order (solid, pore, outside); exact log-probabilities, so softmax() returns them unchanged
        return torch.cat([log_in + F.logsigmoid(-u), log_in + F.logsigmoid(u), F.logsigmoid(r)], 1)


class SlideField(nn.Module):
    """Learned sub-surface slide rate s(x, y) in px/slice from the 2k+1 slice stack.

    The fixed streak bank tests three hypotheses (-1, 0, +1 px/slice); the real stacks slide at
    ~1.2 (comp_full), ~1.15 (uncomp_full) and ~0 (pristine_full) px/slice, and the rate can vary across
    a frame. A small network at 1/4 resolution predicts a smooth continuous field, bounded to
    +-slide_max, which warp_rows uses to re-align each neighbour slice with sub-pixel accuracy. The last
    layer starts at zero, so training starts from "no slide" and the rate is learned end to end; the
    field is returned for inspection (it should match the independently measured slide rates)."""

    def __init__(self, cin: int, slide_max: float = 2.5, width: int = 32):
        super().__init__()
        self.slide_max = slide_max
        self.net = nn.Sequential(
            nn.Conv2d(cin, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        lo = F.avg_pool2d(x, 4, ceil_mode=True)
        s = self.slide_max * torch.tanh(self.net(lo))
        s = F.avg_pool2d(s, 5, stride=1, padding=2, count_include_pad=False)   # smooth field
        return F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)


def warp_rows(img: torch.Tensor, s: torch.Tensor, d: int) -> torch.Tensor:
    """Sample (B, H, W) slice at rows y + s*d (sub-pixel, edge-clamped): undo a slide of s px/slice over d slices."""
    b, h, w = img.shape
    ys = (torch.arange(h, device=img.device, dtype=img.dtype).view(1, h, 1) + s[:, 0] * d).clamp(0, h - 1)
    y0 = ys.floor()
    t = ys - y0                                              # linear interpolation along y (gather is MPS-safe)
    y0 = y0.long()
    y1 = (y0 + 1).clamp(max=h - 1)
    return torch.gather(img, 1, y0) * (1 - t) + torch.gather(img, 1, y1) * t


def frame_quantiles(frames: torch.Tensor, n: int = 16) -> torch.Tensor:
    """(B, H, W) normalised frames -> (B, n) quantiles of each whole frame (subsampled for speed)."""
    flat = frames.flatten(1)
    if flat.shape[1] > 65536:
        flat = flat[:, :: flat.shape[1] // 65536]
    qs = torch.linspace(0.02, 0.98, n, device=frames.device, dtype=frames.dtype)
    return torch.quantile(flat.float().cpu(), qs.float().cpu(), dim=1).T.to(frames.device, frames.dtype)


# ------------------------------------------------------------------------------- ZAF-UNet
class ZAttention(nn.Module):
    """Fuse per-slice features (B, Z, C, h, w) into the centre slice's features (B, C, h, w).

    Every pixel of the centre slice attends over the 2k+1 slices. Keys and values come from a shared
    3x3 convolution, so each slice token carries its local neighbourhood (material that slides ~1 px per
    slice behind a pore stays within reach); learned slice-position embeddings tell the attention which
    slice is past, present or future (milling makes the future brighten, not the past)."""

    def __init__(self, ch: int, z: int, heads: int = 4):
        super().__init__()
        self.heads = heads if ch % heads == 0 else 1
        self.local = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pos = nn.Parameter(torch.zeros(z, ch))
        nn.init.normal_(self.pos, std=0.02)
        self.q, self.k, self.v = nn.Linear(ch, ch), nn.Linear(ch, ch), nn.Linear(ch, ch)
        self.out = nn.Linear(ch, ch)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)     # starts as the centre slice
        self.norm = nn.LayerNorm(ch)

    def forward(self, f):
        b, z, c, h, w = f.shape
        loc = self.local(f.reshape(b * z, c, h, w)).reshape(b, z, c, h, w)
        t = (f + loc).permute(0, 3, 4, 1, 2).reshape(b * h * w, z, c) + self.pos     # tokens over z
        t = self.norm(t)
        centre = f[:, z // 2].permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        nh, dh = self.heads, c // self.heads
        q = self.q(self.norm(centre)).view(-1, 1, nh, dh).transpose(1, 2)
        k = self.k(t).view(-1, z, nh, dh).transpose(1, 2)
        v = self.v(t).view(-1, z, nh, dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)                            # (bhw, nh, 1, dh)
        y = centre + self.out(a.transpose(1, 2).reshape(-1, 1, c))
        return y.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False) if skip is not None else \
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1) if skip is not None else x)


class ZAFUNet(nn.Module):
    """Z-Attention Fusion U-Net for FIB-SEM serial sections.

    1. Shared 2D encoder (ImageNet-pretrained, timm) applied to each of the 2k+1 slices separately.
    2. ZAttention at every encoder scale fuses the slices into the centre slice's features.
    3. 2D U-Net decoder on the fused multi-scale features -> solid / pore / outside.
    """

    def __init__(self, in_channels: int = 7, num_classes: int = 3, encoder: str = "resnet34",
                 pretrained: bool = True, dec=(256, 128, 64, 32, 16), fusion: str = "zattn", freeze_bn: bool = True):
        super().__init__()
        import timm
        self.z = in_channels
        self.enc = timm.create_model(encoder, features_only=True, pretrained=pretrained, in_chans=1)
        # Pretrained encoder BatchNorm is frozen (running statistics and affine): a batch holds 2k+1
        # near-identical slices of the same crop, so batch statistics are strongly correlated and the
        # running averages drift away from what the network sees at test time.
        self.freeze_bn = freeze_bn and pretrained
        if self.freeze_bn:
            for m in self.enc.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    for prm in m.parameters():
                        prm.requires_grad_(False)
        chs = self.enc.feature_info.channels()
        # fusion ablations: "zattn" (ours), "mean" (average the slices' features), "centre" (ignore neighbours)
        self.fusion = fusion
        self.fuse = nn.ModuleList([ZAttention(c, in_channels) if (in_channels > 1 and fusion == "zattn") else nn.Identity()
                                   for c in chs])
        skips = chs[::-1][1:] + [0] * (len(dec) - len(chs) + 1)
        ups, cin = [], chs[-1]
        for i, d in enumerate(dec):
            cs = skips[i] if i < len(skips) else 0
            ups.append(_Up(cin, cs, d)); cin = d
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(dec[-1], num_classes, 1)
        self.multiple = 32
        self.infer_tile = 512                 # per-slice encoding x 7 slices: keep inference tiles small
        self.register_buffer("mean", torch.tensor(0.45)); self.register_buffer("std", torch.tensor(0.225))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.enc.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x):
        b, z, h, w = x.shape
        feats = self.enc(((x - self.mean) / self.std).reshape(b * z, 1, h, w))
        fused = []
        for f, fu in zip(feats, self.fuse):
            f = f.view(b, z, *f.shape[1:])
            fused.append(f[:, 0] if z == 1 else fu(f) if self.fusion == "zattn" else f.mean(1) if self.fusion == "mean"
                         else f[:, z // 2])
        y = fused[-1]
        skips = fused[:-1][::-1]
        for i, up in enumerate(self.ups):
            y = up(y, skips[i] if i < len(skips) else None)
        y = self.head(y)
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False) if y.shape[-2:] != (h, w) else y
