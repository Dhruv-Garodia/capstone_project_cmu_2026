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
ARCHS = ("resunet", "unet_garodia", "transunet", "segformer", "unetpp_r34", "millnet")


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
                 use_bank: bool = True, use_stats: bool = True, use_axial: bool = True):
        super().__init__()
        assert num_classes == 3
        self.k = (in_channels - 1) // 2
        self.streaks = tuple(streaks) if (use_bank and self.k > 0) else (0,)
        self.use_stats, self.use_axial, self.n_quant = use_stats, use_axial, n_quant
        per = (2 * self.k + 3) if self.k > 0 else 0
        cin = 1 + per * len(self.streaks)
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
        self.head_roi = nn.Conv2d(chs[0], 1, 1)
        self.head_pore = nn.Conv2d(chs[0], 1, 1)

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
            feats += [torch.stack(past + fut, 1), fwd[:, None], bwd[:, None], fmax[:, None]]
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
        r, u = self.head_roi(y), self.head_pore(y)
        log_in = F.logsigmoid(-r)
        # class order (solid, pore, outside); exact log-probabilities, so softmax() returns them unchanged
        return torch.cat([log_in + F.logsigmoid(-u), log_in + F.logsigmoid(u), F.logsigmoid(r)], 1)


def frame_quantiles(frames: torch.Tensor, n: int = 16) -> torch.Tensor:
    """(B, H, W) normalised frames -> (B, n) quantiles of each whole frame (subsampled for speed)."""
    flat = frames.flatten(1)
    if flat.shape[1] > 65536:
        flat = flat[:, :: flat.shape[1] // 65536]
    qs = torch.linspace(0.02, 0.98, n, device=frames.device, dtype=frames.dtype)
    return torch.quantile(flat.float().cpu(), qs.float().cpu(), dim=1).T.to(frames.device, frames.dtype)
