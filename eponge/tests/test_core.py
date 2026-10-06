import numpy as np
import pytest

from eponge import metrics as M
from eponge.annotation import implied_threshold, relabel_pores
from eponge.io import classes_to_export, export_to_classes
from eponge.models import UNet
from eponge.preprocess import apply_shifts, estimate_shifts


def test_label_roundtrip():
    cls = np.array([[0, 1, 2]], np.uint8)
    assert (export_to_classes(classes_to_export(cls)) == cls).all()
    with pytest.raises(ValueError):
        export_to_classes(np.array([[7]], np.uint8))


def test_porosity_and_chords():
    pore = np.zeros((10, 20), bool)
    pore[:, 5:10] = True
    assert M.porosity(pore) == pytest.approx(0.25)
    ch = M.chord_lengths(pore, axis=1)
    assert set(ch.tolist()) == {5}


def test_tortuosity_straight_channels_is_one():
    pore = np.zeros((40, 40), bool)
    pore[:, ::2] = True
    t = M.tortuosity_factor(pore, axis=0)
    assert t["tau"] == pytest.approx(1.0, rel=1e-3)
    assert t["porosity"] == pytest.approx(0.5)


def test_tortuosity_blocked_is_inf():
    pore = np.ones((20, 20), bool)
    pore[10] = False
    assert M.tortuosity_factor(pore, axis=0)["tau"] == float("inf")


def test_local_thickness_disc():
    yy, xx = np.mgrid[:64, :64]
    disc = (yy - 32) ** 2 + (xx - 32) ** 2 <= 10 ** 2
    lt = M.local_thickness(disc)
    assert 18 <= np.median(lt[disc]) <= 22


def test_registration_recovers_drift():
    rng = np.random.default_rng(0)
    base = rng.random((128, 128)).astype(np.float32)
    from scipy import ndimage as ndi
    base = ndi.gaussian_filter(base, 2)
    stack = np.stack([ndi.shift(base, (-1.0 * k, 0.0), mode="wrap") for k in range(5)])
    s = estimate_shifts(stack)
    assert s[-1][0] == pytest.approx(4.0, abs=0.2)
    reg = apply_shifts(stack, s)
    assert np.abs(reg[-1][10:-10] - reg[0][10:-10]).mean() < 0.01


def test_implied_threshold_detects_threshold_labels():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (50, 50)).astype(np.uint8)
    mask = np.where(img < 140, 0, 255).astype(np.uint8)
    t, agree = implied_threshold(img, mask)
    assert t == 140 and agree == 1.0


def test_relabel_pores_bimodal():
    img = np.full((60, 60), 210, np.uint8)
    img[20:40, 20:40] = 90
    pore, t = relabel_pores(img, np.ones_like(img, bool))
    assert pore[30, 30] and not pore[5, 5] and 90 < t < 210


def test_unet_shapes():
    m = UNet(5, 3, base=8, depth=3, max_ch=32).eval()
    x = np.random.rand(1, 5, 64, 96).astype(np.float32)
    import torch
    assert m(torch.from_numpy(x)).shape == (1, 3, 64, 96)


def test_analyze_runs():
    cls = np.full((80, 120), 2, np.uint8)
    cls[20:60] = 0
    cls[25:55:4, 10:110] = 1
    r = M.analyze(cls, pixel_nm=6.0)
    assert 0 < r["porosity"] < 1
    assert r["depth_profile"]["thickness_nm_mean"] == pytest.approx(40 * 6.0)


def test_millnet_shift_and_probabilities():
    import torch
    from eponge.models.zoo import MillNet, _shift_y
    img = torch.arange(12.0).reshape(1, 4, 3)
    assert torch.equal(_shift_y(img, 1)[0, 1], img[0, 0]) and torch.equal(_shift_y(img, 1)[0, 0], img[0, 0])
    assert torch.equal(_shift_y(img, -1)[0, 0], img[0, 1]) and torch.equal(_shift_y(img, -1)[0, 3], img[0, 3])
    m = MillNet(7, base=8, depth=3, max_ch=32).eval()
    x = torch.rand(2, 7, 64, 96)
    q = torch.sort(torch.rand(2, 16), 1).values
    out = m(x, q)
    assert out.shape == (2, 3, 64, 96)
    assert torch.allclose(out.exp().sum(1), torch.ones(2, 64, 96), atol=1e-5)  # exact log-probabilities
    assert torch.allclose(m(x), m(x, None))  # stats computed from the frame when not given


def test_millnet_streak_bank_aligns_sliding_material():
    """A feature sliding 1 px/slice down is static in the s = -1 hypothesis channels (s compensates -s px/slice)."""
    import torch
    from eponge.models.zoo import MillNet
    m = MillNet(7, base=8, depth=3, max_ch=32, use_stats=False, use_axial=False)
    x = torch.zeros(1, 7, 32, 8)
    for i, z in enumerate(range(-3, 4)):
        x[0, i, 10 + z] = 1.0          # bright row moving down one row per slice
    f = m.features_in(x)
    per = 2 * m.k + 3
    for s_idx, s in enumerate(m.streaks):
        aligned = f[0, 1 + s_idx * per: 1 + s_idx * per + 2 * m.k]
        static = all(torch.equal(a, x[0, 3]) for a in aligned)
        assert static == (s == -1)
