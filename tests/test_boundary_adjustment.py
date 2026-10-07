import numpy as np

from experiments.auto_correct_real_masks import (
    repair_unsupported_indentations,
    smooth_continuous_surfaces,
    smooth_jagged_surfaces,
)


def make_surfaces(value=10.0, frames=5, width=80):
    return np.full((frames, width), value, dtype=np.float32)


def make_stack(frames=5, height=64, width=80):
    return np.zeros((frames, height, width), dtype=np.uint8)


def repair(stack, surfaces, side="top", **overrides):
    options = {
        "radius": 2,
        "spatial_window": 15,
        "min_depth": 8.0,
        "max_width": 10,
        "edge_quantile": 0.75,
        "max_edge_support": 0.35,
        "temporal_support": 3,
        "x_tolerance": 1,
    }
    options.update(overrides)
    return repair_unsupported_indentations(
        stack, list(range(len(surfaces))), surfaces, side, **options
    )


def test_repairs_unsupported_deep_narrow_top_indentation():
    stack = make_stack()
    surfaces = make_surfaces()
    surfaces[2, 30:35] = 28

    corrected, accepted, audit = repair(stack, surfaces)

    assert audit["candidate_segments"] == 1
    assert audit["accepted_segments"] == 1
    assert accepted[2, 30:35].all()
    assert np.max(corrected[2, 30:35]) < 14


def test_preserves_indentation_with_strong_image_edge():
    stack = make_stack()
    surfaces = make_surfaces()
    surfaces[2, 30:35] = 28
    stack[2, 28:, 30:35] = 255

    corrected, accepted, audit = repair(stack, surfaces)

    assert audit["accepted_segments"] == 0
    assert audit["rejected_image_edge"] == 1
    assert not accepted.any()
    np.testing.assert_array_equal(corrected, surfaces)


def test_preserves_temporally_supported_indentation():
    stack = make_stack()
    surfaces = make_surfaces()
    surfaces[1:4, 30:35] = 28

    corrected, accepted, audit = repair(stack, surfaces)

    assert audit["accepted_segments"] == 0
    assert audit["rejected_temporal_support"] == 3
    assert not accepted.any()
    np.testing.assert_array_equal(corrected, surfaces)


def test_preserves_wide_exterior_hole():
    stack = make_stack()
    surfaces = make_surfaces()
    surfaces[2, 25:45] = 28

    corrected, accepted, audit = repair(
        stack, surfaces, spatial_window=41
    )

    assert audit["accepted_segments"] == 0
    assert audit["rejected_too_wide"] == 1
    assert not accepted.any()
    np.testing.assert_array_equal(corrected, surfaces)


def test_repairs_unsupported_bottom_indentation():
    stack = make_stack()
    surfaces = make_surfaces(value=50)
    surfaces[2, 30:35] = 32

    corrected, accepted, audit = repair(stack, surfaces, side="bottom")

    assert audit["accepted_segments"] == 1
    assert accepted[2, 30:35].all()
    assert np.min(corrected[2, 30:35]) > 46



def smooth(surfaces, **overrides):
    options = {
        "radius": 2,
        "spatial_window": 9,
        "min_deviation": 2.0,
        "max_width": 16,
        "sigma": 1.25,
        "temporal_weight": 0.35,
    }
    options.update(overrides)
    return smooth_jagged_surfaces(
        list(range(len(surfaces))), surfaces, len(surfaces), **options
    )


def test_smooths_temporally_repeated_narrow_spike():
    surfaces = make_surfaces()
    surfaces[:, 30:32] = 28

    corrected, accepted, audit = smooth(surfaces)

    assert audit["accepted_segments"] == len(surfaces)
    assert accepted[:, 30:32].all()
    assert np.max(corrected[:, 30:32]) < 14


def test_preserves_center_of_broad_cavity():
    surfaces = make_surfaces()
    surfaces[:, 25:45] = 28

    corrected, _, _ = smooth(surfaces)

    np.testing.assert_array_equal(corrected[:, 30:40], surfaces[:, 30:40])


def test_preserves_already_smooth_surface():
    surfaces = make_surfaces()
    surfaces[:] = np.linspace(10, 20, surfaces.shape[1])

    corrected, accepted, audit = smooth(surfaces)

    assert audit["accepted_segments"] == 0
    assert not accepted.any()
    np.testing.assert_array_equal(corrected, surfaces)



def test_temporal_prior_cannot_recreate_local_narrow_spike():
    surfaces = make_surfaces()
    surfaces[[0, 1, 3, 4], 25:45] = 28
    surfaces[2, 30:32] = 28

    corrected, accepted, _ = smooth(surfaces)

    assert accepted[2, 30:32].all()
    assert np.max(corrected[2, 30:32]) < 12



def test_continuous_curve_rounds_corners_but_retains_broad_hole():
    surfaces = make_surfaces()
    surfaces[:, 25:45] = 28

    corrected, changed, _ = smooth_continuous_surfaces(
        list(range(len(surfaces))),
        surfaces,
        len(surfaces),
        radius=2,
        sigma=4.0,
        temporal_weight=0.15,
        max_temporal_shift=2.0,
    )

    assert changed.any()
    assert np.min(corrected[:, 33:37]) > 26
    assert np.max(np.abs(np.diff(corrected, axis=1))) < 3


def test_continuous_curve_preserves_flat_surface():
    surfaces = make_surfaces()

    corrected, changed, audit = smooth_continuous_surfaces(
        list(range(len(surfaces))),
        surfaces,
        len(surfaces),
    )

    assert audit["adjusted_columns"] == 0
    assert not changed.any()
    np.testing.assert_array_equal(corrected, surfaces)
