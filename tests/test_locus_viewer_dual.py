"""Tests for the dual-experiment locus view in notebooks/locus_viewer.py.

Written for differential-locus panels: two experiments at one locus, e.g. a
neuron promoter shown active in neurons and silent in a contrasting cell
type. Rendering each experiment's summary figure separately and pasting them
together autoscales each panel to its own data, which erases a genuine
difference in magnitude between conditions instead of displaying it (this is
exactly what CPM scaling and the fixed-axis (`track_ylim`/`logo_value_clip`)
features exist to fix -- see test_locus_viewer_batch.py). This function
grouping by track type, computing a shared axis automatically by default, so
that fix does not depend on the two runs being manually eyeballed and
matched by hand.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import notebooks.locus_viewer as lv  # noqa: E402

IN_WINDOW = lv.IN_WINDOW


def make_resources(exp_id, biosample):
    return {"exp_id": exp_id, "config": {"biosample": biosample}}


def make_tracks(observed_peak, predicted_peak, n=10):
    return {
        "x": np.arange(n, dtype=float),
        "observed_plus": np.full(n, observed_peak),
        "observed_minus": np.full(n, -observed_peak / 2),
        "predicted_plus": np.full(n, predicted_peak),
        "predicted_minus": np.full(n, -predicted_peak / 2),
    }


# --- paired_track_ylim --------------------------------------------------------


def test_paired_track_ylim_takes_the_larger_of_the_two():
    tracks_a = make_tracks(observed_peak=25.0, predicted_peak=20.0)
    tracks_b = make_tracks(observed_peak=2.0, predicted_peak=1.5)
    ylim = lv.paired_track_ylim(tracks_a, tracks_b)
    assert ylim["observed"] == pytest.approx(25.0)
    assert ylim["predicted"] == pytest.approx(20.0)


def test_paired_track_ylim_the_smaller_side_can_set_it_too():
    tracks_a = make_tracks(observed_peak=2.0, predicted_peak=1.5)
    tracks_b = make_tracks(observed_peak=25.0, predicted_peak=20.0)
    ylim = lv.paired_track_ylim(tracks_a, tracks_b)
    assert ylim["observed"] == pytest.approx(25.0)


def test_paired_track_ylim_zero_peak_returns_none_not_zero():
    """format_track_axis rejects a non-positive ylim; an all-zero pair must
    autoscale rather than raise."""
    tracks_a = make_tracks(observed_peak=0.0, predicted_peak=0.0)
    tracks_b = make_tracks(observed_peak=0.0, predicted_peak=0.0)
    ylim = lv.paired_track_ylim(tracks_a, tracks_b)
    assert ylim["observed"] is None
    assert ylim["predicted"] is None


# --- paired_logo_clip ----------------------------------------------------------


def test_paired_logo_clip_scales_count_but_not_profile():
    attrs_a = {"profile": np.full((4, 5), 10.0), "count": np.full((4, 5), 10.0)}
    attrs_b = {"profile": np.full((4, 5), 1.0), "count": np.full((4, 5), 1.0)}
    clip = lv.paired_logo_clip(attrs_a, attrs_b, cpm_scale_a=2.0, cpm_scale_b=2.0)
    assert clip["profile"] == pytest.approx(10.0)   # never scaled
    assert clip["count"] == pytest.approx(20.0)      # 10.0 * 2.0


def test_paired_logo_clip_handles_missing_cpm_scale():
    attrs_a = {"profile": np.full((4, 5), 1.0), "count": np.full((4, 5), 3.0)}
    attrs_b = {"profile": np.full((4, 5), 1.0), "count": np.full((4, 5), 1.0)}
    clip = lv.paired_logo_clip(attrs_a, attrs_b, cpm_scale_a=None, cpm_scale_b=None)
    assert clip["count"] == pytest.approx(3.0)


def test_paired_logo_clip_zero_peak_returns_none():
    attrs_a = {"profile": np.zeros((4, 5)), "count": np.zeros((4, 5))}
    attrs_b = {"profile": np.zeros((4, 5)), "count": np.zeros((4, 5))}
    clip = lv.paired_logo_clip(attrs_a, attrs_b, None, None)
    assert clip["profile"] is None
    assert clip["count"] is None


# --- plot_dual_locus_summary ---------------------------------------------------


@pytest.fixture
def stub_tracks(monkeypatch):
    """track_arrays keyed by which resources dict it was called with, so
    each experiment's row gets distinguishable data."""
    data = {
        "a": make_tracks(observed_peak=25.0, predicted_peak=20.0),
        "b": make_tracks(observed_peak=2.0, predicted_peak=1.5),
    }

    def fake_track_arrays(prediction, resources, *a, **kw):
        return data[resources["exp_id"]]

    monkeypatch.setattr(lv, "track_arrays", fake_track_arrays)
    monkeypatch.setattr(lv, "oriented_logo_matrix", lambda m, rc: m)
    monkeypatch.setattr(lv, "plot_logo", lambda *a, **kw: None)
    return data


def test_eight_rows_in_track_type_order(stub_tracks):
    """Grouped by type (obs A, obs B, pred A, pred B, ...), not by experiment
    (all of A then all of B) -- the whole point is putting what should be
    compared next to each other."""
    attrs = {"profile": np.ones((4, 10)), "count": np.ones((4, 10))}
    fig, axes = lv.plot_dual_locus_summary(
        np.zeros((2, lv.OUT_WINDOW)), attrs, make_resources("a", "neuron"), "a",
        np.zeros((2, lv.OUT_WINDOW)), attrs, make_resources("b", "B cell"), "b",
        "chr1:1-10", "chr1:1-10", "chr1:1-10", 0, 10,
    )
    assert len(axes) == 8


def test_auto_scale_matches_both_rows_to_the_larger_experiment(stub_tracks):
    """The default (track_ylim=None) behaviour: both experiments' observed
    rows share one axis sized to the larger of the two."""
    attrs = {"profile": np.ones((4, 10)), "count": np.ones((4, 10))}
    fig, axes = lv.plot_dual_locus_summary(
        np.zeros((2, lv.OUT_WINDOW)), attrs, make_resources("a", "neuron"), "a",
        np.zeros((2, lv.OUT_WINDOW)), attrs, make_resources("b", "B cell"), "b",
        "chr1:1-10", "chr1:1-10", "chr1:1-10", 0, 10,
    )
    # rows 0, 1 are observed A, observed B
    assert axes[0].get_ylim() == pytest.approx(axes[1].get_ylim())
    assert axes[0].get_ylim() == pytest.approx((-25.0, 25.0))


def test_explicit_track_ylim_overrides_the_auto_default(stub_tracks):
    attrs = {"profile": np.ones((4, 10)), "count": np.ones((4, 10))}
    fig, axes = lv.plot_dual_locus_summary(
        np.zeros((2, lv.OUT_WINDOW)), attrs, make_resources("a", "neuron"), "a",
        np.zeros((2, lv.OUT_WINDOW)), attrs, make_resources("b", "B cell"), "b",
        "chr1:1-10", "chr1:1-10", "chr1:1-10", 0, 10,
        track_ylim={"observed": 100.0, "predicted": 100.0},
    )
    assert axes[0].get_ylim() == pytest.approx((-100.0, 100.0))


def test_explicit_none_per_key_autoscales_independently(stub_tracks):
    """A caller-supplied dict is honoured key-for-key, including an explicit
    None -- opting a specific row out of shared scaling without losing the
    others."""
    attrs = {"profile": np.ones((4, 10)), "count": np.ones((4, 10))}
    fig, axes = lv.plot_dual_locus_summary(
        np.zeros((2, lv.OUT_WINDOW)), attrs, make_resources("a", "neuron"), "a",
        np.zeros((2, lv.OUT_WINDOW)), attrs, make_resources("b", "B cell"), "b",
        "chr1:1-10", "chr1:1-10", "chr1:1-10", 0, 10,
        track_ylim={"observed": None, "predicted": 100.0},
    )
    assert axes[0].get_ylim() != pytest.approx((-25.0, 25.0))
    assert axes[2].get_ylim() == pytest.approx((-100.0, 100.0))


def test_logo_rows_scale_independently_per_head(stub_tracks):
    attrs_a = {"profile": np.full((4, 10), 0.5), "count": np.full((4, 10), 10.0)}
    attrs_b = {"profile": np.full((4, 10), 0.2), "count": np.full((4, 10), 1.0)}
    fig, axes = lv.plot_dual_locus_summary(
        np.zeros((2, lv.OUT_WINDOW)), attrs_a, make_resources("a", "neuron"), "a",
        np.zeros((2, lv.OUT_WINDOW)), attrs_b, make_resources("b", "B cell"), "b",
        "chr1:1-10", "chr1:1-10", "chr1:1-10", 0, 10,
        cpm_scale_a=1.0, cpm_scale_b=1.0,
    )
    # rows 4,5 = profile A,B; rows 6,7 = count A,B
    assert axes[4].get_ylim() == pytest.approx((-0.5, 0.5))
    assert axes[6].get_ylim() == pytest.approx((-10.0, 10.0))


def test_neither_prediction_nor_attributions_are_mutated(stub_tracks):
    """save_dual_locus_viewer_outputs persists these verbatim; scaling them
    here would corrupt the saved raw record."""
    attrs_a = {"profile": np.full((4, 10), 1.0), "count": np.full((4, 10), 1.0)}
    attrs_b = {"profile": np.full((4, 10), 1.0), "count": np.full((4, 10), 1.0)}
    pred_a = np.zeros((2, lv.OUT_WINDOW))
    lv.plot_dual_locus_summary(
        pred_a, attrs_a, make_resources("a", "neuron"), "a",
        np.zeros((2, lv.OUT_WINDOW)), attrs_b, make_resources("b", "B cell"), "b",
        "chr1:1-10", "chr1:1-10", "chr1:1-10", 0, 10,
        cpm_scale_a=5.0, cpm_scale_b=5.0,
    )
    assert np.allclose(pred_a, 0.0)
    assert np.allclose(attrs_a["count"], 1.0)
    assert np.allclose(attrs_a["profile"], 1.0)


# --- save_dual_locus_viewer_outputs --------------------------------------------


def test_save_writes_one_pdf_and_per_experiment_raw_arrays(
    tmp_path, stub_tracks
):
    """Each experiment's raw arrays land under its own subdirectory, using
    the same locus_viewer_arrays.npz filename save_locus_viewer_outputs
    uses -- so nothing downstream needs a second naming convention."""
    attrs = {"profile": np.ones((4, 10)), "count": np.ones((4, 10))}
    lv.save_dual_locus_viewer_outputs(
        tmp_path,
        np.zeros((2, lv.OUT_WINDOW)), attrs, make_resources("a", "neuron"), "a",
        np.zeros((2, lv.OUT_WINDOW)), attrs, make_resources("b", "B cell"), "b",
        "chr1:1-10", "chr1:1-10", "chr1:1-10", 0, 10,
    )
    assert (tmp_path / "locus_viewer_dual_summary.pdf").exists()
    assert (tmp_path / "a" / "locus_viewer_arrays.npz").exists()
    assert (tmp_path / "b" / "locus_viewer_arrays.npz").exists()


def test_save_raw_arrays_round_trip(tmp_path):
    prediction = np.arange(2 * lv.OUT_WINDOW, dtype=np.float32).reshape(
        2, lv.OUT_WINDOW
    )
    attributions = {"profile": np.ones((4, 10)), "count": np.full((4, 10), 2.0)}
    lv.save_raw_locus_arrays(
        tmp_path, prediction, attributions, "chr1:1-10", "chr1:1-10", "chr1:1-10"
    )
    loaded = np.load(tmp_path / "locus_viewer_arrays.npz")
    assert np.allclose(loaded["prediction"], prediction)
    assert np.allclose(loaded["count_deeplift"], 2.0)
