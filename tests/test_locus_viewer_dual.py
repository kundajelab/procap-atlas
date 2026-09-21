"""Tests for the dual-experiment locus view in notebooks/locus_viewer.py.

Written for differential-locus panels: two experiments at one locus, e.g. a
neuron promoter shown active in neurons and silent in a contrasting cell
type. Rendering each experiment's summary figure separately and pasting them
together autoscales each panel to its own data, which erases a genuine
difference in magnitude between conditions instead of displaying it (this is
exactly what CPM scaling and the fixed-axis (`track_ylim`/`logo_value_clip`)
features exist to fix -- see test_locus_viewer_batch.py). This function
groups rows by experiment (each experiment's own four rows stay together),
computing a shared axis automatically by default so the fix does not depend
on the two runs being eyeballed and matched by hand.

Row-label content is tested deliberately, not just axis limits. An earlier
version set each row's label via `ax.set_ylabel(...)` *before*
`apply_compact_summary_axis_style`, which clears `ylabel`/`title`
unconditionally -- every row's label was silently wiped, and with eight
otherwise-identical, unlabelled rows there was no way to tell which
experiment or track a given row was. That shipped because every test here
checked `ax.get_ylim()` and none checked what a row actually said. Labels
are now drawn as an `ax.text(...)` inside the axes (never touched by
`apply_compact_summary_axis_style`) rather than a rotated `ylabel`, since a
rotated two-line label does not fit an eight-row figure's per-row height.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import notebooks.locus_viewer as lv  # noqa: E402

IN_WINDOW = lv.IN_WINDOW

# Row index -> (experiment, kind), for the order plot_dual_locus_summary
# actually draws: each experiment's own four rows together, A then B.
ROW_ORDER = [
    ("a", "observed"), ("a", "predicted"), ("a", "count"), ("a", "profile"),
    ("b", "observed"), ("b", "predicted"), ("b", "count"), ("b", "profile"),
]


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


def row_label(ax):
    """The text of the in-axes corner label, or "" if none was drawn."""
    texts = [t.get_text() for t in ax.texts]
    return texts[0] if texts else ""


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


def draw(resources_a=None, resources_b=None, attrs_a=None, attrs_b=None, **kw):
    resources_a = resources_a or make_resources("a", "neuron")
    resources_b = resources_b or make_resources("b", "B cell")
    attrs = attrs_a or {"profile": np.ones((4, 10)), "count": np.ones((4, 10))}
    attrs_b = attrs_b or attrs
    return lv.plot_dual_locus_summary(
        np.zeros((2, lv.OUT_WINDOW)), attrs, resources_a, "a",
        np.zeros((2, lv.OUT_WINDOW)), attrs_b, resources_b, "b",
        "chr1:1-10", "chr1:1-10", "chr1:1-10", 0, 10, **kw,
    )


def test_eight_rows_grouped_by_experiment_not_by_track_type(stub_tracks):
    """The order requested: each experiment's own four rows together (A's
    observed/predicted/count/profile, then B's), not grouped by track type
    (both observed rows, then both predicted, ...)."""
    fig, axes = draw()
    assert len(axes) == 8
    for ax, (which, kind) in zip(axes, ROW_ORDER):
        exp_id = "a" if which == "a" else "b"
        assert exp_id in row_label(ax)
        assert kind.split()[0] in row_label(ax)  # "count"/"profile" or the raw kind


def test_every_row_has_a_nonempty_label(stub_tracks):
    """The bug this whole file exists to catch: every row must say which
    experiment and which track it is, or eight rows are indistinguishable."""
    fig, axes = draw()
    for ax in axes:
        assert row_label(ax) != ""


def test_labels_name_the_correct_experiment_and_biosample(stub_tracks):
    """resources' exp_id is what stub_tracks keys on ("a"/"b"), independent
    of the exp_id string plot_dual_locus_summary is told to use for its
    labels -- draw()'s two exp_id args are always "a"/"b", so this checks
    the biosample threads through correctly rather than duplicating the
    exp_id check above."""
    fig, axes = draw(
        resources_a=make_resources("a", "neuron"),
        resources_b=make_resources("b", "B cell"),
    )
    assert "a" in row_label(axes[0]) and "neuron" in row_label(axes[0])
    assert "b" in row_label(axes[4]) and "B cell" in row_label(axes[4])


def test_labels_survive_apply_compact_summary_axis_style(stub_tracks):
    """Regression for the exact bug: apply_compact_summary_axis_style clears
    ylabel/title unconditionally, so a label set before it (or via
    set_ylabel/set_title at all) would be wiped. Labels must be drawn as
    ax.text, which that function does not touch."""
    fig, axes = draw()
    for ax in axes:
        assert ax.get_title() == ""
        assert ax.get_ylabel() == ""
        assert row_label(ax) != ""


def test_auto_scale_matches_both_experiments_observed_rows(stub_tracks):
    """The default (track_ylim=None) behaviour: both experiments' observed
    rows share one axis sized to the larger of the two, even though they are
    four rows apart in this ordering."""
    fig, axes = draw()
    assert axes[0].get_ylim() == pytest.approx(axes[4].get_ylim())
    assert axes[0].get_ylim() == pytest.approx((-25.0, 25.0))


def test_explicit_track_ylim_overrides_the_auto_default(stub_tracks):
    fig, axes = draw(track_ylim={"observed": 100.0, "predicted": 100.0})
    assert axes[0].get_ylim() == pytest.approx((-100.0, 100.0))


def test_explicit_none_per_key_autoscales_independently(stub_tracks):
    """A caller-supplied dict is honoured key-for-key, including an explicit
    None -- opting a specific row out of shared scaling without losing the
    others."""
    fig, axes = draw(track_ylim={"observed": None, "predicted": 100.0})
    assert axes[0].get_ylim() != pytest.approx((-25.0, 25.0))
    assert axes[1].get_ylim() == pytest.approx((-100.0, 100.0))


def test_logo_rows_scale_independently_per_head(stub_tracks):
    attrs_a = {"profile": np.full((4, 10), 0.5), "count": np.full((4, 10), 10.0)}
    attrs_b = {"profile": np.full((4, 10), 0.2), "count": np.full((4, 10), 1.0)}
    fig, axes = draw(attrs_a=attrs_a, attrs_b=attrs_b, cpm_scale_a=1.0, cpm_scale_b=1.0)
    # row 2 = A's count, row 3 = A's profile
    assert axes[2].get_ylim() == pytest.approx((-10.0, 10.0))
    assert axes[3].get_ylim() == pytest.approx((-0.5, 0.5))


def test_neither_prediction_nor_attributions_are_mutated(stub_tracks):
    """save_dual_locus_viewer_outputs persists these verbatim; scaling them
    here would corrupt the saved raw record."""
    attrs_a = {"profile": np.full((4, 10), 1.0), "count": np.full((4, 10), 1.0)}
    pred_a = np.zeros((2, lv.OUT_WINDOW))
    lv.plot_dual_locus_summary(
        pred_a, attrs_a, make_resources("a", "neuron"), "a",
        np.zeros((2, lv.OUT_WINDOW)), attrs_a, make_resources("b", "B cell"), "b",
        "chr1:1-10", "chr1:1-10", "chr1:1-10", 0, 10,
        cpm_scale_a=5.0, cpm_scale_b=5.0,
    )
    assert np.allclose(pred_a, 0.0)
    assert np.allclose(attrs_a["count"], 1.0)
    assert np.allclose(attrs_a["profile"], 1.0)


# --- save_dual_locus_viewer_outputs --------------------------------------------


def test_save_writes_one_pdf_and_per_experiment_raw_arrays(tmp_path, stub_tracks):
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
