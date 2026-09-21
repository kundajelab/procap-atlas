"""Tests for the batched prediction/attribution path in notebooks/locus_viewer.py.

Written for the alternate-TSS analysis, which compares many TSSs per gene
across many experiments. The single-region helpers reloaded all seven fold
models on every call and hardcoded example 0, so a sweep over N regions cost
7N model loads and returned one result. The batch path makes folds the outer
loop, so each fold loads once for the whole batch.

Models are stubbed: the arithmetic under test is batching and averaging, not
BPNet.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# huggingface_hub is only needed to *download* the models and is not a dev
# dependency; nothing under test touches it.
if "huggingface_hub" not in sys.modules:
    _hub = types.ModuleType("huggingface_hub")
    _hub.hf_hub_download = lambda *a, **k: None
    sys.modules["huggingface_hub"] = _hub

import notebooks.locus_viewer as lv  # noqa: E402

IN_WINDOW = lv.IN_WINDOW


@pytest.fixture
def resources(tmp_path):
    """Seven fake fold checkpoints and a fasta path that is never read."""
    paths = []
    for fold in range(7):
        path = tmp_path / f"fold_{fold}.torch"
        path.write_bytes(b"")
        paths.append(path)
    return {"model_paths": paths, "fasta": tmp_path / "hg38.fa"}


class CountingLoader:
    """Stand-in for torch.load that records how often each fold is read."""

    def __init__(self, per_fold_value):
        self.loads = []
        self._value = per_fold_value

    def __call__(self, path, **kwargs):
        self.loads.append(Path(path).name)
        model = types.SimpleNamespace(value=self._value(Path(path).name))
        model.eval = lambda: model
        return model


def stub_predict(n_positions=8):
    """predict() returning per-example logits/log-counts keyed off the input."""

    def _predict(model=None, X=None, batch_size=None, device=None):
        n = X.shape[0]
        logits = torch.zeros(n, 2, n_positions)
        log_counts = torch.zeros(n, 1)
        for i in range(n):
            logits[i] += float(i) + model.value
            log_counts[i] += float(i) + model.value
        return logits, log_counts

    return _predict


# --- region_inputs -----------------------------------------------------------


def test_region_inputs_returns_records_aligned_with_the_batch(monkeypatch):
    regions = ["chr1:1000-1099", "chr2:5000-5099", "chr3:77-176"]
    monkeypatch.setattr(
        lv, "extract_loci",
        lambda loci, **kw: torch.zeros(len(loci), 4, IN_WINDOW),
    )
    records, X = lv.region_inputs({"fasta": "x"}, regions)

    assert len(records) == len(regions) == X.shape[0]
    assert list(records["region"]) == regions
    assert list(records["chrom"]) == ["chr1", "chr2", "chr3"]
    # centers must line up row-for-row with the extracted batch
    assert records.loc[0, "center"] == lv.interval_center(1000, 1099)
    assert X.dtype == torch.float32


def test_region_inputs_carries_per_region_logo_offsets(monkeypatch):
    """Each input is centred on its own region, so the offsets are per row."""
    monkeypatch.setattr(
        lv, "extract_loci",
        lambda loci, **kw: torch.zeros(len(loci), 4, IN_WINDOW),
    )
    records, _ = lv.region_inputs(
        {"fasta": "x"}, ["chr1:1000-1099", "chr1:2000-2199"]
    )
    assert records.loc[0, "logo_offsets"] != records.loc[1, "logo_offsets"]
    width_0 = records.loc[0, "logo_offsets"][1] - records.loc[0, "logo_offsets"][0]
    width_1 = records.loc[1, "logo_offsets"][1] - records.loc[1, "logo_offsets"][0]
    assert (width_0, width_1) == (100, 200)


def test_region_inputs_raises_when_a_region_is_dropped(monkeypatch):
    """extract_loci silently drops regions off a chromosome end or with N.
    Returning a short batch would misalign every downstream result."""
    monkeypatch.setattr(
        lv, "extract_loci", lambda loci, **kw: torch.zeros(2, 4, IN_WINDOW)
    )
    with pytest.raises(ValueError, match="returned 2 of 3 regions"):
        lv.region_inputs({"fasta": "x"}, ["chr1:1-2", "chr1:3-4", "chr1:5-6"])


# --- ensemble_predictions ----------------------------------------------------


def test_each_fold_loads_once_for_the_whole_batch(resources, monkeypatch):
    """The point of the batch path: 7 loads, not 7N."""
    loader = CountingLoader(lambda name: 0.0)
    monkeypatch.setattr(lv.torch, "load", loader)
    monkeypatch.setattr(lv, "predict", stub_predict())
    monkeypatch.setattr(lv, "count_scaled_profile", lambda lo, lc: lo)

    lv.ensemble_predictions(resources, torch.zeros(12, 4, IN_WINDOW), 7, "cpu")
    assert len(loader.loads) == 7
    assert len(set(loader.loads)) == 7


def test_n_folds_limits_how_many_models_load(resources, monkeypatch):
    loader = CountingLoader(lambda name: 0.0)
    monkeypatch.setattr(lv.torch, "load", loader)
    monkeypatch.setattr(lv, "predict", stub_predict())
    monkeypatch.setattr(lv, "count_scaled_profile", lambda lo, lc: lo)

    lv.ensemble_predictions(resources, torch.zeros(3, 4, IN_WINDOW), 3, "cpu")
    assert len(loader.loads) == 3


def test_predictions_are_returned_per_example(resources, monkeypatch):
    """(N, ...) out for (N, ...) in -- the single-region path returned only
    example 0, which silently discarded the rest of a batch."""
    monkeypatch.setattr(lv.torch, "load", CountingLoader(lambda name: 0.0))
    monkeypatch.setattr(lv, "predict", stub_predict())
    monkeypatch.setattr(lv, "count_scaled_profile", lambda lo, lc: lo)

    out = lv.ensemble_predictions(resources, torch.zeros(5, 4, IN_WINDOW), 7, "cpu")
    assert out.shape[0] == 5
    # stub_predict adds the example index, so rows must differ and be ordered
    assert np.allclose(out[:, 0, 0], np.arange(5.0))


def test_folds_are_averaged_on_logits_not_on_scaled_profiles(resources, monkeypatch):
    """Averaging count-scaled profiles would weight folds by predicted depth.
    Fold values 0..6 average to 3.0 before scaling."""
    monkeypatch.setattr(
        lv.torch, "load",
        CountingLoader(lambda name: float(name.split("_")[1].split(".")[0])),
    )
    monkeypatch.setattr(lv, "predict", stub_predict())
    seen = {}

    def record(logits, log_counts):
        seen["logits"] = logits
        seen["log_counts"] = log_counts
        return logits

    monkeypatch.setattr(lv, "count_scaled_profile", record)
    lv.ensemble_predictions(resources, torch.zeros(1, 4, IN_WINDOW), 7, "cpu")
    assert np.isclose(seen["logits"][0, 0, 0], 3.0)
    assert np.isclose(seen["log_counts"][0, 0], 3.0)


def test_single_region_helper_returns_example_zero(resources, monkeypatch):
    """Backward compatibility: the notebook's existing cells call this."""
    monkeypatch.setattr(lv.torch, "load", CountingLoader(lambda name: 0.0))
    monkeypatch.setattr(lv, "predict", stub_predict())
    monkeypatch.setattr(lv, "count_scaled_profile", lambda lo, lc: lo)

    X = torch.zeros(1, 4, IN_WINDOW)
    batch = lv.ensemble_predictions(resources, X, 7, "cpu")
    single = lv.ensemble_prediction(resources, X, 7, "cpu")
    assert np.allclose(single, batch[0])
    assert single.shape == batch.shape[1:]


# --- deeplift_attributions_batch ---------------------------------------------


def test_attributions_are_returned_per_example(resources, monkeypatch):
    """The single-region path sliced [0, :, window]; the batch path must slice
    [:, :, window] or a sweep silently returns the first region N times."""
    monkeypatch.setattr(lv.torch, "load", CountingLoader(lambda name: 0.0))
    monkeypatch.setattr(lv, "ProfileWrapper", lambda model: model)
    monkeypatch.setattr(lv, "CountWrapper", lambda model: model)

    def fake_shap(model=None, X=None, **kwargs):
        # example i gets constant i, so per-example slicing is observable
        return torch.arange(X.shape[0], dtype=torch.float32)[:, None, None] * (
            torch.ones_like(X)
        )

    monkeypatch.setattr(lv, "deep_lift_shap", fake_shap)
    X = torch.ones(4, 4, IN_WINDOW)
    out = lv.deeplift_attributions_batch(resources, X, (10, 30), 7, 2, "cpu")

    assert set(out) == {"profile", "count"}
    for values in out.values():
        assert values.shape == (4, 4, 20)
        assert np.allclose(values[:, 0, 0], np.arange(4.0))


def test_attribution_folds_load_once_for_the_whole_batch(resources, monkeypatch):
    loader = CountingLoader(lambda name: 0.0)
    monkeypatch.setattr(lv.torch, "load", loader)
    monkeypatch.setattr(lv, "ProfileWrapper", lambda model: model)
    monkeypatch.setattr(lv, "CountWrapper", lambda model: model)
    monkeypatch.setattr(
        lv, "deep_lift_shap", lambda model=None, X=None, **kw: torch.ones_like(X)
    )
    lv.deeplift_attributions_batch(
        resources, torch.ones(9, 4, IN_WINDOW), (0, 10), 7, 4, "cpu"
    )
    assert len(loader.loads) == 7


def test_single_region_attributions_return_example_zero(resources, monkeypatch):
    monkeypatch.setattr(lv.torch, "load", CountingLoader(lambda name: 0.0))
    monkeypatch.setattr(lv, "ProfileWrapper", lambda model: model)
    monkeypatch.setattr(lv, "CountWrapper", lambda model: model)
    monkeypatch.setattr(
        lv, "deep_lift_shap", lambda model=None, X=None, **kw: torch.ones_like(X)
    )
    X = torch.ones(1, 4, IN_WINDOW)
    single = lv.deeplift_attributions(resources, X, (0, 10), 7, 4, "cpu")
    batch = lv.deeplift_attributions_batch(resources, X, (0, 10), 7, 4, "cpu")
    for head in ("profile", "count"):
        assert single[head].shape == (4, 10)
        assert np.allclose(single[head], batch[head][0])




# --- device selection --------------------------------------------------------
#
# The notebook previously hardcoded `"cuda" if torch.cuda.is_available() else
# "cpu"`, so an Apple-silicon machine fell back to CPU despite MPS being
# available. Attributions on MPS match CPU to float32 rounding (~1e-7 on this
# path), so there is no accuracy reason to skip it.


def test_best_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(lv.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(lv.torch.backends.mps, "is_available", lambda: True)
    assert lv.best_device() == "cuda"


def test_best_device_falls_back_to_metal(monkeypatch):
    monkeypatch.setattr(lv.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(lv.torch.backends.mps, "is_available", lambda: True)
    assert lv.best_device() == "mps"


def test_best_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(lv.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(lv.torch.backends.mps, "is_available", lambda: False)
    assert lv.best_device() == "cpu"


def test_free_device_memory_uses_the_matching_backend(monkeypatch):
    """Calling torch.cuda.empty_cache() on an MPS box frees nothing, so a
    long sweep would accumulate fold allocations."""
    calls = []
    monkeypatch.setattr(lv.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(lv.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(lv.torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(lv.torch.mps, "empty_cache", lambda: calls.append("mps"))
    lv.free_device_memory()
    assert calls == ["mps"]


# --- soft PFM references -----------------------------------------------------


def test_references_are_passed_as_a_callable_not_a_tensor():
    """tangermeme only one-hot-validates *Tensor* references, and the soft PFM
    baseline this viewer exists for is not one-hot -- passing a prebuilt
    tensor raises "references must be one-hot encoded". A callable skips that
    check. This is what c3cbb07 fixed in production, letting the local
    soft-reference fork of deep_lift_shap be deleted."""
    import inspect

    source = inspect.getsource(lv.deeplift_attributions_batch)
    assert "references=nucleotide_frequency_references," in source
    assert "nucleotide_frequency_references(X)" not in source


def test_reference_callable_matches_production():
    """The viewer must not keep a divergent copy of the baseline."""
    from src.bpnet.attribute.attribute_bpnet import (
        nucleotide_frequency_references as production,
    )

    assert lv.nucleotide_frequency_references is production


def test_reference_callable_is_per_example_and_soft():
    X = torch.zeros(3, 4, 8)
    X[0, 0] = 1.0
    X[1, 1] = 1.0
    X[2, 2] = 1.0
    references = lv.nucleotide_frequency_references(X)
    assert references.shape == (3, 1, 4, 8)
    assert references[0, 0, 0, 0] == 1.0
    assert references[1, 0, 0, 0] == 0.0


# --- CPM scaling -------------------------------------------------------------
#
# Observed/predicted coverage and counts-head DeepLIFT are all on a raw-count
# scale that library depth moves directly, so a cross-experiment comparison
# needs them on a common CPM footing. Profile-head DeepLIFT explains a
# softmax output -- a shape distribution, depth-independent by construction
# -- and must never be scaled. Scaling happens only on the locally drawn
# copies inside the plotting functions, never on the arrays saved to
# locus_viewer_arrays.npz, so raw values stay reproducible regardless of how
# a figure was rendered.


def test_cpm_scale_for_reads_total_reads(tmp_path):
    n_reads = tmp_path / "n_reads.txt"
    n_reads.write_text("experiment\tbiosample\ttotal_reads\n"
                       "ENCSR342WAR\tneuron\t50000000\n")
    assert lv.cpm_scale_for("ENCSR342WAR", n_reads) == pytest.approx(1e6 / 5e7)


def test_cpm_scale_for_missing_file_warns_and_returns_none(tmp_path):
    with pytest.warns(UserWarning, match="not found"):
        result = lv.cpm_scale_for("ENCSR342WAR", tmp_path / "missing.txt")
    assert result is None


def test_cpm_scale_for_missing_experiment_warns_and_returns_none(tmp_path):
    n_reads = tmp_path / "n_reads.txt"
    n_reads.write_text("experiment\tbiosample\ttotal_reads\n"
                       "ENCSR342WAR\tneuron\t50000000\n")
    with pytest.warns(UserWarning, match="ENCSR083AMN"):
        result = lv.cpm_scale_for("ENCSR083AMN", n_reads)
    assert result is None


def test_track_arrays_scales_all_four_tracks(resources, monkeypatch):
    monkeypatch.setattr(
        lv, "extract_loci", lambda loci, **kw: torch.zeros(1, 4, IN_WINDOW)
    )
    monkeypatch.setattr(
        lv, "bigwig_values",
        lambda path, chrom, start, end: np.full(end - start, 2.0),
    )
    prediction = np.full((2, lv.OUT_WINDOW), 3.0)
    region = "chr1:1000-1099"
    resources_local = dict(resources, fasta="x")
    resources_local["observed"] = {"plus": "p.bw", "minus": "m.bw"}

    raw = lv.track_arrays(prediction, resources_local, region, region)
    scaled = lv.track_arrays(prediction, resources_local, region, region,
                             cpm_scale=10.0)
    for key in ("observed_plus", "observed_minus", "predicted_plus",
               "predicted_minus"):
        assert np.allclose(scaled[key], raw[key] * 10.0)


def test_track_arrays_default_is_unscaled(resources, monkeypatch):
    """No cpm_scale given must reproduce the pre-existing raw behaviour."""
    monkeypatch.setattr(
        lv, "bigwig_values",
        lambda path, chrom, start, end: np.full(end - start, 5.0),
    )
    prediction = np.full((2, lv.OUT_WINDOW), 7.0)
    region = "chr1:1000-1099"
    resources_local = dict(resources)
    resources_local["observed"] = {"plus": "p.bw", "minus": "m.bw"}
    tracks = lv.track_arrays(prediction, resources_local, region, region)
    assert np.allclose(tracks["predicted_plus"], 7.0)
    assert np.allclose(tracks["observed_plus"], 5.0)


def test_count_logo_is_scaled_profile_logo_is_not(monkeypatch):
    """The one rule this whole feature exists to enforce."""
    monkeypatch.setattr(lv, "oriented_logo_matrix", lambda m, rc: m.copy())
    drawn = {}

    def fake_panel(ax, matrix, title, *a, **kw):
        drawn[title.split()[0]] = matrix

    monkeypatch.setattr(lv, "plot_logo_panel", fake_panel)
    attributions = {
        "profile": np.ones((4, 20)),
        "count": np.ones((4, 20)),
    }
    fig, axes = lv.plot_deeplift_logos(
        attributions, "EXP", "chr1:1-20", 0, 20, cpm_scale=10.0
    )
    assert np.allclose(drawn["profile"], 1.0)
    assert np.allclose(drawn["count"], 10.0)


def test_deeplift_attributions_never_scales_the_returned_dict(
    resources, monkeypatch
):
    """attributions themselves stay raw; only the drawn copy is scaled.
    save_locus_viewer_outputs persists this dict verbatim to
    locus_viewer_arrays.npz, so scaling it here would corrupt the saved
    raw record."""
    monkeypatch.setattr(lv.torch, "load", CountingLoader(lambda name: 0.0))
    monkeypatch.setattr(lv, "ProfileWrapper", lambda model: model)
    monkeypatch.setattr(lv, "CountWrapper", lambda model: model)
    monkeypatch.setattr(
        lv, "deep_lift_shap", lambda model=None, X=None, **kw: torch.ones_like(X)
    )
    X = torch.ones(1, 4, IN_WINDOW)
    out = lv.deeplift_attributions(resources, X, (0, 10), 7, 4, "cpu")
    assert np.allclose(out["count"], 1.0)
    assert np.allclose(out["profile"], 1.0)
