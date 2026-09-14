"""Tests for src/bpnet/motifcompendium/cluster_motifs.py.

MotifCompendium is an optional research dependency and is not installed in
this environment (see CLAUDE.md), so the module is imported against stubs.
That is enough to cover the part worth covering: which clustering algorithm
gets used, and whether the choice is recorded.

The motivating incident: MotifCompendium v1.0.19 changed mc.cluster's default
from "cpm_leiden" to ["cpm_leiden", "k_centroids"], silently adding an
uncapped k-means refinement. cluster_motifs.py passed no algorithm, so the
change arrived with a version bump and left no trace in any output.
"""

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(mc_version="1.0.19", default_algorithm=None):
    """Import cluster_motifs.py against a stubbed MotifCompendium."""
    if default_algorithm is None:
        default_algorithm = ["cpm_leiden", "k_centroids"]

    stub = types.ModuleType("MotifCompendium")
    stub.__version__ = mc_version
    stub.set_compute_options = lambda **kw: None
    stub.build_from_modisco = lambda paths: None
    for sub in ("utils", "utils.analysis", "utils.motif", "utils.plotting"):
        name = f"MotifCompendium.{sub}"
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        if sub == "utils":
            stub.utils = mod
        else:
            setattr(stub.utils, sub.split(".")[-1], mod)
    sys.modules["MotifCompendium"] = stub

    for stale in [m for m in sys.modules if m.endswith("cluster_motifs")]:
        del sys.modules[stale]
    sys.path.insert(0, str(REPO_ROOT / "src" / "bpnet" / "motifcompendium"))
    import cluster_motifs

    return cluster_motifs


class FakeMC:
    """Records the kwargs mc.cluster was called with."""

    def __init__(self, default_algorithm, supports_algorithm_kwargs=True,
                 weight_col_name="weight_col"):
        self.calls = []
        self._default = default_algorithm
        self._supports_kwargs = supports_algorithm_kwargs
        self._weight_col_name = weight_col_name

    def cluster(self, **kwargs):
        self.calls.append(kwargs)

    @property
    def _signature_params(self):
        params = ["similarity_threshold", "save_name", "cluster_on",
                  "cluster_within", "algorithm", self._weight_col_name]
        if self._supports_kwargs:
            params.append("algorithm_kwargs")
        return params


def bind_signature(fake):
    """Give fake.cluster a real signature so inspect.signature works."""
    import inspect

    names = fake._signature_params
    src_params = ", ".join(f"{n}=None" for n in names)
    ns = {}
    exec(f"def cluster(self, {src_params}, **kwargs):\n"
         f"    self.calls.append(dict("
         f"{', '.join(f'{n}={n}' for n in names)}, **kwargs))\n", ns)
    fake.cluster = types.MethodType(ns["cluster"], fake)
    assert "algorithm" in inspect.signature(fake.cluster).parameters
    return fake


def make_fake(cm=None, default_algorithm=None, **kw):
    """A stand-in for a MotifCompendium instance with a real `cluster` signature."""
    fake = FakeMC(default_algorithm or ["cpm_leiden", "k_centroids"], **kw)
    return bind_signature(fake)


# --- resolve_algorithm -------------------------------------------------------


def test_default_is_taken_from_the_installed_library_not_hardcoded():
    """The point of the flag is that the default is version-dependent, so
    resolve_algorithm must read it off mc.cluster rather than assume one."""
    cm = load_module()
    fake = _WithDefault(make_fake(), ["cpm_leiden", "k_centroids"])
    alg, kwargs, label = cm.resolve_algorithm(fake, None, None)
    assert alg == ["cpm_leiden", "k_centroids"]
    assert kwargs is None
    assert label == "cpm_leiden+k_centroids"


class _WithDefault:
    """Wraps a fake so mc.cluster's `algorithm` default is a chosen value."""

    def __init__(self, fake, default):
        import inspect

        self._fake = fake
        self._default = default
        params = list(inspect.signature(fake.cluster).parameters.values())
        new = []
        for prm in params:
            if prm.name == "algorithm":
                prm = prm.replace(default=default)
            new.append(prm)
        self._sig = inspect.Signature(new)

    @property
    def cluster(self):
        fn = self._fake.cluster
        fn.__func__.__signature__ = self._sig
        return fn

    @property
    def calls(self):
        return self._fake.calls


def test_pre_1_0_19_default_resolves_to_leiden_alone():
    cm = load_module(mc_version="1.0.18")
    fake = _WithDefault(make_fake(), "cpm_leiden")
    alg, kwargs, label = cm.resolve_algorithm(fake, None, None)
    assert alg == ["cpm_leiden"] and kwargs is None
    assert label == "cpm_leiden"


def test_explicit_algorithm_overrides_the_library_default():
    cm = load_module()
    fake = _WithDefault(make_fake(), ["cpm_leiden", "k_centroids"])
    alg, kwargs, label = cm.resolve_algorithm(fake, ["cpm_leiden"], None)
    assert alg == ["cpm_leiden"]
    assert label == "cpm_leiden"


def test_kmeans_iterations_bounds_only_the_kmeans_step():
    cm = load_module()
    fake = _WithDefault(make_fake(), ["cpm_leiden", "k_centroids"])
    alg, kwargs, label = cm.resolve_algorithm(fake, None, 10)
    assert kwargs == {"k_centroids": {"n_iterations": 10}}
    assert "cpm_leiden" not in kwargs, "Leiden must not be given n_iterations"
    assert "n_iterations=10" in label


def test_kmeans_iterations_without_a_kmeans_step_is_an_error():
    """Silently ignoring it would leave the user thinking it was bounded."""
    cm = load_module()
    fake = _WithDefault(make_fake(), ["cpm_leiden", "k_centroids"])
    with pytest.raises(ValueError, match="no .*k-means step"):
        cm.resolve_algorithm(fake, ["cpm_leiden"], 10)


# --- cluster_with ------------------------------------------------------------


def test_single_algorithm_is_passed_as_a_string_not_a_list():
    """Pre-1.0.19 signatures type `algorithm` as str."""
    cm = load_module()
    fake = make_fake()
    cm.cluster_with(fake, ["cpm_leiden"], None, similarity_threshold=0.9)
    assert fake.calls[0]["algorithm"] == "cpm_leiden"


def test_multiple_algorithms_are_passed_as_a_list():
    cm = load_module()
    fake = make_fake()
    cm.cluster_with(fake, ["cpm_leiden", "k_centroids"], None,
                    similarity_threshold=0.9)
    assert fake.calls[0]["algorithm"] == ["cpm_leiden", "k_centroids"]


def test_algorithm_kwargs_are_dropped_on_an_older_install():
    """v1.0.18 has no algorithm_kwargs; passing it would raise TypeError."""
    cm = load_module(mc_version="1.0.18")
    fake = make_fake(supports_algorithm_kwargs=False)
    cm.cluster_with(fake, ["cpm_leiden"], {"k_centroids": {"n_iterations": 5}},
                    similarity_threshold=0.9)
    assert "algorithm_kwargs" not in fake.calls[0]


def test_weighted_cluster_on_passes_the_algorithm_through():
    """The across-model stage is the expensive one; it must not silently
    fall back to the library default."""
    cm = load_module()
    fake = make_fake()
    cm.weighted_cluster_on(
        fake, similarity_threshold=0.9, save_name="cluster_final",
        cluster_on="cluster_within_model", weight_col="num_seqlets",
        algorithm=["cpm_leiden"], algorithm_kwargs=None,
    )
    call = fake.calls[0]
    assert call["algorithm"] == "cpm_leiden"
    assert call["weight_col"] == "num_seqlets"


def test_weighted_cluster_on_uses_cluster_on_weight_when_thats_the_name():
    cm = load_module()
    fake = make_fake(weight_col_name="cluster_on_weight")
    cm.weighted_cluster_on(
        fake, similarity_threshold=0.9, save_name="cluster_final",
        cluster_on="cluster_within_model", weight_col="num_seqlets",
        algorithm=["cpm_leiden"], algorithm_kwargs=None,
    )
    assert fake.calls[0]["cluster_on_weight"] == "num_seqlets"


# --- provenance --------------------------------------------------------------


class MetadataMC:
    def __init__(self):
        self.metadata = pd.DataFrame({
            "cluster_final": [0, 0, 1],
            "num_seqlets": [10, 5, 7],
            "model": ["ENCSR1", "ENCSR2", "ENCSR1"],
            "posneg": ["pos", "pos", "pos"],
        })


def test_metadata_records_the_algorithm_and_library_version(tmp_path):
    cm = load_module()
    out = cm.write_cluster_metadata(
        MetadataMC(), "count", out_dir=tmp_path,
        provenance={"mc_version": "1.0.19",
                    "cluster_algorithm": "cpm_leiden+k_centroids",
                    "within_threshold": 0.95, "across_threshold": 0.90},
    )
    written = pd.read_csv(
        tmp_path / "motifcompendium_count_cluster_metadata.tsv", sep="\t")
    for col in ("mc_version", "cluster_algorithm", "within_threshold",
                "across_threshold"):
        assert col in written.columns
    assert set(written["cluster_algorithm"]) == {"cpm_leiden+k_centroids"}
    assert set(written["mc_version"]) == {"1.0.19"}
    assert len(written) == 2, "one row per cluster"
    assert "n_motifs" in out.columns


def test_metadata_is_unchanged_when_no_provenance_is_given(tmp_path):
    """Existing compendia were written without these columns; downstream
    readers must not require them."""
    cm = load_module()
    cm.write_cluster_metadata(MetadataMC(), "count", out_dir=tmp_path)
    written = pd.read_csv(
        tmp_path / "motifcompendium_count_cluster_metadata.tsv", sep="\t")
    assert "mc_version" not in written.columns
    assert "cluster_algorithm" not in written.columns
