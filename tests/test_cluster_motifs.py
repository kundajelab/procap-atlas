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
import warnings
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

    def __init__(self, default_algorithm, weight_col_name="weight_col"):
        self.calls = []
        self._default = default_algorithm
        self._weight_col_name = weight_col_name

    def cluster(self, **kwargs):
        self.calls.append(kwargs)

    @property
    def _signature_params(self):
        return ["similarity_threshold", "save_name", "cluster_on",
                "cluster_within", "algorithm", self._weight_col_name,
                "algorithm_kwargs"]


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
    """The default is version-dependent, so resolve_algorithm must read it off
    mc.cluster rather than assume one -- otherwise the next default change
    takes effect silently, as v1.0.19's did."""
    cm = load_module()
    fake = _WithDefault(make_fake(), ["cpm_leiden", "k_centroids"])
    alg, label = cm.resolve_algorithm(fake, None)
    assert alg == ["cpm_leiden", "k_centroids"]
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
    alg, label = cm.resolve_algorithm(fake, None)
    assert alg == ["cpm_leiden"]
    assert label == "cpm_leiden"


def test_explicit_algorithm_overrides_the_library_default():
    cm = load_module()
    fake = _WithDefault(make_fake(), ["cpm_leiden", "k_centroids"])
    alg, label = cm.resolve_algorithm(fake, ["cpm_leiden"])
    assert alg == ["cpm_leiden"]
    assert label == "cpm_leiden"


def test_the_refinement_is_never_capped():
    """v1.1.0's k_centroids bounds itself: it stops on a repeated membership,
    a stalled objective, or max_iterations=100 even at n_iterations=-1.
    Capping it here would only hide whether it converges, so nothing may
    inject n_iterations -- which is why --kmeans-iterations was removed."""
    cm = load_module()
    fake = make_fake()
    cm.cluster_with(fake, ["cpm_leiden", "k_centroids"],
                    similarity_threshold=0.9)
    assert fake.calls[0]["algorithm_kwargs"] is None
    assert "n_iterations" not in str(fake.calls[0])


# --- cluster_with ------------------------------------------------------------


def test_single_algorithm_is_passed_as_a_string_not_a_list():
    """Pre-1.0.19 signatures type `algorithm` as str."""
    cm = load_module()
    fake = make_fake()
    cm.cluster_with(fake, ["cpm_leiden"], similarity_threshold=0.9)
    assert fake.calls[0]["algorithm"] == "cpm_leiden"


def test_multiple_algorithms_are_passed_as_a_list():
    cm = load_module()
    fake = make_fake()
    cm.cluster_with(fake, ["cpm_leiden", "k_centroids"],
                    similarity_threshold=0.9)
    assert fake.calls[0]["algorithm"] == ["cpm_leiden", "k_centroids"]


def test_weighted_cluster_on_passes_the_algorithm_through():
    """The across-model stage is the expensive one; it must not silently
    fall back to the library default."""
    cm = load_module()
    fake = make_fake()
    cm.weighted_cluster_on(
        fake, similarity_threshold=0.9, save_name="cluster_final",
        cluster_on="cluster_within_model", weight_col="num_seqlets",
        algorithm=["cpm_leiden"],
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
        algorithm=["cpm_leiden"],
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


# --- provenance: version and cluster-average alignment frame ---
#
# MotifCompendium defines no __version__ on any branch (verified on v1.0.19
# and v1.1.0), so the original getattr(MotifCompendium, "__version__") always
# recorded "unknown" -- defeating the column added to catch version-driven
# default changes. The version lives only in setup.py, i.e. in the installed
# distribution metadata.


def test_mc_version_reads_distribution_metadata(monkeypatch):
    cluster_motifs = load_module()
    monkeypatch.setattr(
        cluster_motifs.importlib.metadata, "version", lambda name: "1.1.0"
    )
    assert cluster_motifs.mc_version() == "1.1.0"


def test_mc_version_survives_an_uninstalled_distribution(monkeypatch):
    """The stub sets __version__; a real install does not. Neither may raise."""
    cluster_motifs = load_module(mc_version="1.0.19")

    def missing(name):
        raise cluster_motifs.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cluster_motifs.importlib.metadata, "version", missing)
    assert cluster_motifs.mc_version() == "1.0.19"


def test_cluster_reference_reports_the_v1_1_0_default():
    cluster_motifs = load_module()

    class AveragesMC:
        def cluster_averages(self, clustering, reference="medoid"):
            pass

    assert cluster_motifs.cluster_reference(AveragesMC()) == "medoid"


def test_cluster_reference_flags_the_pre_v1_1_0_frame():
    """Before v1.1.0 the frame was the cluster's lowest-indexed member."""
    cluster_motifs = load_module()

    class OldMC:
        def cluster_averages(self, clustering, weight_col=None, aggregations=None):
            pass

    assert cluster_motifs.cluster_reference(OldMC()) == "first (pre-v1.1.0)"


# --- --algorithm-kwarg -------------------------------------------------------
#
# v1.1.0's _ConvergenceTracker stops at the first iteration whose objective
# fails to improve by more than tol=1e-9, even though the tracker exists
# because lossy averaging makes that objective non-monotone. The count-head
# build hit that rule, so its partition is the best-scoring iteration rather
# than a fixed point; k_centroids.tol=-inf is how that gets measured.


def test_algorithm_kwarg_parses_into_a_per_step_dict():
    cm = load_module()
    assert cm.parse_algorithm_kwargs(["k_centroids.tol=-inf"]) == {
        "k_centroids": {"tol": float("-inf")}
    }


def test_algorithm_kwarg_parses_scalar_types():
    cm = load_module()
    parsed = cm.parse_algorithm_kwargs([
        "k_centroids.max_iterations=250",
        "k_centroids.tol=1e-6",
        "k_centroids.reference=first",
    ])["k_centroids"]
    assert parsed["max_iterations"] == 250
    assert isinstance(parsed["max_iterations"], int)
    assert parsed["tol"] == pytest.approx(1e-6)
    assert parsed["reference"] == "first"


def test_no_algorithm_kwarg_leaves_library_defaults_alone():
    """Absent flag must mean "pass nothing", not "pass an empty dict"."""
    cm = load_module()
    assert cm.parse_algorithm_kwargs(None) is None
    assert cm.parse_algorithm_kwargs([]) is None


def test_malformed_algorithm_kwarg_is_an_error():
    cm = load_module()
    with pytest.raises(ValueError, match="ALGORITHM.KEY=VALUE"):
        cm.parse_algorithm_kwargs(["tol=-inf"])  # no algorithm prefix
    with pytest.raises(ValueError, match="ALGORITHM.KEY=VALUE"):
        cm.parse_algorithm_kwargs(["k_centroids.tol"])  # no value


def test_algorithm_kwarg_for_an_unrun_step_is_an_error():
    """Silently ignoring it would look like the setting had applied."""
    cm = load_module()
    fake = _WithDefault(make_fake(), ["cpm_leiden", "k_centroids"])
    with pytest.raises(ValueError, match="does not run"):
        cm.resolve_algorithm(fake, ["cpm_leiden"], {"k_centroids": {"tol": 0}})


def test_algorithm_kwarg_is_recorded_in_the_label():
    """The label becomes cluster_algorithm in cluster_metadata.tsv, so a
    tuned build must not be indistinguishable from an untuned one."""
    cm = load_module()
    fake = _WithDefault(make_fake(), ["cpm_leiden", "k_centroids"])
    _, label = cm.resolve_algorithm(
        fake, None, {"k_centroids": {"tol": float("-inf")}}
    )
    assert label == "cpm_leiden+k_centroids (k_centroids.tol=-inf)"


def test_algorithm_kwarg_reaches_mc_cluster():
    cm = load_module()
    fake = make_fake()
    cm.cluster_with(fake, ["cpm_leiden", "k_centroids"],
                    {"k_centroids": {"tol": float("-inf")}},
                    similarity_threshold=0.9)
    assert fake.calls[0]["algorithm_kwargs"] == {
        "k_centroids": {"tol": float("-inf")}
    }


# --- convergence accounting --------------------------------------------------
#
# k_centroids warns via warnings.warn, whose default filter prints once per
# code location. mc.cluster invokes k_centroids once per cluster_within group
# (one per experiment, ~219) plus once for cluster_on, so the single line seen
# in a real run means "at least one of ~220 calls", at an unknown stage. The
# stage is what matters: cluster_on produces cluster_final directly.


def _warn(message):
    warnings.warn(message, UserWarning)


def test_repeated_stalls_are_counted_not_deduplicated():
    """The whole point: the default filter would report this once."""
    cm = load_module()
    tally = {}
    with cm.record_convergence("within-model", tally):
        for _ in range(7):
            _warn("k_centroids: objective stopped improving; returning the "
                  "best-scoring iteration.")
    assert tally["within-model"]["stalled"] == 7


def test_each_warning_kind_is_counted_separately():
    cm = load_module()
    tally = {}
    with cm.record_convergence("across-model", tally):
        _warn("k_centroids: membership is cycling; returning the best.")
        _warn("k_centroids: objective stopped improving; returning the best.")
        _warn("k_centroids: did not converge within 100 iterations.")
        _warn("k_centroids: returning 948 clusters rather than the "
              "requested 950, as clusters left empty are dropped.")
    counts = tally["across-model"]
    assert (counts["cycling"], counts["stalled"]) == (1, 1)
    assert (counts["exhausted"], counts["emptied"]) == (1, 1)


def test_stages_are_tallied_separately():
    """A stall in one within-model group perturbs only that experiment; a
    stall in the across-model stage moves the atlas partition."""
    cm = load_module()
    tally = {}
    with cm.record_convergence("within-model", tally):
        _warn("k_centroids: objective stopped improving.")
    with cm.record_convergence("across-model", tally):
        pass
    assert tally["within-model"]["stalled"] == 1
    assert not tally["across-model"]
    assert cm.format_convergence(tally) == (
        "within-model: 1 stalled; across-model: converged"
    )


def test_unrelated_warnings_are_not_swallowed():
    cm = load_module()
    tally = {}
    with pytest.warns(DeprecationWarning, match="unrelated"):
        with cm.record_convergence("within-model", tally):
            warnings.warn("an unrelated deprecation", DeprecationWarning)
    assert not tally["within-model"]


def test_a_clean_run_says_so():
    cm = load_module()
    tally = {}
    with cm.record_convergence("within-model", tally):
        pass
    assert cm.format_convergence(tally) == "within-model: converged"


# --- force_merge_clusters -----------------------------------------------------


class _ForceMergeFake:
    """A stand-in for mc that returns scripted cluster_final memberships."""

    def __init__(self, memberships):
        self.calls = []
        self._memberships = list(memberships)
        self._read_idx = 0

    def __getitem__(self, key):
        assert key == "cluster_final"
        idx = min(self._read_idx, len(self._memberships) - 1)
        self._read_idx += 1
        return list(self._memberships[idx])


def _make_force_merge_fake(memberships, weight_col_name="weight_col"):
    fake = _ForceMergeFake(memberships)
    names = ["similarity_threshold", "save_name", "cluster_on",
             "cluster_within", "algorithm", weight_col_name,
             "algorithm_kwargs", "init_clustering_col"]
    src_params = ", ".join(f"{n}=None" for n in names)
    ns = {}
    exec(f"def cluster(self, {src_params}, **kwargs):\n"
         f"    self.calls.append(dict("
         f"{', '.join(f'{n}={n}' for n in names)}, **kwargs))\n", ns)
    fake.cluster = types.MethodType(ns["cluster"], fake)
    return fake


def test_force_merge_converges_immediately():
    """When DCC + k_centroids don't change membership, loop exits after one pair."""
    cm = load_module()
    fake = _make_force_merge_fake([
        [0, 0, 1, 1, 2],
        [0, 0, 1, 1, 2],
    ])
    convergence = cm.force_merge_clusters(fake, threshold=0.93)
    assert len(fake.calls) == 2


def test_force_merge_iterates_until_stable():
    """Membership changes after first iteration, stabilises on second."""
    cm = load_module()
    fake = _make_force_merge_fake([
        [0, 0, 1, 1, 2],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ])
    convergence = cm.force_merge_clusters(fake, threshold=0.93)
    assert len(fake.calls) == 4


def test_force_merge_dcc_step_kwargs():
    cm = load_module()
    fake = _make_force_merge_fake([[0, 1], [0, 1]])
    cm.force_merge_clusters(fake, threshold=0.93, density=1.0, seed=42)
    dcc_call = fake.calls[0]
    assert dcc_call["algorithm"] == "dcc"
    assert dcc_call["similarity_threshold"] == 0.93
    assert dcc_call["cluster_on"] == "cluster_final"
    assert dcc_call["weight_col"] == "num_seqlets"
    assert dcc_call["density"] == 1.0
    assert dcc_call["seed"] == 42


def test_force_merge_k_centroids_step_kwargs():
    cm = load_module()
    fake = _make_force_merge_fake([[0, 1], [0, 1]])
    cm.force_merge_clusters(fake, threshold=0.93)
    kc_call = fake.calls[1]
    assert kc_call["algorithm"] == "k_centroids"
    assert kc_call["init_clustering_col"] == "cluster_final"
    assert kc_call["weight_col"] == "num_seqlets"


def test_force_merge_uses_cluster_on_weight_on_older_versions():
    cm = load_module()
    fake = _make_force_merge_fake(
        [[0, 1], [0, 1]],
        weight_col_name="cluster_on_weight",
    )
    cm.force_merge_clusters(fake, threshold=0.93)
    dcc_call = fake.calls[0]
    assert dcc_call["cluster_on_weight"] == "num_seqlets"
    assert "weight_col" not in dcc_call


def test_force_merge_returns_convergence_per_iteration():
    cm = load_module()
    fake = _make_force_merge_fake([
        [0, 0, 1, 1, 2],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ])
    convergence = cm.force_merge_clusters(fake, threshold=0.93)
    assert "force-merge-0" in convergence
    assert "force-merge-1" in convergence


def test_force_merge_provenance_recorded(tmp_path):
    cm = load_module()
    cm.write_cluster_metadata(
        MetadataMC(), "count", out_dir=tmp_path,
        provenance={
            "mc_version": "1.1.0",
            "cluster_algorithm": "cpm_leiden+k_centroids",
            "within_threshold": 0.95,
            "across_threshold": 0.90,
            "force_merge_threshold": 0.93,
            "force_merge_density": 1.0,
        },
    )
    written = pd.read_csv(
        tmp_path / "motifcompendium_count_cluster_metadata.tsv", sep="\t")
    assert "force_merge_threshold" in written.columns
    assert written["force_merge_threshold"].iloc[0] == pytest.approx(0.93)
    assert written["force_merge_density"].iloc[0] == pytest.approx(1.0)
