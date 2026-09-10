"""Tests for the Fig. 2 motif-atlas panels.

The real inputs (MotifCompendium cluster metadata, per-experiment
hits_linked.tsv) are produced on the cluster, so these tests build synthetic
fixtures that match those schemas exactly -- cluster_motifs.py's
`cluster_final/posneg/experiments` metadata columns and
link_hits_to_compendium.py's `compendium_motif_name` hits column -- and check
the panel arithmetic against them.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "src" / "bpnet" / "hitcall"))

import _biosample_groups as bg  # noqa: E402
import plot_motif_rarefaction as rare  # noqa: E402

pytest.importorskip("seaborn")
import motif_hit_density as mhd  # noqa: E402


# --------------------------------------------------------------------------
# biosample grouping
# --------------------------------------------------------------------------


def test_metastases_are_not_routed_to_the_destination_organ():
    """Metastatic biosamples are named for where they spread *to*, so an
    organ rule must never claim them ahead of the metastasis rule."""
    assert bg.assign_group("Metastatic Breast Carcinoma in the Brain") == "metastatic_carcinoma"
    assert bg.assign_group("Colon Carcinoma Metastatic in the Lung") == "metastatic_carcinoma"
    assert bg.assign_group("Metastatic Liver Carcinoma in the Adrenal Gland") == "metastatic_carcinoma"


@pytest.mark.parametrize(
    "biosample,expected",
    [
        ("cerebellum", "neural"),
        ("heart left ventricle", "heart"),
        ("right lobe of liver", "liver_biliary"),
        ("HCT116", "gi_tract"),
        ("K562", "blood_immune"),
        ("peripheral blood mononuclear cell", "blood_immune"),
        ("endothelial cell of umbilical vein", "vascular"),
        ("body of pancreas", "pancreas"),
        ("HEK293T", "hek"),
        ("some novel biosample", "other"),
    ],
)
def test_group_assignment(biosample, expected):
    assert bg.assign_group(biosample) == expected


def test_override_tsv_wins_but_unlisted_biosamples_still_fall_back(tmp_path):
    override = tmp_path / "groups.tsv"
    override.write_text("biosample\tgroup\nK562\tmy_custom_group\n")
    experiments = {
        "EXP1": {"biosample": "K562"},
        "EXP2": {"biosample": "cerebellum"},
    }
    group_map, biosample_map = bg.load_group_map(experiments, override, quiet=True)
    assert group_map == {"EXP1": "my_custom_group", "EXP2": "neural"}
    assert biosample_map["EXP1"] == "K562"


def test_write_group_tsv_roundtrips(tmp_path):
    experiments = {"E1": {"biosample": "liver"}, "E2": {"biosample": "liver"}}
    group_map, biosample_map = bg.load_group_map(experiments, quiet=True)
    out = tmp_path / "t.tsv"
    bg.write_group_tsv(biosample_map, group_map, out)
    table = pd.read_csv(out, sep="\t", comment="#")
    assert table.loc[0, "n_experiments"] == 2
    assert table.loc[0, "group"] == "liver_biliary"


# --------------------------------------------------------------------------
# rarefaction
# --------------------------------------------------------------------------


def write_cluster_metadata(path, rows, with_jaspar=True):
    """Write a cluster_motifs.py-shaped cluster metadata TSV."""
    records = []
    for cluster_final, (posneg, exps, jaspar) in enumerate(rows):
        rec = {
            "cluster_final": cluster_final,
            "n_motifs": len(exps),
            "total_seqlets": 100 * len(exps),
            "n_experiments": len(exps),
            "experiments": ",".join(sorted(exps)),
            "posneg": posneg,
        }
        if with_jaspar:
            rec["jaspar_name"] = jaspar
            rec["jaspar_score"] = 0.9 if jaspar else np.nan
        records.append(rec)
    pd.DataFrame(records).to_csv(path, sep="\t", index=False)
    return path


def test_load_presence_parses_and_filters(tmp_path):
    meta_path = write_cluster_metadata(
        tmp_path / "m.tsv",
        [
            ("pos", ["E1", "E2", "E3"], "SP1"),
            ("pos", ["E4"], None),  # dropped entirely by the keep set below
            ("neg", ["E2", "E4"], "GATA1"),
        ],
    )
    meta, universe = rare.load_presence(meta_path, keep_experiments={"E1", "E2", "E3"})
    assert universe == ["E1", "E2", "E3"]
    # cluster 1 lived only in E4 and is unreachable in this universe
    assert list(meta["cluster_final"]) == [0, 2]
    assert list(meta["prevalence"]) == [3, 1]


def test_load_presence_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"cluster_final": [0]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="experiments"):
        rare.load_presence(bad)


def test_uniform_curve_exact_endpoints():
    """At k=1 the expectation is sum(p/N); at k=N every cluster is found."""
    prevalences = np.array([1, 2, 5, 5, 10])
    n = 20
    curve = rare.uniform_curve_exact(prevalences, n)
    assert curve[0] == pytest.approx(prevalences.sum() / n)
    assert curve[-1] == pytest.approx(len(prevalences))
    assert np.all(np.diff(curve) >= -1e-9)  # monotone non-decreasing


def test_uniform_curve_exact_matches_monte_carlo():
    """The closed form must agree with brute-force subsampling."""
    rng = np.random.default_rng(0)
    n = 12
    exp_ids = [f"E{i}" for i in range(n)]
    exp_sets = [
        set(rng.choice(exp_ids, size=int(p), replace=False))
        for p in (1, 3, 3, 6, 9)
    ]
    prevalences = np.array([len(s) for s in exp_sets])
    exact = rare.uniform_curve_exact(prevalences, n)

    groups = {e: "g" for e in exp_ids}
    mc = rare.rarefy(exp_sets, groups, "uniform", n_reps=4000, rng=rng).mean(axis=0)
    assert np.allclose(exact, mc, atol=0.12)


def test_rarefy_curves_are_monotone_and_complete():
    exp_sets = [{"E1", "E2"}, {"E3"}, {"E2", "E3", "E4"}]
    groups = {"E1": "a", "E2": "a", "E3": "b", "E4": "b"}
    for scheme in ("uniform", "diverse", "redundant"):
        curves = rare.rarefy(exp_sets, groups, scheme, 25, np.random.default_rng(1))
        assert curves.shape == (25, 4)
        assert np.all(np.diff(curves, axis=1) >= 0), scheme
        assert np.all(curves[:, -1] == 3), scheme  # all clusters found at k=N


def test_diverse_sampling_beats_redundant_when_motifs_are_group_specific():
    """The panel's actual claim: with a group-specific lexicon, spreading
    experiments across tissues recovers more motifs than piling into one."""
    groups = {}
    exp_sets = []
    for gi, group in enumerate("abcdefgh"):
        members = [f"E{gi}_{j}" for j in range(4)]
        for m in members:
            groups[m] = group
        # 5 motifs private to this group, all present in all of its members
        for _ in range(5):
            exp_sets.append(set(members))
    rng = np.random.default_rng(7)
    diverse = rare.rarefy(exp_sets, groups, "diverse", 60, rng).mean(axis=0)
    redundant = rare.rarefy(exp_sets, groups, "redundant", 60, rng).mean(axis=0)
    mid = len(diverse) // 4
    assert diverse[mid] > redundant[mid] * 1.5


def test_classify_clusters_jaspar_fallback(tmp_path):
    meta_path = write_cluster_metadata(
        tmp_path / "m.tsv",
        [("pos", ["E1"], "SP1"), ("pos", ["E1"], None)],
    )
    meta, _ = rare.load_presence(meta_path)
    classes = rare.classify_clusters(meta, None)
    assert list(classes) == ["TF-matched", "unmatched (core promoter / repeat)"]


def test_classify_clusters_prefers_curated_annotation(tmp_path):
    meta_path = write_cluster_metadata(
        tmp_path / "m.tsv", [("pos", ["E1"], "SP1"), ("pos", ["E1"], None)]
    )
    ann = tmp_path / "ann.tsv"
    ann.write_text("cluster_final\tclass\n0\tcore promoter\n")
    meta, _ = rare.load_presence(meta_path)
    classes = rare.classify_clusters(meta, ann)
    assert list(classes) == ["core promoter", "unannotated"]


# --------------------------------------------------------------------------
# hit density
# --------------------------------------------------------------------------


def make_hitcall_tree(root, experiment, head, motif_counts, n_peaks):
    """Write a hits_linked.tsv + peaks.narrowPeak pair for one experiment."""
    exp_dir = root / f"{experiment}_{head}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for motif, count in motif_counts.items():
        for i in range(count):
            rows.append(
                {
                    "chr": "chr1", "start": 100 + i, "end": 110 + i,
                    "motif_name": "pos_patterns.pattern_0",
                    "strand": "+", "peak_id": i % n_peaks,
                    "compendium_motif_name": motif,
                }
            )
    pd.DataFrame(rows).to_csv(exp_dir / "hits_linked.tsv", sep="\t", index=False)
    with open(exp_dir / "peaks.narrowPeak", "w") as f:
        for i in range(n_peaks):
            f.write(f"chr1\t{i * 1000}\t{i * 1000 + 2114}\tpeak{i}\t0\t.\t0\t0\t0\t1057\n")
    return exp_dir


def test_count_peaks_prefers_narrowpeak(tmp_path):
    root = tmp_path / "hitcalls"
    exp_dir = make_hitcall_tree(root, "EXP1", "profile", {"pos_patterns.0": 5}, n_peaks=40)
    hits = pd.read_csv(exp_dir / "hits_linked.tsv", sep="\t")
    n, source = mhd.count_peaks(exp_dir, hits)
    assert (n, source) == (40, "peaks.narrowPeak")


def test_count_peaks_flags_the_fallback(tmp_path):
    """Without peaks.narrowPeak the count is derived from hits and known to be
    an underestimate, so the provenance string has to say so."""
    exp_dir = tmp_path / "EXP1_profile"
    exp_dir.mkdir()
    hits = pd.DataFrame({"peak_id": [0, 1, 2], "compendium_motif_name": ["pos_patterns.0"] * 3})
    n, source = mhd.count_peaks(exp_dir, hits)
    assert n == 3
    assert "underestimate" in source


def test_collect_densities_normalizes_per_peak(tmp_path):
    root = tmp_path / "hitcalls"
    make_hitcall_tree(root, "EXP1", "profile", {"pos_patterns.0": 20, "pos_patterns.1": 10}, 100)
    make_hitcall_tree(root, "EXP2", "profile", {"pos_patterns.0": 5}, 50)
    density, info = mhd.collect_densities(
        ["EXP1", "EXP2", "EXP_MISSING"], "profile", None, hitcall_dir=root, quiet=True
    )
    assert density.loc["pos_patterns.0", "EXP1"] == pytest.approx(0.20)
    assert density.loc["pos_patterns.1", "EXP1"] == pytest.approx(0.10)
    assert density.loc["pos_patterns.0", "EXP2"] == pytest.approx(0.10)
    # absent motif in EXP2 becomes a zero, not a NaN
    assert density.loc["pos_patterns.1", "EXP2"] == 0.0
    assert "EXP_MISSING" not in density.columns
    assert info.loc["EXP1", "n_peaks"] == 100


def test_discovery_status_separates_structural_zeros(tmp_path):
    root = tmp_path / "hitcalls"
    make_hitcall_tree(root, "EXP1", "profile", {"pos_patterns.0": 4}, 20)
    make_hitcall_tree(root, "EXP2", "profile", {"pos_patterns.1": 4}, 20)
    density, _ = mhd.collect_densities(
        ["EXP1", "EXP2"], "profile", None, hitcall_dir=root, quiet=True
    )
    meta = write_cluster_metadata(
        tmp_path / "m.tsv",
        [("pos", ["EXP1", "EXP2"], "SP1"), ("pos", ["EXP2"], "GATA1")],
    )
    status = mhd.discovery_status(density, meta)
    # cluster 0 was discovered in both, but only EXP1 has hits for it: a real
    # zero in EXP2, not a discovery failure.
    assert status.loc["pos_patterns.0", "EXP2"] == "discovered"
    assert density.loc["pos_patterns.0", "EXP2"] == 0.0
    # cluster 1 was never discovered in EXP1 at all: structurally zero.
    assert status.loc["pos_patterns.1", "EXP1"] == "undiscovered"


def test_specificity_is_zero_for_ubiquitous_and_one_for_private():
    columns = ["E1", "E2", "E3", "E4"]
    group_map = {"E1": "liver", "E2": "liver", "E3": "neural", "E4": "heart"}
    density = pd.DataFrame(
        {
            "E1": [1.0, 1.0, 0.0],
            "E2": [1.0, 0.0, 0.0],
            "E3": [1.0, 0.0, 2.0],
            "E4": [1.0, 0.0, 0.0],
        },
        index=["ubiquitous", "liver_only", "neural_only"],
    )[columns]
    spec = mhd.specificity_table(density, group_map)
    assert spec.loc["ubiquitous", "specificity"] == pytest.approx(0.0, abs=1e-9)
    assert spec.loc["liver_only", "specificity"] == pytest.approx(1.0)
    assert spec.loc["neural_only", "specificity"] == pytest.approx(1.0)
    assert spec.loc["liver_only", "top_group"] == "liver"
    assert spec.loc["neural_only", "top_group"] == "neural"
    assert spec.loc["ubiquitous", "n_groups_detected"] == 3


def test_specificity_averages_within_group_before_scoring():
    """A motif present in every group is ubiquitous even when one group is
    replicated far more than the others -- the panel's whole reason for
    scoring over groups rather than experiments."""
    group_map = {f"H{i}": "gi_tract" for i in range(16)}
    group_map.update({"L1": "liver_biliary", "N1": "neural"})
    density = pd.DataFrame(
        {c: [1.0] for c in group_map}, index=["everywhere"]
    )
    spec = mhd.specificity_table(density, group_map)
    assert spec.loc["everywhere", "specificity"] == pytest.approx(0.0, abs=1e-9)


def test_annotate_labels_adds_jaspar_names(tmp_path):
    meta = write_cluster_metadata(
        tmp_path / "m.tsv", [("pos", ["E1"], "SP1"), ("neg", ["E1"], None)]
    )
    labels = mhd.annotate_labels(
        pd.Index(["pos_patterns.0", "neg_patterns.1"]), meta
    )
    assert labels["pos_patterns.0"] == "pos_patterns.0 (SP1)"
    assert labels["neg_patterns.1"] == "neg_patterns.1"


def test_depth_colors_match_figure_1d_tiers():
    colors = mhd.depth_colors(
        {"A": 5e6, "B": 15e6, "C": 30e6}, ["A", "B", "C", "D"]
    )
    assert colors["A"] == "#d73027"  # <10M
    assert colors["B"] == "#fee090"  # 10-20M
    assert colors["C"] == "#4575b4"  # >20M
    assert colors["D"] == "#cccccc"  # unknown


# --------------------------------------------------------------------------
# end-to-end, against real accessions from the repo config
# --------------------------------------------------------------------------


def real_experiments(n):
    import yaml

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "experiment_config.yaml"))
    read_counts = rare.load_read_counts()
    picked = [
        e for e in cfg["experiments"]
        if read_counts.get(e, 0) >= 10e6
        and "uncapped" not in str(cfg["experiments"][e].get("library_construction", "")).lower()
        and e != "ENCSR973QQI"
    ]
    return picked[:n]


def test_rarefaction_cli_end_to_end(tmp_path):
    exps = real_experiments(12)
    rows = [("pos", exps, "SP1")]  # ubiquitous
    rows += [("pos", exps[:2], "GATA1"), ("pos", exps[2:4], "HNF4A")]
    rows += [("neg", [e], None) for e in exps]  # singletons
    meta = write_cluster_metadata(tmp_path / "meta.tsv", rows)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/plot_motif_rarefaction.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
            "--n-reps", "20",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    curves = pd.read_csv(tmp_path / "out" / "motif_rarefaction_profile.tsv", sep="\t")
    full = curves[
        (curves["scheme"] == "uniform") & (curves["motif_class"] == "__all__")
    ].sort_values("k")
    assert full["mean"].iloc[-1] == pytest.approx(len(rows))
    assert (tmp_path / "out" / "motif_rarefaction_profile.png").exists()
    assert (tmp_path / "out" / "motif_prevalence_profile.tsv").exists()


def test_hit_density_cli_end_to_end(tmp_path):
    exps = real_experiments(8)
    root = tmp_path / "hitcalls"
    for i, exp in enumerate(exps):
        counts = {"pos_patterns.0": 50}
        if i < 3:
            counts["pos_patterns.1"] = 30
        make_hitcall_tree(root, exp, "profile", counts, n_peaks=100)
    meta = write_cluster_metadata(
        tmp_path / "meta.tsv",
        [("pos", exps, "SP1"), ("pos", exps[:3], "GATA1")],
    )

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_hit_density.py"),
            "--hitcall-dir", str(root), "--cluster-metadata", str(meta),
            "--out-dir", str(tmp_path / "out"), "--min-experiments", "2",
            "--mask-undiscovered",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    density = pd.read_csv(out / "motif_hit_density_profile.tsv", sep="\t", index_col=0)
    assert density.loc["pos_patterns.0"].eq(0.5).all()
    spec = pd.read_csv(out / "motif_specificity_profile.tsv", sep="\t", index_col=0)
    assert spec.loc["pos_patterns.1", "specificity"] > spec.loc["pos_patterns.0", "specificity"]
    status = pd.read_csv(out / "motif_hit_density_profile_status.tsv", sep="\t", index_col=0)
    assert (status.loc["pos_patterns.1", exps[3:]] == "undiscovered").all()
    assert (out / "motif_hit_density_profile.png").exists()
    assert (out / "motif_hit_density_profile_columns.tsv").exists()


def test_balanced_selection_keeps_tissue_restricted_motifs():
    """A restricted motif that is heavily used in its own tissues must survive
    selection, even though its summed density is far below that of a weak
    motif spread over every experiment."""
    spec = pd.DataFrame(
        {
            # broad but weak: present in 100 experiments at 0.10 each
            "total_hits_per_peak": [10.0] * 6 + [3.5, 3.5],
            "mean_hits_per_peak_detected": [0.10] * 6 + [0.35, 0.35],
            "n_experiments_detected": [100] * 6 + [10, 10],
            "specificity": [0.02] * 6 + [0.95, 0.92],
        },
        index=[f"broad{i}" for i in range(6)] + ["lineage_a", "lineage_b"],
    )
    picked = mhd.select_clusters(spec, top_n=4, sort_by="balanced")
    assert "lineage_a" in picked.index
    assert "lineage_b" in picked.index
    assert len(picked) == 4


def test_balanced_selection_falls_back_when_pool_is_thin():
    spec = pd.DataFrame(
        {
            "total_hits_per_peak": [5.0, 4.0],
            "mean_hits_per_peak_detected": [0.5, 0.4],
            "n_experiments_detected": [10, 10],
            "specificity": [0.1, 0.9],
        },
        index=["a", "b"],
    )
    picked = mhd.select_clusters(spec, top_n=10, sort_by="balanced")
    assert set(picked.index) == {"a", "b"}


def test_sort_by_total_hits_still_available():
    spec = pd.DataFrame(
        {
            "total_hits_per_peak": [1.0, 9.0],
            "mean_hits_per_peak_detected": [0.1, 0.9],
            "n_experiments_detected": [10, 10],
            "specificity": [0.9, 0.1],
        },
        index=["low", "high"],
    )
    picked = mhd.select_clusters(spec, top_n=1, sort_by="total_hits")
    assert list(picked.index) == ["high"]


def write_pattern_to_cluster(path, rows):
    """Write a cluster_motifs.py-shaped pattern-to-cluster mapping TSV.

    `rows` is the same (posneg, experiments, jaspar) shape as
    write_cluster_metadata, so the two loaders can be compared directly.
    """
    records = []
    for cluster_final, (posneg, exps, _jaspar) in enumerate(rows):
        for i, exp in enumerate(sorted(exps)):
            records.append(
                {
                    "experiment": exp,
                    "local_motif_name": f"{posneg}_patterns.pattern_{i}",
                    "compendium_motif_name": f"{posneg}_patterns.{cluster_final}",
                }
            )
    pd.DataFrame(records).to_csv(path, sep="\t", index=False)
    return path


def test_mapping_loader_matches_metadata_loader(tmp_path):
    """The two inputs must give identical presence sets -- that equivalence is
    the whole reason the mapping is a safe early substitute."""
    rows = [
        ("pos", ["E1", "E2", "E3"], "SP1"),
        ("neg", ["E2", "E4"], "GATA1"),
        ("pos", ["E4"], None),
    ]
    meta_path = write_cluster_metadata(tmp_path / "meta.tsv", rows)
    map_path = write_pattern_to_cluster(tmp_path / "map.tsv", rows)

    from_meta, universe_meta = rare.load_presence(meta_path)
    from_map, universe_map = rare.load_presence_from_mapping(map_path)

    assert universe_meta == universe_map

    # Compare on content keyed by cluster identity, not row order: the
    # metadata TSV is sorted by total_seqlets while the mapping is grouped by
    # motif name, and no curve depends on row order.
    def keyed(df):
        return {
            int(row["cluster_final"]): (row["posneg"], row["prevalence"], row["exp_set"])
            for _, row in df.iterrows()
        }

    assert keyed(from_meta) == keyed(from_map)
    assert keyed(from_map)[0][0] == "pos"
    assert keyed(from_map)[1][0] == "neg"


def test_mapping_loader_respects_keep_set(tmp_path):
    rows = [("pos", ["E1", "E2", "E3"], "SP1"), ("pos", ["E4"], None)]
    map_path = write_pattern_to_cluster(tmp_path / "map.tsv", rows)
    meta, universe = rare.load_presence_from_mapping(
        map_path, keep_experiments={"E1", "E2"}
    )
    assert universe == ["E1", "E2"]
    assert len(meta) == 1  # the E4-only cluster is unreachable
    assert meta.loc[0, "prevalence"] == 2


def test_mapping_loader_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"experiment": ["E1"]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="compendium_motif_name"):
        rare.load_presence_from_mapping(bad)


def test_classify_clusters_single_class_without_jaspar(tmp_path):
    """Mapping-derived tables have no JASPAR column, so stratification must
    collapse to one class rather than crashing."""
    rows = [("pos", ["E1", "E2"], "SP1"), ("neg", ["E2"], None)]
    map_path = write_pattern_to_cluster(tmp_path / "map.tsv", rows)
    meta, _ = rare.load_presence_from_mapping(map_path)
    classes = rare.classify_clusters(meta, None)
    assert set(classes) == {"all motifs"}


def test_rarefaction_cli_accepts_pattern_to_cluster(tmp_path):
    exps = real_experiments(12)
    rows = [("pos", exps, "SP1"), ("pos", exps[:3], "GATA1")]
    rows += [("neg", [e], None) for e in exps]
    map_path = write_pattern_to_cluster(tmp_path / "map.tsv", rows)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/plot_motif_rarefaction.py"),
            "--pattern-to-cluster", str(map_path),
            "--out-dir", str(tmp_path / "out"), "--n-reps", "20",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    curves = pd.read_csv(tmp_path / "out" / "motif_rarefaction_profile.tsv", sep="\t")
    full = curves[
        (curves["scheme"] == "uniform") & (curves["motif_class"] == "__all__")
    ].sort_values("k")
    assert full["mean"].iloc[-1] == pytest.approx(len(rows))
    assert (tmp_path / "out" / "motif_rarefaction_profile.png").exists()


# --------------------------------------------------------------------------
# abundance-threshold sweep
# --------------------------------------------------------------------------


def write_cluster_metadata_with_seqlets(path, rows):
    """Like write_cluster_metadata but with explicit per-cluster seqlet totals.

    `rows` is (posneg, experiments, jaspar, total_seqlets).
    """
    records = []
    for cluster_final, (posneg, exps, jaspar, total_seqlets) in enumerate(rows):
        records.append(
            {
                "cluster_final": cluster_final,
                "n_motifs": len(exps),
                "total_seqlets": total_seqlets,
                "n_experiments": len(exps),
                "experiments": ",".join(sorted(exps)),
                "posneg": posneg,
                "jaspar_name": jaspar,
                "jaspar_score": 0.9 if jaspar else np.nan,
            }
        )
    pd.DataFrame(records).to_csv(path, sep="\t", index=False)
    return path


def test_load_presence_computes_seqlets_per_motif(tmp_path):
    meta_path = write_cluster_metadata_with_seqlets(
        tmp_path / "m.tsv",
        [("pos", ["E1", "E2", "E3", "E4"], "SP1", 400), ("pos", ["E1", "E2"], None, 40)],
    )
    meta, _ = rare.load_presence(meta_path)
    assert list(meta["seqlets_per_motif"]) == [100.0, 20.0]


def test_sweep_requires_seqlet_columns(tmp_path):
    """The pattern-to-cluster fallback carries no seqlet counts, so the sweep
    must refuse rather than silently sweeping nothing."""
    map_path = write_pattern_to_cluster(
        tmp_path / "map.tsv", [("pos", ["E1", "E2"], None)]
    )
    meta, _ = rare.load_presence_from_mapping(map_path)
    with pytest.raises(ValueError, match="cluster_metadata"):
        rare.sweep_abundance(meta, 2, [0.0, 100.0], [1])


def build_sweep_meta(n_total=100, n_broad=40, n_narrow=60):
    """Clusters where abundance tracks prevalence, as on the real compendium.

    Broad clusters are prevalent AND abundant; narrow ones are restricted AND
    sparse. That coupling is what makes an abundance floor act as a prevalence
    floor.
    """
    exps = [f"E{i}" for i in range(n_total)]
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(n_broad):
        members = set(rng.choice(exps, size=80, replace=False))
        rows.append({"exp_set": members, "prevalence": 80, "seqlets_per_motif": 900.0})
    for _ in range(n_narrow):
        members = set(rng.choice(exps, size=4, replace=False))
        rows.append({"exp_set": members, "prevalence": 4, "seqlets_per_motif": 30.0})
    return pd.DataFrame(rows), n_total


def test_sweep_shrinks_lexicon_monotonically():
    meta, n_total = build_sweep_meta()
    _, summary = rare.sweep_abundance(meta, n_total, [0.0, 100.0, 1000.0], [5])
    # the 1000 threshold drops everything, leaving <10 clusters -> skipped
    assert list(summary["min_seqlets_per_motif"]) == [0.0, 100.0]
    assert list(summary["n_clusters"]) == [100, 40]


def test_sweep_fraction_reaches_one_at_full_atlas():
    meta, n_total = build_sweep_meta()
    curves, _ = rare.sweep_abundance(meta, n_total, [0.0, 100.0], [5])
    for threshold, sub in curves.groupby("min_seqlets_per_motif"):
        assert sub.sort_values("k")["fraction"].iloc[-1] == pytest.approx(1.0)


def test_sweep_reproduces_the_confound_direction():
    """The sweep's whole finding: because abundance tracks prevalence, raising
    the floor raises the fraction a small sample recovers. If this ever
    reverses, the interpretation in the docstring is wrong."""
    meta, n_total = build_sweep_meta()
    _, summary = rare.sweep_abundance(meta, n_total, [0.0, 100.0], [5])
    low = summary.set_index("min_seqlets_per_motif").loc[0.0, "fraction_at_k5"]
    high = summary.set_index("min_seqlets_per_motif").loc[100.0, "fraction_at_k5"]
    assert high > low


def test_sweep_cli_end_to_end(tmp_path):
    exps = real_experiments(30)
    rows = [("pos", exps, "SP1", 30 * 900) for _ in range(8)]
    rows += [("pos", exps[:20], f"BROAD{i}", 20 * 300) for i in range(8)]
    rows += [("pos", exps[i : i + 3], f"TF{i}", 3 * 30) for i in range(0, 24, 3)]
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/plot_motif_rarefaction.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
            "--n-reps", "10", "--sweep",
            "--sweep-thresholds", "0", "100", "--sweep-marks", "1", "5",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    assert (out / "motif_rarefaction_sweep_profile.png").exists()
    summary = pd.read_csv(
        out / "motif_rarefaction_sweep_profile_summary.tsv", sep="\t"
    )
    assert list(summary["min_seqlets_per_motif"]) == [0.0, 100.0]
    assert summary.loc[1, "n_clusters"] < summary.loc[0, "n_clusters"]
    # the confound diagnostic and weakest-form claim are both reported
    assert "prevalence vs seqlets/motif r =" in result.stderr
    assert "Weakest-form claim" in result.stderr


def test_min_seqlets_per_motif_filters_and_reports(tmp_path):
    exps = real_experiments(12)
    rows = [("pos", exps, "SP1", 12 * 900)] * 3
    rows += [("pos", exps[:3], "TF", 3 * 20)] * 3
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/plot_motif_rarefaction.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
            "--n-reps", "10", "--min-seqlets-per-motif", "100",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "kept 3/6 clusters" in result.stderr


def test_min_seqlets_per_motif_rejected_without_metadata(tmp_path):
    exps = real_experiments(6)
    rows = [("pos", exps, None), ("pos", exps[:2], None)]
    map_path = write_pattern_to_cluster(tmp_path / "map.tsv", rows)
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/plot_motif_rarefaction.py"),
            "--pattern-to-cluster", str(map_path),
            "--out-dir", str(tmp_path / "out"), "--min-seqlets-per-motif", "50",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "needs total_seqlets and n_motifs" in result.stderr


# --------------------------------------------------------------------------
# group concentration
# --------------------------------------------------------------------------

import motif_group_concentration as mgc  # noqa: E402


def test_expected_n_groups_matches_monte_carlo():
    rng = np.random.default_rng(0)
    sizes = np.array([40, 25, 20, 10, 5, 5, 3, 2])
    n_total = int(sizes.sum())
    labels = np.repeat(np.arange(len(sizes)), sizes)
    for p in (2, 5, 20):
        exact = mgc.expected_n_groups(p, sizes, n_total)
        draws = [
            len(set(rng.choice(labels, size=p, replace=False))) for _ in range(4000)
        ]
        assert exact == pytest.approx(np.mean(draws), abs=0.08), p


def test_prob_single_group_matches_closed_form_at_p2():
    """At p=2 the probability reduces to sum n_g(n_g-1) / N(N-1), which is the
    hand calculation used to check the real data."""
    sizes = np.array([41, 28, 21, 18, 15, 11, 8, 8, 8, 7, 7, 6, 5, 4, 3, 3, 2, 2, 1])
    n_total = int(sizes.sum())
    expected = (sizes * (sizes - 1)).sum() / (n_total * (n_total - 1))
    assert mgc.prob_single_group(2, sizes, n_total) == pytest.approx(expected)


def test_prob_single_group_zero_when_no_group_is_large_enough():
    sizes = np.array([3, 3, 3])
    assert mgc.prob_single_group(4, sizes, 9) == pytest.approx(0.0)


def test_poisson_binomial_is_exact_not_approximate():
    probs = np.array([0.1, 0.5, 0.9])
    # P[X >= 0] = 1; P[X >= 3] = product
    assert mgc.poisson_binomial_sf(probs, 0) == pytest.approx(1.0)
    assert mgc.poisson_binomial_sf(probs, 3) == pytest.approx(0.1 * 0.5 * 0.9)
    # P[X >= 1] = 1 - product of complements
    assert mgc.poisson_binomial_sf(probs, 1) == pytest.approx(
        1 - (0.9 * 0.5 * 0.1)
    )


def test_poisson_binomial_mean_matches_sum_of_probs():
    rng = np.random.default_rng(4)
    probs = rng.uniform(0, 0.3, 40)
    dist = np.array(
        [
            mgc.poisson_binomial_sf(probs, k) - mgc.poisson_binomial_sf(probs, k + 1)
            for k in range(len(probs) + 1)
        ]
    )
    assert dist.sum() == pytest.approx(1.0)
    mean = (dist * np.arange(len(probs) + 1)).sum()
    assert mean == pytest.approx(probs.sum())


def build_concentration_meta(group_map, concentrated, prevalence):
    """One cluster per entry in `concentrated`; True = all experiments drawn
    from a single group, False = drawn across groups."""
    by_group = {}
    for e, g in group_map.items():
        by_group.setdefault(g, []).append(e)
    groups = sorted(by_group)
    rows = []
    for i, conc in enumerate(concentrated):
        if conc:
            pool = by_group[groups[i % len(groups)]]
            members = set(pool[:prevalence])
        else:
            members = {by_group[groups[(i + j) % len(groups)]][0] for j in range(prevalence)}
        rows.append({"exp_set": members, "prevalence": len(members)})
    return pd.DataFrame(rows)


def test_concentration_is_one_for_random_spread_and_low_for_concentrated():
    group_map = {f"g{g}_e{i}": f"g{g}" for g in range(8) for i in range(5)}
    n_total = len(group_map)
    spread = build_concentration_meta(group_map, [False] * 8, prevalence=4)
    conc = build_concentration_meta(group_map, [True] * 8, prevalence=4)

    a_spread = mgc.annotate_concentration(spread, group_map, n_total)
    a_conc = mgc.annotate_concentration(conc, group_map, n_total)

    # spread clusters hit one group per experiment -> at or above expectation
    assert a_spread["concentration"].median() > 1.0
    assert (a_spread["n_groups"] == 4).all()
    # concentrated clusters sit in exactly one group -> well below
    assert (a_conc["n_groups"] == 1).all()
    assert a_conc["concentration"].median() < 0.4
    assert a_conc["is_single_group"].all()


def test_summary_reports_single_group_enrichment():
    group_map = {f"g{g}_e{i}": f"g{g}" for g in range(8) for i in range(5)}
    n_total = len(group_map)
    meta = pd.concat(
        [
            build_concentration_meta(group_map, [True] * 6, prevalence=3),
            build_concentration_meta(group_map, [False] * 6, prevalence=3),
        ],
        ignore_index=True,
    )
    annotated = mgc.annotate_concentration(meta, group_map, n_total)
    annotated["motif_class"] = ["conc"] * 6 + ["spread"] * 6
    summary = mgc.summarize(annotated, ["motif_class"]).set_index("motif_class")

    assert summary.loc["conc", "n_single_group"] == 6
    assert summary.loc["spread", "n_single_group"] == 0
    assert summary.loc["conc", "single_group_p"] < 1e-6
    assert summary.loc["conc", "single_group_enrichment"] > 5
    assert summary.loc["spread", "single_group_p"] == pytest.approx(1.0)


def test_group_level_changes_the_expectation():
    """Coarse grouping lowers E[n_groups], which is why a conclusion must hold
    at both levels before it can be trusted."""
    sizes_coarse = np.array([40, 30, 30])
    sizes_fine = np.array([10] * 10)
    n_total = 100
    assert mgc.expected_n_groups(5, sizes_coarse, n_total) < mgc.expected_n_groups(
        5, sizes_fine, n_total
    )


def test_concentration_cli_runs_at_both_levels(tmp_path):
    exps = real_experiments(40)
    rows = [("pos", exps, "SP1", 40 * 900) for _ in range(4)]
    rows += [("pos", exps[:6], "GATA1", 6 * 400) for _ in range(4)]
    rows += [("pos", exps[i : i + 3], None, 3 * 60) for i in range(0, 24, 3)]
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)

    for level in ("tissue", "biosample"):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
                "--cluster-metadata", str(meta), "--group-level", level,
                "--out-dir", str(tmp_path / "out"),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        out = tmp_path / "out"
        per_cluster = pd.read_csv(
            out / f"motif_concentration_profile_{level}.tsv", sep="\t"
        )
        assert {"n_groups", "expected_n_groups", "concentration"} <= set(
            per_cluster.columns
        )
        assert (per_cluster["expected_n_groups"] > 0).all()
        summary = pd.read_csv(
            out / f"motif_concentration_profile_{level}_summary.tsv", sep="\t"
        )
        assert "single_group_p" in summary.columns
        assert (out / f"motif_concentration_profile_{level}.png").exists()


def test_concentration_cli_rejects_mapping_input(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
            "--cluster-metadata", str(tmp_path / "nope.tsv"),
            "--out-dir", str(tmp_path / "out"),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "pattern-to-cluster mapping is not sufficient" in result.stderr


def test_jaspar_score_threshold_reclassifies(tmp_path):
    exps = real_experiments(20)
    rows = [("pos", exps, "WEAK", 20 * 500) for _ in range(6)]
    rows += [("pos", exps[:5], "STRONG", 5 * 500) for _ in range(6)]
    meta_path = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)
    m = pd.read_csv(meta_path, sep="\t")
    m.loc[m.jaspar_name == "WEAK", "jaspar_score"] = 0.80
    m.loc[m.jaspar_name == "STRONG", "jaspar_score"] = 0.95
    m.to_csv(meta_path, sep="\t", index=False)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
            "--cluster-metadata", str(meta_path), "--out-dir", str(tmp_path / "out"),
            "--jaspar-score-threshold", "0.85",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "moved 6 clusters into the unmatched class" in result.stderr


def test_prevalence_one_clusters_are_uninformative_for_concentration():
    """At p=1 the statistics are degenerate: n_groups is always 1, and both
    E[n_groups] and P[n_groups=1] are exactly 1. Singletons therefore cannot
    provide evidence either way, which is why the script warns about them."""
    sizes = np.array([41, 28, 21, 18, 15, 11, 8, 8, 8, 7, 7, 6, 5, 4, 3, 3, 2, 2, 1])
    n_total = int(sizes.sum())
    assert mgc.expected_n_groups(1, sizes, n_total) == pytest.approx(1.0)
    assert mgc.prob_single_group(1, sizes, n_total) == pytest.approx(1.0)


def test_singletons_dilute_enrichment_but_not_the_pvalue():
    """Deterministic (q=1) terms shift observed and expected equally and add no
    variance, so the tail probability is unchanged while the effect-size ratio
    collapses -- the reason the warning targets the ratio specifically."""
    sizes = np.array([41, 28, 21, 18, 15, 11, 8, 8, 8, 7, 7, 6, 5, 4, 3, 3, 2, 2, 1])
    n_total = int(sizes.sum())
    q = mgc.prob_single_group(2, sizes, n_total)
    base = np.full(22, q)
    with_singletons = np.concatenate([base, np.ones(235)])

    p_base = mgc.poisson_binomial_sf(base, 9)
    p_diluted = mgc.poisson_binomial_sf(with_singletons, 9 + 235)
    assert p_diluted == pytest.approx(p_base, rel=1e-6)

    enrich_base = 9 / base.sum()
    enrich_diluted = (9 + 235) / with_singletons.sum()
    assert enrich_base > 4.0
    assert enrich_diluted < 1.1


def test_concentration_cli_warns_on_min_cluster_experiments_one(tmp_path):
    exps = real_experiments(20)
    rows = [("pos", exps[:4], "TF", 4 * 100) for _ in range(6)]
    rows += [("pos", [exps[i]], None, 100) for i in range(6)]
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
            "--min-cluster-experiments", "1",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "carry no information for this test" in result.stderr
