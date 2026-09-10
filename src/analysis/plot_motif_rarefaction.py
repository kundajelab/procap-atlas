#!/usr/bin/env python3
"""Motif-lexicon rarefaction: how many consensus motifs does the atlas buy?

Manuscript panel (Fig. 2a inset). Answers "why 200+ experiments rather than a
few cell lines?" by measuring how the size of the deduplicated MotifCompendium
lexicon grows as experiments are added, and -- more to the point -- showing that
what drives that growth is *biosample diversity*, not experiment count.

Input is src/bpnet/motifcompendium/cluster_motifs.py's
`motifcompendium_{head}_cluster_metadata.tsv`, whose `experiments` column is a
comma-joined list of every experiment contributing a motif to that cluster.
That is exactly a cluster x experiment presence/absence matrix, so no hit
calling or attribution data is needed here.

Three sampling schemes are compared at each subset size k:

  uniform    k experiments drawn uniformly at random.
  diverse    k experiments drawn round-robin across biosample groups, so each
             new experiment comes from the least-represented group available.
  redundant  k experiments drawn by filling one biosample group completely
             before starting the next.

The `uniform` curve's mean is computed in closed form rather than by sampling:
for a cluster present in p of N experiments, the chance a random size-k subset
contains at least one of them is 1 - C(N-p, k)/C(N, k), so the expected lexicon
size is the sum of that over clusters.

That closed form is also why there is deliberately no permutation null here. A
tempting one -- keep each cluster's prevalence p but randomize *which*
experiments it appears in -- is provably vacuous: the expectation above depends
only on p, never on which experiments, so a prevalence-matched null reproduces
the observed `uniform` curve exactly. Structure in the matrix is only visible to
a sampling scheme that is itself structured, which is what `diverse` and
`redundant` are for. Do not re-add a uniform-subsampling null.

Read depth is the main confound: motif discovery power scales with library size
[Fig. 1d], so a curve computed over all experiments partly measures depth
rather than biology. Use --min-reads (matching cluster_motifs.py's own default
of 10M) to hold that roughly fixed.

Outputs (in --out-dir):
  motif_rarefaction_{head}.tsv        long-form curve table (k, scheme, motif_class, mean, lo, hi)
  motif_rarefaction_{head}.{png,pdf}  the panel
  motif_prevalence_{head}.tsv         per-cluster prevalence and group breadth

Usage:
    python src/analysis/plot_motif_rarefaction.py
    python src/analysis/plot_motif_rarefaction.py --head count --min-reads 10000000
    python src/analysis/plot_motif_rarefaction.py --min-cluster-experiments 2
    python src/analysis/plot_motif_rarefaction.py --biosample-groups configs/biosample_groups.tsv
"""

import argparse
import sys
from math import exp, lgamma
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _biosample_groups import load_group_map, write_group_tsv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet"

# Abundance thresholds for the sensitivity sweep, in seqlets per contributing
# motif. Deliberately spans two orders of magnitude rather than centring on a
# chosen value: the whole point of the sweep is that no single threshold is
# defensible (see sweep_abundance).
DEFAULT_SWEEP_THRESHOLDS = (0.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
DEFAULT_SWEEP_MARKS = (1, 5, 10, 25, 50, 100)

SCHEME_STYLE = {
    "diverse": ("#1b7837", "-", "diverse (round-robin across tissues)"),
    "uniform": ("#404040", "-", "uniform random"),
    "redundant": ("#b2182b", "-", "redundant (one tissue at a time)"),
}


def load_read_counts(path: Path = N_READS_PATH) -> dict[str, float]:
    """Parse configs/n_reads.txt (experiment, biosample, pl, mn, total)."""
    counts: dict[str, float] = {}
    if not path.exists():
        return counts
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                counts[parts[0]] = float(parts[4])
    return counts


def load_presence(
    metadata_path: Path, keep_experiments: set[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Build a cluster x experiment presence table from cluster metadata.

    Returns (clusters, experiments) where `clusters` has one row per cluster
    with an `exp_set` column of contributing experiment accessions (restricted
    to `keep_experiments` if given) and `experiments` is the sorted universe of
    experiments actually retained.
    """
    meta = pd.read_csv(metadata_path, sep="\t")
    for required in ("cluster_final", "experiments"):
        if required not in meta.columns:
            raise ValueError(
                f"{metadata_path} is missing required column {required!r}; "
                "regenerate it with src/bpnet/motifcompendium/cluster_motifs.py"
            )

    def parse(cell: object) -> set[str]:
        if not isinstance(cell, str):
            return set()
        exps = {e.strip() for e in cell.split(",") if e.strip()}
        return exps & keep_experiments if keep_experiments is not None else exps

    meta["exp_set"] = meta["experiments"].map(parse)
    meta["prevalence"] = meta["exp_set"].map(len)
    if {"total_seqlets", "n_motifs"} <= set(meta.columns):
        # Prevalence-normalized abundance: a cluster's total_seqlets is summed
        # over its contributing motifs, and n_motifs tracks prevalence closely
        # (within-model clustering collapses each experiment to ~one motif per
        # cluster), so total_seqlets alone is essentially a prevalence proxy.
        #
        # Note this ratio is computed over *all* motifs the compendium assigned
        # to the cluster, including any from experiments dropped by
        # --min-reads: cluster_metadata.tsv carries only the aggregates, not
        # per-motif seqlet counts, so it cannot be recomputed over the
        # retained subset. It is an abundance proxy, not an exact count.
        meta["seqlets_per_motif"] = meta["total_seqlets"] / meta["n_motifs"].replace(
            0, np.nan
        )
    # A cluster whose every contributing experiment was filtered out is not
    # discoverable within this universe at any k, so it cannot be counted.
    meta = meta[meta["prevalence"] > 0].reset_index(drop=True)
    universe = sorted(set().union(*meta["exp_set"])) if len(meta) else []
    return meta, universe


def load_presence_from_mapping(
    mapping_path: Path, keep_experiments: set[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Build the same cluster x experiment presence table from the
    pattern-to-cluster mapping instead of the cluster metadata.

    cluster_motifs.py writes `motifcompendium_{head}_pattern_to_cluster.tsv`
    immediately after clustering, but `..._cluster_metadata.tsv` only much
    later -- after the cluster-average h5 export, JASPAR annotation of the
    averages, forward/reverse logo generation, MEME export and per-cluster SVG
    logo rendering (it merges the logo paths in, which is the only reason it
    waits on them). Those stages can take hours on a full atlas run, and this
    panel needs none of them: grouping the mapping's `experiment` column by
    `compendium_motif_name` recovers exactly the same presence sets.

    What is lost is only the metadata columns -- `total_seqlets`, `n_motifs`
    and `jaspar_name` -- so stratified curves fall back to a single class
    unless --annotation-tsv is supplied.
    """
    mapping = pd.read_csv(mapping_path, sep="\t")
    required = {"experiment", "compendium_motif_name"}
    if not required <= set(mapping.columns):
        raise ValueError(
            f"{mapping_path} is missing {sorted(required - set(mapping.columns))}; "
            "regenerate it with src/bpnet/motifcompendium/cluster_motifs.py"
        )
    if keep_experiments is not None:
        mapping = mapping[mapping["experiment"].isin(keep_experiments)]

    grouped = (
        mapping.groupby("compendium_motif_name")["experiment"]
        .apply(lambda s: set(s))
        .reset_index(name="exp_set")
    )
    # compendium_motif_name is f"{posneg}_patterns.{cluster_final}"
    split = grouped["compendium_motif_name"].str.rsplit(".", n=1, expand=True)
    grouped["posneg"] = split[0].str.replace("_patterns", "", regex=False)
    grouped["cluster_final"] = pd.to_numeric(split[1], errors="coerce").astype("Int64")
    grouped["prevalence"] = grouped["exp_set"].map(len)
    grouped = grouped[grouped["prevalence"] > 0]
    # groupby orders by the motif-name string, which puts every neg_patterns
    # cluster ahead of every pos_patterns one; sort by cluster id instead so
    # the prevalence TSV is deterministic and diffable. Row order does not
    # affect any curve (they are all order-invariant sums over clusters).
    grouped = grouped.sort_values(["posneg", "cluster_final"]).reset_index(drop=True)
    universe = sorted(set().union(*grouped["exp_set"])) if len(grouped) else []
    return grouped, universe


def uniform_curve_exact(prevalences: np.ndarray, n_total: int) -> np.ndarray:
    """Expected lexicon size at every k = 1..n_total under uniform sampling.

    E[detected at k] = sum_c (1 - C(N - p_c, k) / C(N, k)), evaluated in log
    space so that N ~ 200 stays numerically comfortable.
    """
    ks = np.arange(1, n_total + 1)
    out = np.zeros(n_total, dtype=float)

    def log_comb(n: int, k: int) -> float:
        return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

    for p in prevalences:
        p = int(p)
        miss = np.zeros(n_total, dtype=float)
        for i, k in enumerate(ks):
            n_miss = n_total - p
            miss[i] = 0.0 if k > n_miss else exp(log_comb(n_miss, k) - log_comb(n_total, k))
        out += 1.0 - miss
    return out


def sweep_abundance(
    meta: pd.DataFrame,
    n_total: int,
    thresholds: tuple[float, ...] | list[float],
    marks: tuple[int, ...] | list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rarefaction under a range of abundance floors on seqlets_per_motif.

    Reports the sensitivity of the panel's headline number to a filtering
    choice that has no defensible single value. `seqlets_per_motif` is only
    partly decoupled from prevalence (measured r ~ 0.4 on the real count-head
    compendium), so raising the floor preferentially removes low-prevalence,
    tissue-restricted clusters. The surviving lexicon is therefore more
    ubiquitous, and a small experiment sample recovers a *larger* fraction of
    it -- the fraction-recovered metric rises with the threshold as an
    artifact of the confound, not as evidence about noise.

    Because the denominator moves with the threshold, fractions are not
    comparable across thresholds and no single row of this table is "the"
    answer. What the sweep supports is the weakest-form claim: the largest
    fraction recovered at a given k, over every threshold tested, is an upper
    bound on what a k-experiment study can see.

    Uses the closed-form uniform expectation only. The sweep is about
    abundance sensitivity, not sampling scheme, and the structured schemes
    would add Monte-Carlo noise to a comparison that is exact without it.

    Returns (curves, summary): `curves` is long-form (threshold, k, mean,
    fraction, n_clusters); `summary` is one row per threshold with the
    fraction recovered at each k in `marks`.
    """
    if "seqlets_per_motif" not in meta.columns:
        raise ValueError(
            "abundance sweep needs total_seqlets and n_motifs, which only "
            "cluster_metadata.tsv carries -- not the pattern-to-cluster "
            "mapping. Rerun without --pattern-to-cluster once "
            "cluster_metadata.tsv exists."
        )

    curve_rows, summary_rows = [], []
    ks = np.arange(1, n_total + 1)
    for threshold in thresholds:
        kept = meta[meta["seqlets_per_motif"] >= threshold]
        n_clusters = len(kept)
        if n_clusters < 10:
            print(
                f"  skipping threshold {threshold:g}: only {n_clusters} clusters left",
                file=sys.stderr,
            )
            continue
        curve = uniform_curve_exact(kept["prevalence"].to_numpy(), n_total)
        curve_rows.append(
            pd.DataFrame(
                {
                    "min_seqlets_per_motif": threshold,
                    "k": ks,
                    "mean": curve,
                    "fraction": curve / n_clusters,
                    "n_clusters": n_clusters,
                }
            )
        )
        row = {"min_seqlets_per_motif": threshold, "n_clusters": n_clusters}
        for k in marks:
            if k <= n_total:
                row[f"fraction_at_k{k}"] = curve[k - 1] / n_clusters
                row[f"mean_at_k{k}"] = curve[k - 1]
        summary_rows.append(row)

    if not curve_rows:
        raise ValueError("no abundance threshold retained enough clusters to sweep")
    return pd.concat(curve_rows, ignore_index=True), pd.DataFrame(summary_rows)


def plot_sweep(
    curves: pd.DataFrame, head: str, out_stem: Path, mark: int, subtitle: str
) -> None:
    """Two-panel sweep figure: absolute lexicon size, and fraction recovered.

    Both are shown because neither alone is honest. Absolute counts keep a
    fixed meaning across thresholds but each curve ends at a different total;
    fractions are directly readable as "what a k-experiment study sees" but
    have a denominator that moves with the threshold.
    """
    thresholds = sorted(curves["min_seqlets_per_motif"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {t: cmap(i / max(1, len(thresholds) - 1)) for i, t in enumerate(thresholds)}

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for threshold in thresholds:
        sub = curves[curves["min_seqlets_per_motif"] == threshold].sort_values("k")
        n = int(sub["n_clusters"].iloc[0])
        label = f"\u2265{threshold:g} (n={n})"
        axes[0].plot(sub["k"], sub["mean"], color=colors[threshold], lw=1.5, label=label)
        axes[1].plot(sub["k"], sub["fraction"], color=colors[threshold], lw=1.5, label=label)

    axes[0].set_ylabel("Consensus motifs recovered")
    axes[1].set_ylabel("Fraction of that threshold's lexicon")
    for ax in axes:
        ax.set_xlabel("Experiments included")
        ax.spines[["top", "right"]].set_visible(False)

    at_mark = curves[curves["k"] == mark]
    if not at_mark.empty:
        worst = at_mark["fraction"].max()
        axes[1].axvline(mark, color="0.6", ls=":", lw=1)
        axes[1].axhline(worst, color="#b2182b", ls="--", lw=1)
        # Below the dashed line rather than above it: every curve sits above
        # the bound past small k, so the space above is occupied.
        axes[1].annotate(
            f"k={mark}: \u2264{worst:.0%} at every threshold",
            xy=(0.30 * float(curves["k"].max()), max(0.04, worst - 0.14)),
            fontsize=7, color="#b2182b",
        )
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(
        title="min seqlets/motif", frameon=False, fontsize=6, title_fontsize=7,
        loc="lower right",
    )
    fig.suptitle(
        f"Abundance-threshold sensitivity — {head} head\n{subtitle}", fontsize=9
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}", file=sys.stderr)
    plt.close(fig)


def _order_diverse(groups: dict[str, str], rng: np.random.Generator) -> list[str]:
    """Round-robin over biosample groups, shuffling within and between rounds."""
    by_group: dict[str, list[str]] = {}
    for exp_id, group in groups.items():
        by_group.setdefault(group, []).append(exp_id)
    for members in by_group.values():
        rng.shuffle(members)
    order: list[str] = []
    while any(by_group.values()):
        live = [g for g, m in by_group.items() if m]
        rng.shuffle(live)
        for g in live:
            order.append(by_group[g].pop())
    return order


def _order_redundant(groups: dict[str, str], rng: np.random.Generator) -> list[str]:
    """Exhaust one biosample group before moving to the next."""
    by_group: dict[str, list[str]] = {}
    for exp_id, group in groups.items():
        by_group.setdefault(group, []).append(exp_id)
    group_order = list(by_group)
    rng.shuffle(group_order)
    order: list[str] = []
    for g in group_order:
        members = by_group[g]
        rng.shuffle(members)
        order.extend(members)
    return order


def _order_uniform(groups: dict[str, str], rng: np.random.Generator) -> list[str]:
    order = list(groups)
    rng.shuffle(order)
    return order


ORDERERS = {
    "uniform": _order_uniform,
    "diverse": _order_diverse,
    "redundant": _order_redundant,
}


def rarefy(
    exp_sets: list[set[str]],
    groups: dict[str, str],
    scheme: str,
    n_reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Monte-Carlo rarefaction curves, shape (n_reps, n_experiments).

    Each replicate draws one full ordering of experiments under `scheme` and
    walks it, so curve[r, k-1] is the lexicon size after the first k
    experiments of that ordering -- curves are monotone within a replicate,
    which they would not be if each k were sampled independently.
    """
    n_total = len(groups)
    curves = np.zeros((n_reps, n_total), dtype=int)
    orderer = ORDERERS[scheme]

    # experiment -> clusters it contributes to, built once and reused across
    # replicates (the matrix never changes, only the visiting order does).
    index: dict[str, list[int]] = {}
    for ci, exps in enumerate(exp_sets):
        for e in exps:
            index.setdefault(e, []).append(ci)

    for r in range(n_reps):
        order = orderer(groups, rng)
        # A cluster is credited the first time any of its experiments is
        # visited, so each cluster is counted exactly once per replicate.
        seen = [False] * len(exp_sets)
        detected = 0
        for k, exp_id in enumerate(order):
            for ci in index.get(exp_id, ()):
                if not seen[ci]:
                    seen[ci] = True
                    detected += 1
            curves[r, k] = detected
    return curves


def classify_clusters(meta: pd.DataFrame, annotation_tsv: Path | None) -> pd.Series:
    """Assign each cluster a class label for stratified curves.

    Prefers a curated `cluster_final<TAB>class` table. Falls back to a JASPAR
    proxy: clusters with a JASPAR match are labelled "TF-matched" and those
    without "unmatched (core promoter / repeat)". The fallback is a proxy, not
    an annotation -- JASPAR2026 has essentially no coverage of core promoter
    elements (Inr, TATA, DPE), which is why cluster_motifs.py's own reports
    annotate them by hand.
    """
    if annotation_tsv is not None:
        ann = pd.read_csv(annotation_tsv, sep="\t", comment="#")
        class_col = next(
            (c for c in ("class", "motif_class") if c in ann.columns), None
        )
        if "cluster_final" not in ann.columns or class_col is None:
            raise ValueError(
                f"{annotation_tsv} must have a cluster_final column and a "
                "class (or motif_class) column"
            )
        mapping = dict(zip(ann["cluster_final"], ann[class_col]))
        return meta["cluster_final"].map(mapping).fillna("unannotated")
    if "jaspar_name" in meta.columns and meta["jaspar_name"].notna().any():
        matched = meta["jaspar_name"].notna() & (meta["jaspar_name"].astype(str) != "")
        return pd.Series(
            np.where(matched, "TF-matched", "unmatched (core promoter / repeat)"),
            index=meta.index,
        )
    return pd.Series("all motifs", index=meta.index)


def plot_curves(curves: pd.DataFrame, head: str, out_stem: Path, subtitle: str) -> None:
    """Draw the rarefaction panel: one line per (scheme, class)."""
    classes = [c for c in curves["motif_class"].unique() if c != "__all__"]
    n_panels = 1 + (len(classes) if len(classes) > 1 else 0)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.0 * n_panels, 3.4), sharex=True, squeeze=False
    )
    axes = axes[0]

    panels = [("all motifs", "__all__")] + (
        [(c, c) for c in classes] if len(classes) > 1 else []
    )
    for ax, (title, cls) in zip(axes, panels):
        sub = curves[curves["motif_class"] == cls]
        for scheme, (color, ls, label) in SCHEME_STYLE.items():
            s = sub[sub["scheme"] == scheme].sort_values("k")
            if s.empty:
                continue
            ax.plot(s["k"], s["mean"], color=color, ls=ls, lw=1.6, label=label)
            if s["lo"].notna().any():
                ax.fill_between(s["k"], s["lo"], s["hi"], color=color, alpha=0.15, lw=0)
        ax.set_xlabel("Experiments included")
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Consensus motifs recovered")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(
        f"Motif lexicon rarefaction — {head} head\n{subtitle}", fontsize=9
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}", file=sys.stderr)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--head", default="profile", choices=["profile", "count"],
        help="attribution head whose compendium to rarefy (default: profile)",
    )
    parser.add_argument(
        "--cluster-metadata", type=Path, default=None, metavar="PATH",
        help="override motifcompendium/bpnet/motifcompendium_{head}_cluster_metadata.tsv",
    )
    parser.add_argument(
        "--pattern-to-cluster", type=Path, default=None, metavar="PATH",
        help="use motifcompendium_{head}_pattern_to_cluster.tsv instead of the "
             "cluster metadata. cluster_motifs.py writes it hours earlier (right "
             "after clustering, before the logo/h5/report exports), and it "
             "carries the same presence information -- only the seqlet counts "
             "and JASPAR labels are lost",
    )
    parser.add_argument(
        "--min-reads", type=float, default=10_000_000, metavar="N",
        help="drop experiments below N total reads, holding discovery power "
             "roughly fixed (default: 10000000, matching cluster_motifs.py)",
    )
    parser.add_argument(
        "--min-cluster-experiments", type=int, default=1, metavar="N",
        help="only count clusters seen in >= N experiments; 2 excludes "
             "singletons, the least reproducible clusters (default: 1)",
    )
    parser.add_argument(
        "--min-seqlets-per-motif", type=float, default=0.0, metavar="N",
        help="drop clusters below N seqlets per contributing motif "
             "(total_seqlets / n_motifs). Deliberately defaults to 0: the "
             "sweep below shows the recovered fraction rises monotonically "
             "with this floor as an artifact of its residual correlation with "
             "prevalence, so no single value is defensible. Requires "
             "cluster_metadata.tsv, not the pattern-to-cluster mapping",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="also emit the abundance-threshold sensitivity panel",
    )
    parser.add_argument(
        "--sweep-thresholds", type=float, nargs="+", default=None, metavar="N",
        help=f"seqlets-per-motif floors to sweep (default: "
             f"{' '.join(f'{t:g}' for t in DEFAULT_SWEEP_THRESHOLDS)}); "
             "implies --sweep",
    )
    parser.add_argument(
        "--sweep-marks", type=int, nargs="+", default=list(DEFAULT_SWEEP_MARKS),
        metavar="K", help="experiment counts to tabulate in the sweep summary "
             f"(default: {' '.join(str(k) for k in DEFAULT_SWEEP_MARKS)})",
    )
    parser.add_argument(
        "--sweep-mark", type=int, default=5, metavar="K",
        help="experiment count annotated on the sweep panel; 5 is the scale of "
             "a typical single-lab cell-line panel (default: 5)",
    )
    parser.add_argument(
        "--annotation-tsv", type=Path, default=None, metavar="PATH",
        help="curated cluster_final<TAB>class table for stratified curves "
             "(default: fall back to a JASPAR-match proxy)",
    )
    parser.add_argument(
        "--biosample-groups", type=Path, default=None, metavar="PATH",
        help="curated biosample<TAB>group table (see --write-group-tsv)",
    )
    parser.add_argument(
        "--write-group-tsv", type=Path, default=None, metavar="PATH",
        help="write the resolved biosample->group table here and exit, for curation",
    )
    parser.add_argument(
        "--n-reps", type=int, default=200, metavar="N",
        help="Monte-Carlo replicates per structured scheme (default: 200)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figures" / "motif_atlas",
        metavar="DIR", help="output directory (default: figures/motif_atlas/)",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        experiments_cfg = yaml.safe_load(f)["experiments"]

    group_map, biosample_map = load_group_map(experiments_cfg, args.biosample_groups)
    if args.write_group_tsv:
        write_group_tsv(biosample_map, group_map, args.write_group_tsv)
        return

    read_counts = load_read_counts()
    if args.min_reads > 0 and not read_counts:
        print(
            f"ERROR: --min-reads {args.min_reads:,.0f} requested but "
            f"{N_READS_PATH} is missing; pass --min-reads 0 to skip depth filtering",
            file=sys.stderr,
        )
        sys.exit(1)
    keep = {e for e in experiments_cfg if read_counts.get(e, 0) >= args.min_reads}

    default_metadata = MC_DIR / f"motifcompendium_{args.head}_cluster_metadata.tsv"
    default_mapping = MC_DIR / f"motifcompendium_{args.head}_pattern_to_cluster.tsv"

    if args.pattern_to_cluster is not None:
        source, from_mapping = args.pattern_to_cluster, True
    elif args.cluster_metadata is not None:
        source, from_mapping = args.cluster_metadata, False
    elif default_metadata.exists():
        source, from_mapping = default_metadata, False
    elif default_mapping.exists():
        # cluster_motifs.py is probably still running its export/report
        # stages; the mapping is already final and sufficient for this panel.
        source, from_mapping = default_mapping, True
        print(
            f"NOTE: {default_metadata.name} not present yet; falling back to "
            f"{default_mapping.name}, which carries the same presence "
            "information (seqlet counts and JASPAR labels unavailable).",
            file=sys.stderr,
        )
    else:
        print(f"ERROR: no compendium input found in {MC_DIR}", file=sys.stderr)
        print(
            "Expected either "
            f"{default_metadata.name} or {default_mapping.name}. Run "
            f"src/bpnet/motifcompendium/cluster_motifs.py --head {args.head} first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not source.exists():
        print(f"ERROR: input not found: {source}", file=sys.stderr)
        sys.exit(1)

    if from_mapping:
        meta, universe = load_presence_from_mapping(source, keep_experiments=keep)
    else:
        meta, universe = load_presence(source, keep_experiments=keep)
    if args.min_cluster_experiments > 1:
        meta = meta[meta["prevalence"] >= args.min_cluster_experiments].reset_index(
            drop=True
        )
    if args.min_seqlets_per_motif > 0:
        if "seqlets_per_motif" not in meta.columns:
            print(
                "ERROR: --min-seqlets-per-motif needs total_seqlets and "
                "n_motifs, which only cluster_metadata.tsv carries",
                file=sys.stderr,
            )
            sys.exit(1)
        before = len(meta)
        meta = meta[
            meta["seqlets_per_motif"] >= args.min_seqlets_per_motif
        ].reset_index(drop=True)
        print(
            f"{args.head}: abundance floor {args.min_seqlets_per_motif:g} "
            f"seqlets/motif kept {len(meta)}/{before} clusters",
            file=sys.stderr,
        )
    if meta.empty or not universe:
        print("ERROR: no clusters survived filtering", file=sys.stderr)
        sys.exit(1)

    if "seqlets_per_motif" in meta.columns and len(meta) > 2:
        # The diagnostic that justifies not choosing a threshold: if abundance
        # correlates with prevalence, any abundance floor is also a prevalence
        # floor, and so biases against tissue-restricted clusters.
        r = meta[["prevalence", "seqlets_per_motif"]].corr().iloc[0, 1]
        print(
            f"{args.head}: prevalence vs seqlets/motif r = {r:.2f} "
            "(an abundance floor is partly a prevalence floor)",
            file=sys.stderr,
        )

    meta["motif_class"] = classify_clusters(meta, args.annotation_tsv)
    meta["n_groups"] = meta["exp_set"].map(lambda s: len({group_map[e] for e in s}))
    groups = {e: group_map[e] for e in universe}
    n_total = len(universe)
    print(
        f"{args.head}: {len(meta):,} clusters over {n_total} experiments "
        f"in {len(set(groups.values()))} biosample groups",
        file=sys.stderr,
    )

    rng = np.random.default_rng(args.seed)
    rows = []
    ks = np.arange(1, n_total + 1)
    for cls, sub in [("__all__", meta)] + list(meta.groupby("motif_class")):
        exp_sets = list(sub["exp_set"])
        exact = uniform_curve_exact(sub["prevalence"].to_numpy(), n_total)
        for scheme in ("uniform", "diverse", "redundant"):
            curves = rarefy(exp_sets, groups, scheme, args.n_reps, rng)
            mean = exact if scheme == "uniform" else curves.mean(axis=0)
            lo, hi = np.percentile(curves, [5, 95], axis=0)
            rows.append(
                pd.DataFrame(
                    {
                        "k": ks, "scheme": scheme, "motif_class": cls,
                        "mean": mean, "lo": lo, "hi": hi,
                        "mc_mean": curves.mean(axis=0),
                        "n_clusters_total": len(sub),
                    }
                )
            )
    curve_df = pd.concat(rows, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"motif_rarefaction_{args.head}"
    curve_df.to_csv(stem.with_suffix(".tsv"), sep="\t", index=False)
    print(f"Saved {stem.with_suffix('.tsv')}", file=sys.stderr)

    prevalence_path = args.out_dir / f"motif_prevalence_{args.head}.tsv"
    cols = [c for c in ("cluster_final", "posneg", "jaspar_name", "jaspar_score",
                        "n_motifs", "total_seqlets") if c in meta.columns]
    meta[cols + ["prevalence", "n_groups", "motif_class"]].to_csv(
        prevalence_path, sep="\t", index=False
    )
    print(f"Saved {prevalence_path}", file=sys.stderr)

    subtitle = (
        f"n={n_total} experiments (>={args.min_reads/1e6:.0f}M reads), "
        f"{len(meta):,} clusters"
    )
    plot_curves(curve_df, args.head, stem, subtitle)

    if args.sweep or args.sweep_thresholds is not None:
        thresholds = (
            args.sweep_thresholds
            if args.sweep_thresholds is not None
            else list(DEFAULT_SWEEP_THRESHOLDS)
        )
        try:
            sweep_curves, sweep_summary = sweep_abundance(
                meta, n_total, thresholds, args.sweep_marks
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        sweep_stem = args.out_dir / f"motif_rarefaction_sweep_{args.head}"
        sweep_curves.to_csv(sweep_stem.with_suffix(".tsv"), sep="\t", index=False)
        summary_path = args.out_dir / f"motif_rarefaction_sweep_{args.head}_summary.tsv"
        sweep_summary.to_csv(summary_path, sep="\t", index=False)
        print(f"Saved {sweep_stem.with_suffix('.tsv')}", file=sys.stderr)
        print(f"Saved {summary_path}", file=sys.stderr)
        plot_sweep(sweep_curves, args.head, sweep_stem, args.sweep_mark, subtitle)

        print("\nAbundance-threshold sensitivity:", file=sys.stderr)
        for _, row in sweep_summary.iterrows():
            marks = "  ".join(
                f"k={k}: {row[f'fraction_at_k{k}']:5.1%}"
                for k in args.sweep_marks
                if f"fraction_at_k{k}" in row
            )
            print(
                f"  >={row['min_seqlets_per_motif']:>6.0f} seqlets/motif "
                f"(n={int(row['n_clusters']):>4})  {marks}",
                file=sys.stderr,
            )
        mark_col = f"fraction_at_k{args.sweep_mark}"
        if mark_col in sweep_summary.columns:
            worst = sweep_summary[mark_col].max()
            argworst = sweep_summary.loc[sweep_summary[mark_col].idxmax()]
            print(
                f"\nWeakest-form claim: a {args.sweep_mark}-experiment study "
                f"recovers at most {worst:.0%} of the {args.head}-head lexicon "
                f"at any threshold tested\n  (upper bound set by "
                f">={argworst['min_seqlets_per_motif']:g} seqlets/motif, "
                f"n={int(argworst['n_clusters'])} clusters), so at least "
                f"{1 - worst:.0%} is missed regardless.",
                file=sys.stderr,
            )

    overall = curve_df[
        (curve_df["scheme"] == "uniform") & (curve_df["motif_class"] == "__all__")
    ]
    at = {int(k): float(v) for k, v in zip(overall["k"], overall["mean"])}
    marks = [k for k in (1, 5, 10, 25, 50, 100, n_total) if k in at]
    print("Uniform-sampling lexicon size:", file=sys.stderr)
    for k in marks:
        print(f"  k={k:>4}: {at[k]:8.1f} clusters ({at[k]/at[n_total]:5.1%} of full)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
