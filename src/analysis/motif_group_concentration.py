#!/usr/bin/env python3
"""Are motif clusters concentrated in particular tissues, or spread at random?

Tests whether the experiments a MotifCompendium cluster was discovered in are
drawn from fewer biosample groups than chance would give. This is the
discriminating test that abundance thresholds cannot provide: a real
lineage-restricted motif found in 8 liver experiments has n_groups = 1, while a
cluster spread over 8 arbitrary experiments has n_groups near the random
expectation, and both can have identical seqlet counts.

Two statistics, both exact rather than sampled. For a cluster of prevalence p
over N experiments partitioned into groups of size n_g:

  E[n_groups | p] = sum_g [1 - C(N - n_g, p) / C(N, p)]
  P[n_groups = 1 | p] = sum_g C(n_g, p) / C(N, p)

`concentration` is n_groups / E[n_groups], so ~1 means indistinguishable from a
random draw of experiments and <<1 means tissue-concentrated. The
single-group count is tested against its exact Poisson-binomial distribution
over the clusters in each row of the report, which is the sharper test at low
prevalence -- where `concentration` has almost no dynamic range (at p=2 the
ratio can only take two values).

--group-level controls what "group" means, and it changes the question:

  tissue     the keyword grouping in _biosample_groups.py. Asks whether a motif
             is lineage-restricted. Note this grouping is coarse -- on the real
             atlas three groups hold 45% of experiments, and blood_immune alone
             spans erythroid, T/NK, B, myeloid and lymphoid-tissue biosamples --
             so it has limited power to see lineage restriction *within* those
             groups.
  biosample  the raw ENCODE biosample string. Asks whether a motif's discovery
             is replicate-driven (HCT116 x16, brain metastases x10, PBMC x8,
             K562 x7). Assumption-free, no curation, and higher resolution, but
             most biosamples appear once, so P[n_groups = 1] is near zero for
             p > 1 and the single-group test loses meaning.

Run both. A conclusion that holds at both levels is not a grouping artifact.

Rows of the report are split by --split-by, and the default (motif_class) is
the split that matters: pooling all clusters together dilutes the signal, since
on real count-head data the JASPAR-matched majority is spread near-randomly
while the unmatched minority is not.

Outputs (in --out-dir):
  motif_concentration_{head}_{level}.tsv          per-cluster observed/expected/ratio
  motif_concentration_{head}_{level}_summary.tsv  per-split aggregate and enrichment test
  motif_concentration_{head}_{level}.{png,pdf}    n_groups vs prevalence against expectation

Usage:
    python src/analysis/motif_group_concentration.py --head count
    python src/analysis/motif_group_concentration.py --head count --group-level biosample
    python src/analysis/motif_group_concentration.py --head count --min-cluster-experiments 3
    python src/analysis/motif_group_concentration.py --head count --jaspar-score-threshold 0.85
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
from plot_motif_rarefaction import (  # noqa: E402
    MC_DIR,
    classify_clusters,
    load_presence,
    load_read_counts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
ABUNDANCE_BINS = (0, 25, 50, 100, 500, np.inf)
ABUNDANCE_LABELS = ("<25", "25-50", "50-100", "100-500", ">=500")


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def expected_n_groups(p: int, group_sizes: np.ndarray, n_total: int) -> float:
    """E[number of groups represented] for a uniform random p-subset."""
    total = 0.0
    for n_g in group_sizes:
        miss = _log_comb(n_total - int(n_g), p) - _log_comb(n_total, p)
        total += 1.0 - (0.0 if miss == -np.inf else exp(miss))
    return total


def prob_single_group(p: int, group_sizes: np.ndarray, n_total: int) -> float:
    """P[all p experiments fall in one group] for a uniform random p-subset.

    Groups are disjoint, so the events "all in group g" are mutually exclusive
    and the sum is exact rather than a union bound.
    """
    denom = _log_comb(n_total, p)
    total = 0.0
    for n_g in group_sizes:
        num = _log_comb(int(n_g), p)
        if num != -np.inf:
            total += exp(num - denom)
    return total


def poisson_binomial_sf(probs: np.ndarray, observed: int) -> float:
    """P[X >= observed] for X = sum of independent Bernoulli(probs).

    Exact by dynamic programming rather than a Poisson or normal approximation:
    the per-cluster probabilities here are small and very unequal (they depend
    on each cluster's own prevalence), which is exactly the regime where those
    approximations are least trustworthy.
    """
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for i, q in enumerate(probs):
        dist[1 : i + 2] = dist[1 : i + 2] * (1 - q) + dist[0 : i + 1] * q
        dist[0] *= 1 - q
    if observed <= 0:
        return 1.0
    return float(dist[observed:].sum())


def annotate_concentration(
    meta: pd.DataFrame, group_map: dict[str, str], n_total: int
) -> pd.DataFrame:
    """Add n_groups / expected_n_groups / concentration / p_single per cluster."""
    sizes = (
        pd.Series([group_map[e] for e in group_map])
        .value_counts()
        .to_numpy()
    )
    out = meta.copy()
    out["n_groups"] = out["exp_set"].map(
        lambda s: len({group_map[e] for e in s if e in group_map})
    )
    # Cache by prevalence: both statistics depend only on p and the group sizes.
    cache_e: dict[int, float] = {}
    cache_s: dict[int, float] = {}
    for p in sorted(set(out["prevalence"])):
        cache_e[p] = expected_n_groups(int(p), sizes, n_total)
        cache_s[p] = prob_single_group(int(p), sizes, n_total)
    out["expected_n_groups"] = out["prevalence"].map(cache_e)
    out["concentration"] = out["n_groups"] / out["expected_n_groups"]
    out["p_single_expected"] = out["prevalence"].map(cache_s)
    out["is_single_group"] = out["n_groups"] == 1
    return out


def summarize(annotated: pd.DataFrame, split_cols: list[str]) -> pd.DataFrame:
    """Aggregate per split, with the exact single-group enrichment test.

    Reports both the median of per-cluster ratios and a pooled ratio
    (sum observed / sum expected). The pooled version is the more stable
    summary: per-cluster ratios are extremely granular at low prevalence, so
    their median can jump between two adjacent values.
    """
    rows = []
    for key, sub in annotated.groupby(split_cols, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        probs = sub["p_single_expected"].to_numpy()
        observed_single = int(sub["is_single_group"].sum())
        rows.append(
            {
                **dict(zip(split_cols, key)),
                "n_clusters": len(sub),
                "median_prevalence": sub["prevalence"].median(),
                "median_n_groups": sub["n_groups"].median(),
                "median_expected": round(float(sub["expected_n_groups"].median()), 2),
                "median_concentration": round(float(sub["concentration"].median()), 3),
                "pooled_concentration": round(
                    float(sub["n_groups"].sum() / sub["expected_n_groups"].sum()), 3
                ),
                "n_single_group": observed_single,
                "expected_single_group": round(float(probs.sum()), 2),
                "single_group_enrichment": (
                    round(observed_single / probs.sum(), 2) if probs.sum() > 0 else np.nan
                ),
                "single_group_p": poisson_binomial_sf(probs, observed_single),
            }
        )
    return pd.DataFrame(rows)


def plot_concentration(
    annotated: pd.DataFrame, head: str, level: str, out_stem: Path, subtitle: str
) -> None:
    """n_groups against prevalence, with the random expectation overlaid.

    Points below the curve are more tissue-concentrated than chance. Plotted
    against prevalence rather than binned because the expectation itself is a
    function of prevalence -- binning would hide the comparison being made.
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    classes = sorted(annotated["motif_class"].unique())
    palette = dict(zip(classes, ["#404040", "#b2182b", "#1b7837", "#2166ac"]))
    rng = np.random.default_rng(0)
    for cls in classes:
        sub = annotated[annotated["motif_class"] == cls]
        # jitter only for legibility; both axes are small integers
        ax.scatter(
            sub["prevalence"] * np.exp(rng.normal(0, 0.02, len(sub))),
            sub["n_groups"] + rng.normal(0, 0.08, len(sub)),
            s=10, alpha=0.55, lw=0, color=palette.get(cls, "#777777"),
            label=f"{cls} (n={len(sub)})",
        )
    curve = annotated[["prevalence", "expected_n_groups"]].drop_duplicates().sort_values(
        "prevalence"
    )
    ax.plot(
        curve["prevalence"], curve["expected_n_groups"],
        color="black", lw=1.4, ls="--", label="random expectation",
    )
    ax.set_xscale("log")
    # Matplotlib's default log minor-tick labels collide badly over the ~2-200
    # range this axis spans, so set explicit integer ticks and suppress minors.
    ticks = [t for t in (2, 3, 5, 10, 20, 50, 100, 200) if t <= annotated["prevalence"].max()]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Experiments the cluster was discovered in")
    ax.set_ylabel(f"Distinct {level} groups")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.suptitle(
        f"Discovery concentration — {head} head, {level} level\n{subtitle}", fontsize=9
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
        help="attribution head whose compendium to test (default: profile)",
    )
    parser.add_argument(
        "--group-level", default="tissue", choices=["tissue", "biosample"],
        help="what counts as a group; run both, a conclusion holding at both "
             "is not a grouping artifact (default: tissue)",
    )
    parser.add_argument(
        "--cluster-metadata", type=Path, default=None, metavar="PATH",
        help="override motifcompendium_{head}_cluster_metadata.tsv",
    )
    parser.add_argument(
        "--min-reads", type=float, default=10_000_000, metavar="N",
        help="drop experiments below N total reads (default: 10000000)",
    )
    parser.add_argument(
        "--min-cluster-experiments", type=int, default=2, metavar="N",
        help="only test clusters seen in >= N experiments. Raise to 3 to check "
             "whether a result rests on clusters sitting at the reproducibility "
             "floor (default: 2)",
    )
    parser.add_argument(
        "--jaspar-score-threshold", type=float, default=None, metavar="X",
        help="treat a cluster as TF-matched only above this JASPAR score; by "
             "default any non-empty jaspar_name counts, which admits matches "
             "as weak as ~0.82",
    )
    parser.add_argument(
        "--annotation-tsv", type=Path, default=None, metavar="PATH",
        help="curated cluster_final<TAB>class table, overriding the JASPAR proxy",
    )
    parser.add_argument(
        "--split-by", nargs="+", default=["motif_class"],
        choices=["motif_class", "abundance_band", "posneg"],
        help="report rows to split by (default: motif_class)",
    )
    parser.add_argument(
        "--biosample-groups", type=Path, default=None, metavar="PATH",
        help="curated biosample<TAB>group table, used when --group-level tissue",
    )
    parser.add_argument(
        "--write-group-tsv", type=Path, default=None, metavar="PATH",
        help="write the resolved biosample->group table here and exit",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figures" / "motif_atlas",
        metavar="DIR", help="output directory (default: figures/motif_atlas/)",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        experiments_cfg = yaml.safe_load(f)["experiments"]

    tissue_map, biosample_map = load_group_map(
        experiments_cfg, args.biosample_groups, quiet=args.group_level != "tissue"
    )
    if args.write_group_tsv:
        write_group_tsv(biosample_map, tissue_map, args.write_group_tsv)
        return

    read_counts = load_read_counts()
    keep = {e for e in experiments_cfg if read_counts.get(e, 0) >= args.min_reads}

    metadata_path = args.cluster_metadata or (
        MC_DIR / f"motifcompendium_{args.head}_cluster_metadata.tsv"
    )
    if not metadata_path.exists():
        print(f"ERROR: cluster metadata not found: {metadata_path}", file=sys.stderr)
        print(
            "This test needs total_seqlets/n_motifs/jaspar_name, so the "
            "pattern-to-cluster mapping is not sufficient.",
            file=sys.stderr,
        )
        sys.exit(1)

    meta, universe = load_presence(metadata_path, keep_experiments=keep)
    meta = meta[meta["prevalence"] >= args.min_cluster_experiments].reset_index(drop=True)
    if meta.empty:
        print("ERROR: no clusters survived filtering", file=sys.stderr)
        sys.exit(1)

    source_map = tissue_map if args.group_level == "tissue" else biosample_map
    group_map = {e: source_map[e] for e in universe}
    n_total = len(universe)

    meta["motif_class"] = classify_clusters(meta, args.annotation_tsv)
    if args.jaspar_score_threshold is not None:
        if "jaspar_score" not in meta.columns:
            print("ERROR: no jaspar_score column to threshold", file=sys.stderr)
            sys.exit(1)
        weak = meta["jaspar_score"].fillna(0) < args.jaspar_score_threshold
        meta.loc[weak, "motif_class"] = "unmatched (core promoter / repeat)"
        print(
            f"{args.head}: JASPAR score floor {args.jaspar_score_threshold} moved "
            f"{int(weak.sum())} clusters into the unmatched class",
            file=sys.stderr,
        )
    if "seqlets_per_motif" in meta.columns:
        meta["abundance_band"] = pd.cut(
            meta["seqlets_per_motif"], list(ABUNDANCE_BINS), labels=list(ABUNDANCE_LABELS)
        )
    split_cols = [c for c in args.split_by if c in meta.columns]
    if not split_cols:
        print(f"ERROR: none of {args.split_by} available", file=sys.stderr)
        sys.exit(1)

    print(
        f"{args.head}: {len(meta):,} clusters over {n_total} experiments in "
        f"{len(set(group_map.values()))} {args.group_level} groups "
        f"(prevalence >= {args.min_cluster_experiments})",
        file=sys.stderr,
    )

    annotated = annotate_concentration(meta, group_map, n_total)
    summary = summarize(annotated, split_cols)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"motif_concentration_{args.head}_{args.group_level}"
    keep_cols = [
        c for c in (
            "cluster_final", "posneg", "jaspar_name", "jaspar_score", "motif_class",
            "abundance_band", "n_motifs", "total_seqlets", "seqlets_per_motif",
            "prevalence", "n_groups", "expected_n_groups", "concentration",
            "p_single_expected", "is_single_group",
        ) if c in annotated.columns
    ]
    annotated[keep_cols].to_csv(stem.with_suffix(".tsv"), sep="\t", index=False)
    summary_path = stem.parent / f"{stem.name}_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"Saved {stem.with_suffix('.tsv')}", file=sys.stderr)
    print(f"Saved {summary_path}", file=sys.stderr)

    subtitle = (
        f"n={n_total} experiments, {len(annotated):,} clusters, "
        f"prevalence >= {args.min_cluster_experiments}"
    )
    plot_concentration(annotated, args.head, args.group_level, stem, subtitle)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(f"\n{args.group_level}-level concentration:", file=sys.stderr)
        print(summary.to_string(index=False), file=sys.stderr)
    print(
        "\nconcentration ~1 = spread like a random draw of experiments; "
        "<<1 = concentrated.\nsingle_group_p is the exact Poisson-binomial "
        "P[X >= observed] for one-group clusters.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
