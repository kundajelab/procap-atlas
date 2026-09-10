#!/usr/bin/env python3
"""Atlas-wide motif x experiment hit-density matrix and tissue-specificity scores.

Manuscript panel (Fig. 2a). Aggregates every experiment's Fi-NeMo hits by the
atlas-wide MotifCompendium cluster they were linked to, normalizes to hits per
peak, and renders a clustered motif x experiment heatmap with biosample-group
and read-depth annotations -- the panel separating a universal core-promoter/
housekeeping block from tissue-restricted lineage-TF blocks.

Input is each experiment's `hits_linked.tsv`, written by
src/bpnet/hitcall/link_hits_to_compendium.py, whose `compendium_motif_name`
column is the only cross-experiment-comparable motif identity available: hits
are called per experiment against that experiment's own MoDISco motifs, so the
raw `motif_name` means a different motif in every experiment.

Two confounds are made explicit rather than hidden, because both are real and
both will be asked about:

  Discovery power. A cluster can only receive hits in an experiment whose own
    MoDISco run discovered a motif assigned to it. So a zero is ambiguous: the
    motif may be unused, or may simply never have been discovered at that
    library size. Pass --cluster-metadata to separate the two -- cells whose
    experiment is absent from the cluster's own `experiments` list are reported
    as undiscovered (and, with --mask-undiscovered, dropped from the heatmap
    rather than drawn as true zeros).

  Read depth. Deeper libraries yield more peaks, more seqlets, and more
    discovered motifs. Hit counts are normalized per peak, and a depth strip is
    drawn alongside the biosample-group strip so depth-driven column structure
    is visible instead of being mistaken for tissue structure.

Specificity is scored over biosample *groups* rather than experiments, so that
whichever biosamples happen to be replicated many times (HCT116 n=16,
Metastatic Breast Carcinoma in the Brain n=10, PBMC n=8) cannot dominate:
per-group mean hits/peak is normalized to a distribution q over the groups
where the cluster was detected, and specificity is 1 - H(q)/log(G), so 0 is
perfectly ubiquitous and 1 is confined to a single group.

Outputs (in --out-dir):
  motif_hit_density_{head}.tsv           cluster x experiment hits/peak (full, unfiltered)
  motif_hit_density_{head}_status.tsv    per-cell discovered/undiscovered mask (with --cluster-metadata)
  motif_specificity_{head}.tsv           per-cluster specificity, breadth, top group
  motif_hit_density_{head}.{png,pdf}     the clustered heatmap panel
  motif_hit_density_{head}_columns.tsv   per-experiment column annotations

Usage:
    python src/analysis/motif_hit_density.py
    python src/analysis/motif_hit_density.py --head count --min-trim-len 6
    python src/analysis/motif_hit_density.py --cluster-metadata motifcompendium/bpnet/motifcompendium_profile_cluster_metadata.tsv --mask-undiscovered
    python src/analysis/motif_hit_density.py --top-n 60 --sort-by specificity
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bpnet" / "hitcall"))
from _biosample_groups import load_group_map, write_group_tsv  # noqa: E402
from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, trim_suffix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
HITCALL_DIR = REPO_ROOT / "hitcalls" / "bpnet"
MODISCO_DIR = REPO_ROOT / "modisco" / "bpnet"
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet"

# cluster_motifs.py's own default exclusions, mirrored so the matrix columns
# match the experiment universe the compendium was actually built from.
DEFAULT_BLACKLIST = ("ENCSR973QQI",)


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


def resolve_dirs(
    experiment: str,
    head: str,
    min_trim_len: int | None,
    hitcall_dir: Path = HITCALL_DIR,
) -> tuple[Path, Path]:
    """Return (exp_dir, hits_dir) for an experiment/head.

    peaks.narrowPeak is cached at the exp_dir level because it depends only on
    experiment/head/--region-width, while hit files move into a trim-suffixed
    subdirectory whenever hit calling used non-default trimming -- so the two
    are resolved separately. Mirrors link_hits_to_compendium.py.
    """
    trim_coords = (
        MODISCO_DIR / f"{experiment}_{head}_trim_coords_min{min_trim_len}bp.tsv"
        if min_trim_len is not None
        else None
    )
    suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, trim_coords)
    exp_dir = hitcall_dir / f"{experiment}_{head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir
    return exp_dir, hits_dir


def count_peaks(exp_dir: Path, hits: pd.DataFrame) -> tuple[int, str]:
    """Number of peaks hits were called over, and where that number came from.

    Prefers peaks.narrowPeak, the exact region set call_hits_bpnet.py wrote and
    the arrays were aligned to. Falls back to max(peak_id) + 1, which
    *underestimates* the region count whenever trailing peaks received no hits
    at all -- hence the explicit provenance string, so a fallback column can be
    spotted in the output rather than silently skewing that experiment's
    densities upward.
    """
    narrowpeak = exp_dir / "peaks.narrowPeak"
    if narrowpeak.exists():
        with open(narrowpeak) as f:
            return sum(1 for line in f if line.strip()), "peaks.narrowPeak"
    if "peak_id" in hits.columns and len(hits):
        return int(hits["peak_id"].max()) + 1, "max(peak_id)+1 [underestimate]"
    return 0, "unavailable"


def collect_densities(
    experiments: list[str],
    head: str,
    min_trim_len: int | None,
    hitcall_dir: Path = HITCALL_DIR,
    quiet: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every experiment's hits_linked.tsv into a cluster x experiment table.

    Returns (density, column_info): `density` is hits per peak indexed by
    compendium motif name with one column per experiment, `column_info` carries
    per-experiment peak counts, hit totals, and peak-count provenance.
    """
    series: dict[str, pd.Series] = {}
    info_rows = []
    for experiment in experiments:
        exp_dir, hits_dir = resolve_dirs(experiment, head, min_trim_len, hitcall_dir)
        hits_path = hits_dir / "hits_linked.tsv"
        if not hits_path.exists():
            if not quiet:
                print(f"  {experiment}: no hits_linked.tsv, skipping", file=sys.stderr)
            continue
        hits = pd.read_csv(
            hits_path, sep="\t", usecols=lambda c: c in {"compendium_motif_name", "peak_id"}
        )
        if "compendium_motif_name" not in hits.columns:
            print(
                f"  {experiment}: {hits_path} has no compendium_motif_name column; "
                "rerun link_hits_to_compendium.py",
                file=sys.stderr,
            )
            continue
        n_peaks, provenance = count_peaks(exp_dir, hits)
        if n_peaks <= 0:
            print(f"  {experiment}: peak count unavailable, skipping", file=sys.stderr)
            continue
        # Hits with no compendium cluster (unmapped local motifs) are counted
        # for the totals but cannot be placed in a row.
        n_unmapped = int(hits["compendium_motif_name"].isna().sum())
        counts = hits["compendium_motif_name"].dropna().value_counts()
        series[experiment] = counts / n_peaks
        info_rows.append(
            {
                "experiment": experiment,
                "n_peaks": n_peaks,
                "n_hits": len(hits),
                "n_hits_unmapped": n_unmapped,
                "n_clusters_detected": int(counts.size),
                "hits_per_peak": len(hits) / n_peaks,
                "peak_count_source": provenance,
            }
        )

    if not series:
        return pd.DataFrame(), pd.DataFrame()
    density = pd.DataFrame(series).fillna(0.0)
    density.index.name = "compendium_motif_name"
    return density, pd.DataFrame(info_rows).set_index("experiment")


def discovery_status(
    density: pd.DataFrame, cluster_metadata: Path
) -> pd.DataFrame:
    """Per-cell "discovered"/"undiscovered" mask from the compendium metadata.

    A cluster can only receive hits in experiments that contributed a motif to
    it, so cells outside that set are structurally zero (never discovered) and
    must not be read as "this motif is unused in this tissue".
    """
    meta = pd.read_csv(cluster_metadata, sep="\t")
    if not {"cluster_final", "posneg", "experiments"} <= set(meta.columns):
        raise ValueError(
            f"{cluster_metadata} needs cluster_final, posneg and experiments columns"
        )
    # cluster_motifs.py names clusters f"{posneg}_patterns.{cluster_final}"
    # when exporting the pattern->cluster mapping, which is the identity that
    # ends up in hits_linked.tsv's compendium_motif_name.
    name = (
        meta["posneg"].astype(str) + "_patterns." + meta["cluster_final"].astype(int).astype(str)
    )
    exp_sets = {
        n: {e.strip() for e in str(c).split(",") if e.strip()}
        for n, c in zip(name, meta["experiments"])
    }
    status = pd.DataFrame(
        "undiscovered", index=density.index, columns=density.columns, dtype=object
    )
    for motif in density.index:
        contributing = exp_sets.get(motif)
        if contributing is None:
            continue
        cols = [c for c in density.columns if c in contributing]
        status.loc[motif, cols] = "discovered"
    return status


def specificity_table(
    density: pd.DataFrame, group_map: dict[str, str]
) -> pd.DataFrame:
    """Per-cluster tissue specificity from group-mean hit densities.

    Averaging within biosample group before scoring keeps heavily replicated
    biosamples from dominating. Specificity is 1 - H(q)/log(G) over the groups
    with nonzero density, so 0 is uniform across all groups and 1 is confined
    to one group.
    """
    groups = pd.Series({c: group_map.get(c, "other") for c in density.columns})
    group_means = density.T.groupby(groups).mean().T  # clusters x groups
    n_groups = group_means.shape[1]

    totals = group_means.sum(axis=1)
    q = group_means.div(totals.replace(0, np.nan), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(q * np.log(q)).sum(axis=1, skipna=True)
    max_entropy = np.log(n_groups) if n_groups > 1 else 1.0
    spec = 1.0 - entropy / max_entropy

    top_group = group_means.idxmax(axis=1)
    top_share = q.max(axis=1)
    n_detected = (density > 0).sum(axis=1)
    out = pd.DataFrame(
        {
            "mean_hits_per_peak": density.mean(axis=1),
            # Density averaged over only the experiments where the motif was
            # actually called. Unlike the two columns above, this does not
            # shrink just because a motif is restricted to few tissues, so it
            # is the right measure of "when this motif is used, how heavily",
            # and the one select_clusters gates tissue-restricted motifs on.
            "mean_hits_per_peak_detected": density.sum(axis=1)
            / n_detected.replace(0, np.nan),
            "total_hits_per_peak": density.sum(axis=1),
            "n_experiments_detected": n_detected,
            "n_groups_detected": (group_means > 0).sum(axis=1),
            "specificity": spec,
            "top_group": top_group,
            "top_group_share": top_share,
        }
    )
    out.index.name = "compendium_motif_name"
    return out.join(group_means.add_prefix("group_mean:"))


def annotate_labels(
    motifs: pd.Index, cluster_metadata: Path | None
) -> dict[str, str]:
    """Row labels of the form "pos_patterns.42 (SP1)" where JASPAR matched."""
    labels = {m: m for m in motifs}
    if cluster_metadata is None or not Path(cluster_metadata).exists():
        return labels
    meta = pd.read_csv(cluster_metadata, sep="\t")
    if "jaspar_name" not in meta.columns:
        return labels
    name = (
        meta["posneg"].astype(str) + "_patterns." + meta["cluster_final"].astype(int).astype(str)
    )
    for motif, jaspar in zip(name, meta["jaspar_name"]):
        if motif in labels and isinstance(jaspar, str) and jaspar:
            labels[motif] = f"{motif} ({jaspar})"
    return labels


def depth_colors(read_counts: dict[str, float], columns: list[str]) -> pd.Series:
    """Fig. 1d's read-depth tiers: red <10M, yellow 10-20M, blue >20M."""
    tier_color = {"<10M": "#d73027", "10-20M": "#fee090", ">20M": "#4575b4",
                  "unknown": "#cccccc"}

    def tier(exp_id: str) -> str:
        n = read_counts.get(exp_id)
        if n is None:
            return "unknown"
        if n < 10e6:
            return "<10M"
        return "10-20M" if n < 20e6 else ">20M"

    return pd.Series({c: tier_color[tier(c)] for c in columns})


def select_clusters(
    spec: pd.DataFrame, top_n: int, sort_by: str
) -> pd.DataFrame:
    """Pick which clusters to draw.

    "balanced" (the default) is what the panel actually needs: ranking purely
    by total hits fills every row with ubiquitous and broadly-used motifs and
    crowds out the tissue-restricted lineage motifs the panel exists to show,
    while ranking purely by specificity fills it with low-abundance noise. So
    half the slots go to the most-used clusters and half to the most
    tissue-restricted ones.

    The specificity half is gated on `mean_hits_per_peak_detected` -- density
    among the experiments where the motif was actually called -- and not on
    summed or mean-over-all-columns density, both of which shrink in direct
    proportion to how few tissues a motif is restricted to. Gating on those
    would systematically exclude exactly the tissue-restricted motifs this
    half of the panel exists to show.
    """
    if top_n <= 0:
        return spec.sort_values("total_hits_per_peak", ascending=False)

    if sort_by == "balanced":
        n_usage = max(1, top_n // 2)
        by_usage = spec.sort_values("total_hits_per_peak", ascending=False).head(n_usage)
        rest = spec.drop(index=by_usage.index)
        if not rest.empty:
            floor = spec["mean_hits_per_peak_detected"].median()
            substantial = rest[rest["mean_hits_per_peak_detected"] >= floor]
            if substantial.empty:
                substantial = rest
            by_spec = substantial.sort_values("specificity", ascending=False).head(
                top_n - len(by_usage)
            )
        else:
            by_spec = rest
        picked = pd.concat([by_usage, by_spec])
        # Draw in usage order so the universal block reads top-down.
        return picked.sort_values("total_hits_per_peak", ascending=False)

    sort_col = {
        "total_hits": "total_hits_per_peak",
        "specificity": "specificity",
        "breadth": "n_experiments_detected",
    }[sort_by]
    return spec.sort_values(sort_col, ascending=False).head(top_n)


def plot_heatmap(
    density: pd.DataFrame,
    status: pd.DataFrame | None,
    spec: pd.DataFrame,
    group_map: dict[str, str],
    read_counts: dict[str, float],
    labels: dict[str, str],
    head: str,
    out_stem: Path,
    mask_undiscovered: bool,
) -> None:
    """Clustered log-scale heatmap with biosample-group and depth column strips."""
    plot_df = np.log10(density + 1e-4)
    mask = None
    if mask_undiscovered and status is not None:
        mask = (status == "undiscovered").reindex_like(density).fillna(False)

    groups = [group_map.get(c, "other") for c in density.columns]
    unique_groups = sorted(set(groups))
    palette = sns.color_palette("tab20", len(unique_groups))
    group_colors = dict(zip(unique_groups, palette))
    col_colors = pd.DataFrame(
        {
            "biosample group": pd.Series(groups, index=density.columns).map(group_colors),
            "read depth": depth_colors(read_counts, list(density.columns)),
        }
    )

    row_colors = pd.DataFrame(
        {
            "specificity": pd.Series(
                sns.color_palette("Purples", as_cmap=True)(
                    spec.loc[density.index, "specificity"].clip(0, 1).fillna(0)
                ).tolist(),
                index=density.index,
            ).map(lambda rgba: rgba[:3])
        }
    )

    g = sns.clustermap(
        plot_df,
        mask=mask,
        method="ward",
        metric="euclidean",
        cmap="magma",
        col_colors=col_colors,
        row_colors=row_colors,
        figsize=(max(9.0, 0.06 * plot_df.shape[1] + 5), max(7.0, 0.16 * plot_df.shape[0] + 3)),
        xticklabels=False,
        yticklabels=[labels.get(m, m) for m in plot_df.index],
        cbar_kws={"label": "log$_{10}$ hits per peak"},
        dendrogram_ratio=(0.10, 0.06),
    )
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.set_xlabel(f"{plot_df.shape[1]} experiments")
    g.ax_heatmap.tick_params(axis="y", labelsize=6)

    # Placed in figure coordinates below the heatmap: clustermap's own axes
    # are all occupied (y tick labels sit to the right of the heatmap, and a
    # legend on ax_col_dendrogram lands on top of the heatmap itself).
    group_handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[grp], label=grp)
        for grp in unique_groups
    ]
    depth_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=f"{tier} reads")
        for tier, color in (
            ("<10M", "#d73027"), ("10-20M", "#fee090"), (">20M", "#4575b4"),
        )
    ]
    g.figure.legend(
        handles=group_handles, title="Biosample group", ncol=5, fontsize=6,
        title_fontsize=7, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
    )
    g.figure.legend(
        handles=depth_handles, title="Library size", ncol=3, fontsize=6,
        title_fontsize=7, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
    )
    g.figure.suptitle(
        f"Motif hit density across the PRO-cap atlas — {head} head", fontsize=10, y=1.0
    )

    for ext in ("png", "pdf"):
        path = out_stem.with_suffix(f".{ext}")
        g.figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}", file=sys.stderr)
    plt.close(g.figure)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--head", default="profile", choices=["profile", "count"],
        help="attribution head the hits were called against (default: profile)",
    )
    parser.add_argument(
        "--min-trim-len", type=int, default=None, metavar="BP",
        help="must match hitcall/launch.py's --min-trim-len, if any; resolves "
             "the trim-coords-suffixed hits directory",
    )
    parser.add_argument(
        "--min-reads", type=float, default=10_000_000, metavar="N",
        help="drop experiments below N total reads (default: 10000000, "
             "matching cluster_motifs.py)",
    )
    parser.add_argument(
        "--blacklist", nargs="*", default=list(DEFAULT_BLACKLIST), metavar="EXP_ID",
        help=f"experiments to exclude (default: {' '.join(DEFAULT_BLACKLIST)})",
    )
    parser.add_argument(
        "--keep-uncapped", action="store_true",
        help="keep uncapped-enrichment experiments, which cluster_motifs.py drops",
    )
    parser.add_argument(
        "--cluster-metadata", type=Path, default=None, metavar="PATH",
        help="motifcompendium_{head}_cluster_metadata.tsv, for JASPAR row labels "
             "and the discovered/undiscovered mask (default: auto-detect)",
    )
    parser.add_argument(
        "--mask-undiscovered", action="store_true",
        help="leave cells blank where the cluster was never discovered in that "
             "experiment, instead of drawing them as true zeros",
    )
    parser.add_argument(
        "--top-n", type=int, default=60, metavar="N",
        help="clusters to draw in the heatmap; 0 draws all (default: 60)",
    )
    parser.add_argument(
        "--sort-by", default="balanced",
        choices=["balanced", "total_hits", "specificity", "breadth"],
        help="how to pick the --top-n clusters drawn; 'balanced' splits the "
             "slots between most-used and most tissue-restricted so neither "
             "crowds the other out (default: balanced)",
    )
    parser.add_argument(
        "--min-experiments", type=int, default=2, metavar="N",
        help="drop clusters detected in fewer than N experiments (default: 2)",
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
        "--hitcall-dir", type=Path, default=HITCALL_DIR, metavar="DIR",
        help=f"root of per-experiment hit-call outputs (default: "
             f"{HITCALL_DIR.relative_to(REPO_ROOT)}/)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figures" / "motif_atlas",
        metavar="DIR", help="output directory (default: figures/motif_atlas/)",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        experiments_cfg = yaml.safe_load(f)["experiments"]

    group_map, biosample_map = load_group_map(experiments_cfg, args.biosample_groups)
    if args.write_group_tsv:
        write_group_tsv(biosample_map, group_map, args.write_group_tsv)
        return

    read_counts = load_read_counts()
    selected = []
    for exp_id, exp in experiments_cfg.items():
        if exp_id in args.blacklist:
            continue
        if not args.keep_uncapped and "uncapped" in str(
            exp.get("library_construction", "")
        ).lower():
            continue
        if read_counts.get(exp_id, 0) < args.min_reads:
            continue
        selected.append(exp_id)
    print(
        f"{args.head}: {len(selected)} experiments selected of {len(experiments_cfg)}",
        file=sys.stderr,
    )

    density, column_info = collect_densities(
        selected, args.head, args.min_trim_len,
        hitcall_dir=args.hitcall_dir, quiet=args.quiet,
    )
    if density.empty:
        print(
            "ERROR: no hits_linked.tsv files found. Run "
            "src/bpnet/hitcall/launch_link.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"{args.head}: {density.shape[0]:,} clusters x {density.shape[1]} experiments",
        file=sys.stderr,
    )

    cluster_metadata = args.cluster_metadata
    if cluster_metadata is None:
        candidate = MC_DIR / f"motifcompendium_{args.head}_cluster_metadata.tsv"
        cluster_metadata = candidate if candidate.exists() else None

    status = None
    if cluster_metadata is not None:
        status = discovery_status(density, cluster_metadata)
        n_undiscovered = int((status == "undiscovered").to_numpy().sum())
        total = status.size
        print(
            f"{args.head}: {n_undiscovered:,}/{total:,} cells "
            f"({n_undiscovered / total:.1%}) are structurally zero "
            "(cluster never discovered in that experiment)",
            file=sys.stderr,
        )
    elif args.mask_undiscovered:
        print(
            "ERROR: --mask-undiscovered needs --cluster-metadata "
            "(or the default cluster metadata to exist)",
            file=sys.stderr,
        )
        sys.exit(1)

    spec = specificity_table(density, group_map)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    density.to_csv(args.out_dir / f"motif_hit_density_{args.head}.tsv", sep="\t")
    spec.to_csv(args.out_dir / f"motif_specificity_{args.head}.tsv", sep="\t")
    column_info.assign(
        biosample=[biosample_map.get(e, "unknown") for e in column_info.index],
        biosample_group=[group_map.get(e, "other") for e in column_info.index],
        total_reads=[read_counts.get(e, np.nan) for e in column_info.index],
    ).to_csv(args.out_dir / f"motif_hit_density_{args.head}_columns.tsv", sep="\t")
    if status is not None:
        status.to_csv(args.out_dir / f"motif_hit_density_{args.head}_status.tsv", sep="\t")
    for name in ("motif_hit_density", "motif_specificity"):
        print(f"Saved {args.out_dir / f'{name}_{args.head}.tsv'}", file=sys.stderr)

    plot_spec = spec[spec["n_experiments_detected"] >= args.min_experiments]
    ordered = select_clusters(plot_spec, args.top_n, args.sort_by)
    if ordered.empty:
        print("ERROR: no clusters left to plot after filtering", file=sys.stderr)
        sys.exit(1)
    print(
        f"{args.head}: drawing {len(ordered)} of {len(plot_spec)} clusters "
        f"(--sort-by {args.sort_by})",
        file=sys.stderr,
    )

    plot_heatmap(
        density.loc[ordered.index],
        status,
        spec,
        group_map,
        read_counts,
        annotate_labels(density.index, cluster_metadata),
        args.head,
        args.out_dir / f"motif_hit_density_{args.head}",
        args.mask_undiscovered,
    )

    print("\nMost tissue-restricted clusters (by specificity):", file=sys.stderr)
    top = plot_spec.sort_values("specificity", ascending=False).head(10)
    for motif, row in top.iterrows():
        print(
            f"  {motif:<28} spec={row['specificity']:.3f} "
            f"groups={int(row['n_groups_detected']):>2} top={row['top_group']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
