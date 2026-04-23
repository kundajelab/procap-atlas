import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
MODISCO_DIR = REPO_ROOT / "modisco" / "bpnet"
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet"
JASPAR_PATH = (
    REPO_ROOT / "data" / "JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt"
)

# Filter thresholds from MotifCompendium tutorial 6 (adjust as needed)
_FILTER_RULES = [
    # (name,                metric,                     op,   threshold)
    ("1_singlepeak",        "motif_entropy",             "<",  0.6),
    ("2_noisemix",          "motif_entropy",             ">",  0.8),
    ("3_noisypeaks",        "weighted_base_entropy",     ">",  0.6),
    ("4_broadsingle_1",     "weighted_position_entropy", ">",  0.85),
    ("4_broadsingle_2",     "posbase_entropy_score",     ">",  0.5),
    ("5_gcbias_1",          "copair_entropy_score",      ">",  0.35),
    ("5_gcbias_2",          "copair_composition",        ">",  0.45),
    ("6_dinucrepeat_1",     "dinuc_entropy_score",       ">",  0.5),
    ("6_dinucrepeat_2",     "dinuc_composition",         ">",  0.875),
    ("6_dinucrepeat_3",     "dinuc_score",               ">",  0.45),
    ("8_posneg_inverted",   "posneg_inverted",           "==", True),
    ("9_truncated",         "truncated",                 "==", True),
]

_METRIC_LIST = [
    "motif_entropy",
    "weighted_base_entropy",
    "weighted_position_entropy",
    "posbase_entropy_score",
    "copair_entropy_score",
    "copair_composition",
    "dinuc_entropy_score",
    "dinuc_composition",
    "dinuc_score",
    "posneg_inverted",
    "truncated",
]


def _apply_filter_threshold(mc, flag_col, metric, operation, threshold):
    ops = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    mc[flag_col] = mc[flag_col] | ops[operation](mc[metric], threshold)


def filter_mc(
    mc,
    min_seqlets=100,
    revive_threshold=0.9,
    tsv_path=None,
    pre_tsv_path=None,
    pre_html_path=None,
):
    """Flag and remove noisy motifs using default thresholds from tutorial 6.

    Steps:
      1. Calculate 11 filter metrics.
      2. Flag motifs violating any threshold (one column per rule).
      3. If JASPAR file exists and revive_threshold is set, un-flag motifs
         that match a known JASPAR motif above that similarity.
      4. Print a table of removed motifs with per-rule exclusion reasons.
      5. Return (mc_filtered, mc_removed).
    """
    utils_analysis.calculate_filters(mc=mc, metric_list=_METRIC_LIST)

    # Save pre-filter reports
    if pre_tsv_path is not None:
        display_cols = ["model", "posneg", "num_seqlets"] + _METRIC_LIST
        display_cols = [c for c in display_cols if c in mc.metadata.columns]
        mc.metadata[display_cols].to_csv(pre_tsv_path, sep="\t", index=False)
        print(f"  pre-filter table saved to {pre_tsv_path}")
    if pre_html_path is not None:
        mc.summary_table_html(str(pre_html_path))
        print(f"  pre-filter HTML report saved to {pre_html_path}")

    # Apply each rule into its own flag column so we can report reasons later
    all_rules = list(_FILTER_RULES) + [
        ("7_minseqlets", "num_seqlets", "<", min_seqlets)
    ]
    for name, metric, op, threshold in all_rules:
        col = f"flag_{name}"
        mc.metadata[col] = False
        _apply_filter_threshold(mc, col, metric, op, threshold)

    flag_cols = [f"flag_{name}" for name, *_ in all_rules]
    flag_col = "flag_remove"
    mc.metadata[flag_col] = mc.metadata[flag_cols].any(axis=1)

    n_flagged = mc.metadata[flag_col].sum()
    print(f"  flagged {n_flagged} / {len(mc.metadata)} motifs before revival")

    # Annotate with best JASPAR hit (always, if file exists)
    if JASPAR_PATH.exists():
        utils_analysis.assign_label_from_pfms(
            mc=mc, pfm_file=str(JASPAR_PATH), save_col_prefix="JASPAR"
        )
    else:
        print(f"  JASPAR file not found at {JASPAR_PATH}, skipping annotation")

    # Revive flagged motifs that match a known JASPAR motif
    if revive_threshold is not None and "JASPAR_score0" in mc.metadata.columns:
        mc[flag_col] = mc[flag_col] & (mc["JASPAR_score0"] < revive_threshold)
        # Also clear per-rule flags for revived motifs so they don't show in the table
        for col in flag_cols:
            mc.metadata.loc[~mc.metadata[flag_col] & mc.metadata[col], col] = False
        n_revived = n_flagged - mc.metadata[flag_col].sum()
        print(
            f"  revived {n_revived} motifs matching JASPAR "
            f"(similarity >= {revive_threshold})"
        )

    mc_filtered = mc[~mc.metadata[flag_col]]
    mc_removed = mc[mc.metadata[flag_col]]
    print(
        f"  kept {len(mc_filtered.metadata)}, removed {len(mc_removed.metadata)} motifs"
    )

    # Build and print exclusion reason table
    removed_meta = mc_removed.metadata.copy()
    rule_names = [name for name, *_ in all_rules]
    removed_meta["reasons"] = removed_meta[flag_cols].apply(
        lambda row: ", ".join(
            name for name, col in zip(rule_names, flag_cols) if row[col]
        ),
        axis=1,
    )
    display_cols = ["model", "posneg", "num_seqlets"] + _METRIC_LIST + ["reasons"]
    display_cols = [c for c in display_cols if c in removed_meta.columns]
    if tsv_path is not None:
        removed_meta[display_cols].to_csv(tsv_path, sep="\t", index=False)
        print(f"  exclusion table saved to {tsv_path}")

    return mc_filtered, mc_removed


def main():
    parser = argparse.ArgumentParser(
        description="Cluster modisco motifs across experiments using MotifCompendium."
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=10_000_000,
        help="Minimum total reads to include an experiment (default: 10M)",
    )
    parser.add_argument(
        "--blacklist",
        nargs="+",
        default=["ENCSR973QQI"],
        metavar="EXP_ID",
        help="Experiment IDs to exclude",
    )
    parser.add_argument(
        "--min-seqlets",
        type=int,
        default=50,
        help="Minimum seqlets per motif to keep (default: 100)",
    )
    parser.add_argument(
        "--revive-threshold",
        type=float,
        default=0.9,
        help="JASPAR similarity threshold to un-flag filtered motifs "
        "(default: 0.9; set to 0 to disable revival)",
    )
    parser.add_argument(
        "--within-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for within-experiment clustering (default: 0.95)",
    )
    parser.add_argument(
        "--across-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for cross-experiment clustering (default: 0.85)",
    )
    args = parser.parse_args()

    revive_threshold = args.revive_threshold if args.revive_threshold > 0 else None

    MotifCompendium.set_compute_options(
        max_cpus=4, use_gpu=True, max_chunk=1152, progress_bar=True
    )

    # Load experiment config and read counts
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    read_counts = dict(
        zip(
            *pd.read_csv(
                N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
            ).values.T
        )
    )

    blacklist = set(args.blacklist)

    # Filter experiments
    experiments = []
    for exp_id, meta in config["experiments"].items():
        if exp_id in blacklist:
            continue
        if "uncapped" in meta.get("library_construction", "").lower():
            continue
        if read_counts.get(exp_id, 0) < args.min_reads:
            continue
        experiments.append(exp_id)

    print(f"Using {len(experiments)} experiments after filtering")

    # Build dicts of experiment -> h5 path (skip missing files)
    count_dict = {}
    profile_dict = {}
    for exp_id in experiments:
        count_h5 = MODISCO_DIR / f"{exp_id}_count.modisco.h5"
        profile_h5 = MODISCO_DIR / f"{exp_id}_profile.modisco.h5"
        if count_h5.exists():
            count_dict[exp_id] = str(count_h5)
        if profile_h5.exists():
            profile_dict[exp_id] = str(profile_h5)

    print(
        f"Found {len(count_dict)} count h5 files, {len(profile_dict)} profile h5 files"
    )

    count_mc = MotifCompendium.build_from_modisco(count_dict)
    profile_mc = MotifCompendium.build_from_modisco(profile_dict)

    count_mc.save(str(MC_DIR / "motifcompendium_count_raw.mc"))
    profile_mc.save(str(MC_DIR / "motifcompendium_profile_raw.mc"))
    print("Raw MC objects saved")

    # Similarity distributions (pre-filter)
    utils_analysis.plot_similarity_distribution(
        count_mc, str(MC_DIR / "motifcompendium_count_similarity_distribution.html")
    )
    utils_analysis.plot_similarity_distribution(
        profile_mc, str(MC_DIR / "motifcompendium_profile_similarity_distribution.html")
    )

    # Motif filtering
    print("Filtering count motifs...")
    count_mc, count_mc_removed = filter_mc(
        count_mc,
        min_seqlets=args.min_seqlets,
        revive_threshold=revive_threshold,
        pre_tsv_path=MC_DIR / "motifcompendium_count_prefilter.tsv",
        pre_html_path=MC_DIR / "motifcompendium_count_prefilter.html",
        tsv_path=MC_DIR / "motifcompendium_count_removed.tsv",
    )
    count_mc.save(str(MC_DIR / "motifcompendium_count_filtered.mc"))
    count_mc_removed.save(str(MC_DIR / "motifcompendium_count_removed.mc"))

    print("Filtering profile motifs...")
    profile_mc, profile_mc_removed = filter_mc(
        profile_mc,
        min_seqlets=args.min_seqlets,
        revive_threshold=revive_threshold,
        pre_tsv_path=MC_DIR / "motifcompendium_profile_prefilter.tsv",
        pre_html_path=MC_DIR / "motifcompendium_profile_prefilter.html",
        tsv_path=MC_DIR / "motifcompendium_profile_removed.tsv",
    )
    profile_mc.save(str(MC_DIR / "motifcompendium_profile_filtered.mc"))
    profile_mc_removed.save(str(MC_DIR / "motifcompendium_profile_removed.mc"))
    print("Filtered MC objects saved")

    # Threshold sweep: visualize number of clusters vs similarity threshold
    similarity_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for mc, head in [(count_mc, "count"), (profile_mc, "profile")]:
        num_clusters = []
        for t in similarity_thresholds:
            mc.cluster(similarity_threshold=t, save_name=f"cluster_{t}")
            num_clusters.append(max(mc[f"cluster_{t}"]) + 1)
        fig, ax = plt.subplots()
        ax.plot(similarity_thresholds, num_clusters, marker="o")
        ax.set_xlabel("Similarity Threshold")
        ax.set_ylabel("Number of Clusters")
        ax.set_title(f"Number of Clusters vs Similarity Threshold ({head})")
        fig.savefig(
            str(MC_DIR / f"motifcompendium_{head}_threshold_sweep.png"), dpi=150
        )
        plt.close(fig)
        print(f"{head}: threshold sweep done")

    # Two-step clustering:
    #   Step 1 — collapse redundant motifs within each experiment at high threshold
    #   Step 2 — cluster across experiments, weighted by num_seqlets, using
    #             the within-experiment clusters as the unit of aggregation
    for mc, head in [(count_mc, "count"), (profile_mc, "profile")]:
        mc.cluster(
            similarity_threshold=args.within_threshold,
            save_name="cluster_within_model",
            cluster_within="model",
            cluster_on_weight="num_seqlets",
        )
        mc.cluster(
            similarity_threshold=args.across_threshold,
            save_name="cluster_final",
            cluster_on="cluster_within_model",
            cluster_on_weight="num_seqlets",
        )
        n_final = max(mc["cluster_final"]) + 1
        print(
            f"{head}: {n_final} final clusters "
            f"(within_threshold={args.within_threshold}, "
            f"across_threshold={args.across_threshold})"
        )
        utils_analysis.export_compendium_clustered_modisco(
            mc,
            "cluster_final",
            str(MC_DIR / f"motifcompendium_{head}_cluster_averages.h5"),
            weight_col="num_seqlets",
        )
        mc_avg = mc.cluster_averages("cluster_final")
        if JASPAR_PATH.exists():
            utils_analysis.assign_label_from_pfms(
                mc=mc_avg, pfm_file=str(JASPAR_PATH), save_col_prefix="JASPAR"
            )
        utils_analysis.export_compendium_meme(
            mc_avg,
            str(MC_DIR / f"motifcompendium_{head}_cluster_averages.meme"),
        )

        # Save cluster metadata table
        agg = (
            mc.metadata.groupby("cluster_final")
            .agg(
                n_motifs=("cluster_final", "count"),
                total_seqlets=("num_seqlets", "sum"),
                n_experiments=("model", "nunique"),
                experiments=("model", lambda x: ",".join(sorted(x.unique()))),
                posneg=("posneg", "first"),
            )
            .reset_index()
        )
        # Include best JASPAR hit per cluster (highest score among member motifs)
        if "JASPAR_score0" in mc.metadata.columns:
            jaspar_agg = (
                mc.metadata
                .sort_values("JASPAR_score0", ascending=False)
                .groupby("cluster_final")[["JASPAR_name0", "JASPAR_score0"]]
                .first()
                .rename(columns={"JASPAR_name0": "jaspar_name", "JASPAR_score0": "jaspar_score"})
                .reset_index()
            )
            agg = agg.merge(jaspar_agg, on="cluster_final", how="left")
        agg.to_csv(
            MC_DIR / f"motifcompendium_{head}_cluster_metadata.tsv",
            sep="\t",
            index=False,
        )
        print(f"{head}: cluster averages saved")

        # HTML report: motif logos for each cluster average
        mc_avg.summary_table_html(
            str(MC_DIR / f"motifcompendium_{head}_cluster_report.html")
        )
        # Per-cluster motif collection HTMLs, split by pos/neg (one file per cluster)
        cluster_html_dir = MC_DIR / f"motifcompendium_{head}_clusters"
        cluster_html_dir.mkdir(exist_ok=True)
        for posneg in ("pos", "neg"):
            mc_pn = mc[mc["posneg"] == posneg]
            for cluster_id in sorted(set(mc_pn["cluster_final"])):
                mc_cluster = mc_pn[mc_pn["cluster_final"] == cluster_id]
                mc_cluster.motif_collection_html(
                    str(cluster_html_dir / f"{posneg}_cluster_{cluster_id:04d}.html"),
                    "cluster_final",
                )
        print(f"{head}: per-cluster HTML reports saved to {cluster_html_dir}")


if __name__ == "__main__":
    main()
