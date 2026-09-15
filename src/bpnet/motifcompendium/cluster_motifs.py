import argparse
import html
import importlib.metadata
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import MotifCompendium.utils.motif as utils_motif
import MotifCompendium.utils.plotting as utils_plotting
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


def load_experiments(min_reads, blacklist):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    read_counts = dict(
        zip(
            *pd.read_csv(
                N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
            ).values.T
        )
    )

    # Skip blacklisted, uncapped, and low quality experiments.
    experiments = []
    for exp_id, meta in config["experiments"].items():
        if exp_id in blacklist:
            continue
        if "uncapped" in meta.get("library_construction", "").lower():
            continue
        if read_counts.get(exp_id, 0) < min_reads:
            continue
        experiments.append(exp_id)

    return experiments


def collect_modisco_paths(experiments, head):
    h5_paths = {}
    for exp_id in experiments:
        h5_path = MODISCO_DIR / f"{exp_id}_{head}.modisco.h5"
        if h5_path.exists():
            h5_paths[exp_id] = str(h5_path)
    return h5_paths


def assign_jaspar_labels(mc):
    if JASPAR_PATH.exists():
        utils_analysis.assign_label_from_pfms(
            mc=mc, pfm_file=str(JASPAR_PATH), save_col_prefix="JASPAR"
        )
    else:
        print(f"  JASPAR file not found at {JASPAR_PATH}, skipping annotation")


def mc_version():
    """The installed MotifCompendium version.

    `MotifCompendium` exposes no `__version__` — neither v1.0.19 nor v1.1.0
    defines one — so `getattr(MotifCompendium, "__version__", "unknown")`
    always recorded `"unknown"`, which defeats the point of stamping it. The
    version only exists in `setup.py`, so read it from the installed
    distribution metadata instead.
    """
    try:
        return importlib.metadata.version("MotifCompendium")
    except importlib.metadata.PackageNotFoundError:
        return getattr(MotifCompendium, "__version__", "unknown")


def cluster_reference(mc):
    """`cluster_averages`' alignment-frame setting, or None before v1.1.0.

    v1.1.0 added a `reference` parameter defaulting to `"medoid"`; before it,
    a cluster average was always framed on the cluster's lowest-indexed
    member. We never pass it, so this records whichever default applied —
    the cluster-average motifs in the h5, the MEME export and the logos
    differ between the two even at an identical partition.
    """
    params = inspect.signature(mc.cluster_averages).parameters
    if "reference" not in params:
        return "first (pre-v1.1.0)"
    return params["reference"].default


def resolve_algorithm(mc, algorithm=None, kmeans_iterations=None):
    """(algorithm, algorithm_kwargs) to pass to `mc.cluster`, plus a label.

    `mc.cluster`'s default changed silently between MotifCompendium releases:
    v1.0.18 ran `"cpm_leiden"` alone, v1.0.19 changed the default to
    `["cpm_leiden", "k_centroids"]` -- Leiden followed by an *uncapped*
    k-means refinement (`n_iterations=-1`, exiting only on exact membership
    equality). Nothing here passed `algorithm`, so the second stage arrived
    with an upstream version bump and no record in any output.

    It is cheap at count-head scale (5,639 motifs, ~4 min) and ruinous at
    profile-head scale (~38,000 motifs): each iteration rebuilds every
    centroid and recomputes an N x k float64 similarity on the GPU. A profile
    build sat in it for 85 h having previously completed in hours on v1.0.18.

    Passing `--algorithm cpm_leiden` restores the pre-v1.0.19 behaviour.
    `--kmeans-iterations` bounds the refinement instead of removing it.
    Returns the label so the choice can be recorded in the outputs.
    """
    if algorithm is None:
        default = inspect.signature(mc.cluster).parameters["algorithm"].default
        algorithm = list(default) if isinstance(default, (list, tuple)) else [default]
    algorithm = list(algorithm)

    algorithm_kwargs = None
    if kmeans_iterations is not None:
        kmeans_steps = [a for a in algorithm if a.startswith("k_")]
        if not kmeans_steps:
            raise ValueError(
                f"--kmeans-iterations was given but {algorithm} has no "
                "k-means step to bound; drop the flag or add k_centroids"
            )
        # keyed by algorithm name so only the k-means step is bounded
        algorithm_kwargs = {a: {"n_iterations": kmeans_iterations}
                            for a in kmeans_steps}

    label = "+".join(algorithm)
    if kmeans_iterations is not None:
        label += f" (n_iterations={kmeans_iterations})"
    return algorithm, algorithm_kwargs, label


def cluster_with(mc, algorithm, algorithm_kwargs, **cluster_kwargs):
    """`mc.cluster` with an explicit algorithm, tolerating older versions.

    `algorithm_kwargs` only exists from v1.0.19; on an older install the
    default was `"cpm_leiden"` alone, so there is nothing to bound and the
    argument is dropped rather than raising.
    """
    params = inspect.signature(mc.cluster).parameters
    if "algorithm" in params:
        cluster_kwargs["algorithm"] = (
            algorithm if len(algorithm) > 1 else algorithm[0]
        )
    if algorithm_kwargs is not None and "algorithm_kwargs" in params:
        cluster_kwargs["algorithm_kwargs"] = algorithm_kwargs
    mc.cluster(**cluster_kwargs)


def weighted_cluster_on(mc, similarity_threshold, save_name, cluster_on,
                        weight_col, algorithm=None, algorithm_kwargs=None):
    cluster_kwargs = {
        "similarity_threshold": similarity_threshold,
        "save_name": save_name,
        "cluster_on": cluster_on,
    }
    if "weight_col" in inspect.signature(mc.cluster).parameters:
        cluster_kwargs["weight_col"] = weight_col
    else:
        cluster_kwargs["cluster_on_weight"] = weight_col
    cluster_with(mc, algorithm or ["cpm_leiden"], algorithm_kwargs,
                 **cluster_kwargs)


def export_pattern_to_cluster_mapping(mc, head, out_dir=MC_DIR):
    """Map each experiment's own per-pattern Fi-NeMo motif_name (from calling
    hits directly against that experiment's own modisco.h5, the default
    hitcall/call_hits_bpnet.py source) to the atlas-wide MotifCompendium
    cluster it was assigned to, for link_hits_to_compendium.py.

    mc.metadata's "name" column is built by MotifCompendium.build_from_modisco
    as f"{model}-{posneg}.{pattern}" (posneg="pos"/"neg", pattern=the literal
    modisco.h5 pattern key e.g. "pattern_3") -- see
    MotifCompendium.MotifCompendium.build_from_modisco and
    MotifCompendium.utils.loader.load_modisco(s). Stripping the known
    f"{model}-{posneg}." prefix recovers "pattern", which combined with
    posneg reconstructs the exact motif_name Fi-NeMo assigns when it calls
    hits against that experiment's own modisco.h5 directly
    (f"{posneg}_patterns.{pattern}", e.g. "pos_patterns.pattern_3") --
    MotifCompendium uses "pos"/"neg", but modisco-lite h5s (and Fi-NeMo's own
    motif_name convention) use "pos_patterns"/"neg_patterns".
    """
    df = mc.metadata[["name", "model", "posneg", "cluster_final"]]
    prefixes = df["model"].astype(str) + "-" + df["posneg"].astype(str) + "."
    patterns = [
        name[len(prefix) :] for name, prefix in zip(df["name"], prefixes)
    ]
    mapping = pd.DataFrame(
        {
            "experiment": df["model"],
            "local_motif_name": df["posneg"].astype(str)
            + "_patterns."
            + pd.Series(patterns, index=df.index),
            "compendium_motif_name": df["posneg"].astype(str)
            + "_patterns."
            + df["cluster_final"].astype(int).astype(str),
        }
    )
    mapping_path = out_dir / f"motifcompendium_{head}_pattern_to_cluster.tsv"
    mapping.to_csv(mapping_path, sep="\t", index=False)
    print(f"{head}: pattern->cluster mapping saved to {mapping_path}")
    return mapping_path


def write_cluster_metadata(mc, head, logo_paths=None, out_dir=MC_DIR,
                           provenance=None):
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
        .sort_values("total_seqlets", ascending=False)
    )

    if "JASPAR_score0" in mc.metadata.columns:
        jaspar_agg = (
            mc.metadata.sort_values("JASPAR_score0", ascending=False)
            .groupby("cluster_final")[["JASPAR_name0", "JASPAR_score0"]]
            .first()
            .rename(
                columns={
                    "JASPAR_name0": "jaspar_name",
                    "JASPAR_score0": "jaspar_score",
                }
            )
            .reset_index()
        )
        agg = agg.merge(jaspar_agg, on="cluster_final", how="left")

    if logo_paths is not None:
        agg = agg.merge(logo_paths, on="cluster_final", how="left")

    # Provenance travels with the table the analyses actually read. A silent
    # upstream default change (MotifCompendium v1.0.19 adding k_centroids to
    # mc.cluster's default) left no trace in any output, and cost a week
    # before it was found by reading the library's git history. Two constant
    # columns are cheap insurance against the next one.
    if provenance:
        for key, value in provenance.items():
            agg[key] = value

    metadata_path = out_dir / f"motifcompendium_{head}_cluster_metadata.tsv"
    agg.to_csv(metadata_path, sep="\t", index=False)
    print(f"{head}: cluster metadata saved to {metadata_path}")
    return agg


def cluster_average_with_metadata(mc):
    return mc.cluster_averages(
        "cluster_final",
        weight_col="num_seqlets",
        aggregations=[
            ("cluster_final", "count", "n_motifs"),
            ("num_seqlets", "sum", "total_seqlets"),
            ("model", "unique", "n_experiments"),
            ("model", "concat", "experiments"),
            ("posneg", "concat", "posneg"),
        ],
    )


def ensure_forward_reverse_logos(mc, logo_trimming=True):
    if "logo (fwd)" not in mc.images():
        mc.add_logos(
            mc.get_standard_motif_stack(),
            "logo (fwd)",
            logo_trimming,
        )
    if "logo (rev)" not in mc.images():
        mc.add_logos(
            utils_motif.reverse_complement(mc.get_standard_motif_stack()),
            "logo (rev)",
            logo_trimming,
        )


def export_cluster_logo_svgs(
    mc, head, batch_size=100, logo_trimming=True, out_dir=MC_DIR
):
    logo_dir = out_dir / f"motifcompendium_{head}_cluster_logos"
    fwd_dir = logo_dir / "fwd"
    rev_dir = logo_dir / "rev"
    fwd_dir.mkdir(parents=True, exist_ok=True)
    rev_dir.mkdir(parents=True, exist_ok=True)

    motifs = mc.get_standard_motif_stack()
    rev_motifs = utils_motif.reverse_complement(motifs)
    source_clusters = mc.metadata["source_cluster"].tolist()

    records = []
    fwd_paths = []
    rev_paths = []
    for rank, cluster_id in enumerate(source_clusters, start=1):
        stem = f"rank_{rank:04d}_cluster_{int(cluster_id):04d}"
        fwd_path = fwd_dir / f"{stem}_fwd.svg"
        rev_path = rev_dir / f"{stem}_rev.svg"
        fwd_paths.append(fwd_path)
        rev_paths.append(rev_path)
        records.append(
            {
                "cluster_final": cluster_id,
                "logo_fwd_svg": str(fwd_path.relative_to(out_dir)),
                "logo_rev_svg": str(rev_path.relative_to(out_dir)),
            }
        )

    # plot_motifs() saves via motif_logo.ax.figure, but the underlying
    # LogoPlottingInput.ax is never populated when plotting starts from ax=None
    # (MotifCompendium's plot_many_motif_logos()/_plot_motif_logo() only plots
    # onto a locally-created Axes and never writes it back), so batched saving
    # raises AttributeError. plot_motif() saves via its own local fig instead,
    # so save per-motif rather than batching through plot_motifs().
    for start in range(0, len(motifs), batch_size):
        stop = min(start + batch_size, len(motifs))
        for motif, path in zip(motifs[start:stop], fwd_paths[start:stop]):
            utils_plotting.plot_motif(motif, trim=logo_trimming, save_loc=str(path))
        for motif, path in zip(rev_motifs[start:stop], rev_paths[start:stop]):
            utils_plotting.plot_motif(motif, trim=logo_trimming, save_loc=str(path))
        plt.close("all")

    logo_paths = pd.DataFrame.from_records(records)
    logo_paths.to_csv(
        out_dir / f"motifcompendium_{head}_cluster_logo_paths.tsv",
        sep="\t",
        index=False,
    )
    print(f"{head}: cluster SVG logos saved to {logo_dir}")
    return logo_paths


def cluster_summary_html(cluster_metadata):
    display_cols = [
        "cluster_final",
        "posneg",
        "n_motifs",
        "total_seqlets",
        "n_experiments",
        "jaspar_name",
        "jaspar_score",
        "logo_fwd_svg",
        "logo_rev_svg",
    ]
    display_cols = [c for c in display_cols if c in cluster_metadata.columns]

    rows = []
    for _, row in cluster_metadata[display_cols].iterrows():
        cells = []
        for col in display_cols:
            value = row[col]
            if col == "jaspar_score" and pd.notna(value):
                value = f"{value:.3f}"
            elif pd.isna(value):
                value = ""
            if col.startswith("logo_") and value:
                escaped_value = html.escape(str(value))
                cells.append(
                    f'<td><a href="{escaped_value}">{escaped_value}</a></td>'
                )
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    header = "".join(f"<th>{html.escape(col)}</th>" for col in display_cols)
    return f"""
<section>
  <h1>Cluster Summary</h1>
  <p>All motifs are retained. <code>total_seqlets</code> is the sum of
  MotifCompendium <code>num_seqlets</code> over all member motifs in each
  final cluster.</p>
  <table border="1" cellspacing="0" cellpadding="4">
    <thead><tr>{header}</tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>
"""


def inject_cluster_summary(report_path, title, cluster_metadata):
    report_html = report_path.read_text()
    summary_html = cluster_summary_html(cluster_metadata)
    title_html = f"<title>{html.escape(title)}</title>"

    if "<title>" not in report_html:
        report_html = report_html.replace("<head>", f"<head>{title_html}", 1)
    if "<body>" in report_html:
        report_html = report_html.replace("<body>", f"<body>\n{summary_html}", 1)
    else:
        report_html = summary_html + report_html

    report_path.write_text(report_html)


def write_cluster_summary_html(cluster_metadata, report_path, title):
    report_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
</head>
<body>
  {cluster_summary_html(cluster_metadata)}
</body>
</html>
"""
    report_path.write_text(report_html)


def process_head(
    head,
    h5_paths,
    within_threshold,
    across_threshold,
    logo_report_top_n,
    per_cluster_html,
    export_svg_logos,
    svg_logo_batch_size,
    out_dir=MC_DIR,
    algorithm=None,
    kmeans_iterations=None,
):
    if not h5_paths:
        print(f"{head}: no modisco h5 files found, skipping")
        return

    print(f"{head}: building MotifCompendium from {len(h5_paths)} h5 files")
    mc = MotifCompendium.build_from_modisco(h5_paths)
    mc.save(str(out_dir / f"motifcompendium_{head}_all_raw.mc"))

    utils_analysis.plot_similarity_distribution(
        mc, str(out_dir / f"motifcompendium_{head}_similarity_distribution.html")
    )

    assign_jaspar_labels(mc)

    algorithm, algorithm_kwargs, algorithm_label = resolve_algorithm(
        mc, algorithm, kmeans_iterations
    )
    print(f"{head}: clustering with {algorithm_label} "
          f"(MotifCompendium {mc_version()}, "
          f"cluster-average frame {cluster_reference(mc)})")

    cluster_with(
        mc, algorithm, algorithm_kwargs,
        similarity_threshold=within_threshold,
        save_name="cluster_within_model",
        cluster_within="model",
    )
    weighted_cluster_on(
        mc,
        similarity_threshold=across_threshold,
        save_name="cluster_final",
        cluster_on="cluster_within_model",
        weight_col="num_seqlets",
        algorithm=algorithm,
        algorithm_kwargs=algorithm_kwargs,
    )
    mc.save(str(out_dir / f"motifcompendium_{head}_all_clustered.mc"))

    n_final = max(mc["cluster_final"]) + 1
    print(
        f"{head}: {n_final} final clusters "
        f"(within_threshold={within_threshold}, across_threshold={across_threshold})"
    )

    export_pattern_to_cluster_mapping(mc, head, out_dir=out_dir)

    utils_analysis.export_compendium_clustered_modisco(
        mc,
        "cluster_final",
        str(out_dir / f"motifcompendium_{head}_cluster_averages.h5"),
        weight_col="num_seqlets",
    )
    mc_avg = cluster_average_with_metadata(mc)
    assign_jaspar_labels(mc_avg)
    ensure_forward_reverse_logos(mc_avg)
    mc_avg = mc_avg.sort("total_seqlets", ascending=False)
    utils_analysis.export_compendium_meme(
        mc_avg,
        str(out_dir / f"motifcompendium_{head}_cluster_averages.meme"),
    )

    logo_paths = None
    if export_svg_logos:
        logo_paths = export_cluster_logo_svgs(
            mc_avg,
            head,
            batch_size=svg_logo_batch_size,
            out_dir=out_dir,
        )

    cluster_metadata = write_cluster_metadata(
        mc, head, logo_paths=logo_paths, out_dir=out_dir,
        provenance={
            "mc_version": mc_version(),
            "cluster_algorithm": algorithm_label,
            "cluster_reference": cluster_reference(mc),
            "within_threshold": within_threshold,
            "across_threshold": across_threshold,
        },
    )

    if logo_report_top_n > 0:
        mc_avg_report = mc_avg[:logo_report_top_n]
    else:
        mc_avg_report = mc_avg

    report_path = out_dir / f"motifcompendium_{head}_cluster_report.html"
    report_columns = [
        "name",
        "source_cluster",
        "total_seqlets",
        "n_motifs",
        "n_experiments",
        "posneg",
        "JASPAR_name0",
        "JASPAR_score0",
    ]
    report_columns = [c for c in report_columns if c in mc_avg_report.columns()]
    mc_avg_report.summary_table_html(str(report_path), columns=report_columns)
    summary_path = out_dir / f"motifcompendium_{head}_cluster_summary.html"
    write_cluster_summary_html(
        cluster_metadata,
        summary_path,
        f"MotifCompendium {head} cluster summary, all motifs retained",
    )
    print(f"{head}: cluster report saved to {report_path}")
    if logo_report_top_n > 0:
        print(
            f"{head}: logo report limited to top {logo_report_top_n} clusters "
            "by total_seqlets"
        )
    print(f"{head}: cluster summary saved to {summary_path}")

    if not per_cluster_html:
        print(f"{head}: per-cluster HTML reports skipped")
        return

    cluster_html_dir = out_dir / f"motifcompendium_{head}_clusters"
    cluster_html_dir.mkdir(exist_ok=True)
    cluster_metadata_by_id = cluster_metadata.set_index("cluster_final")
    for posneg in ("pos", "neg"):
        mc_pn = mc[mc["posneg"] == posneg]
        for cluster_id in sorted(set(mc_pn["cluster_final"])):
            mc_cluster = mc_pn[mc_pn["cluster_final"] == cluster_id]
            cluster_path = cluster_html_dir / f"{posneg}_cluster_{cluster_id:04d}.html"
            mc_cluster.motif_collection_html(str(cluster_path), "cluster_final")
            inject_cluster_summary(
                cluster_path,
                f"{head} {posneg} cluster {cluster_id}",
                cluster_metadata_by_id.loc[[cluster_id]].reset_index(),
            )
    print(f"{head}: per-cluster HTML reports saved to {cluster_html_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster all modisco motifs across experiments using MotifCompendium "
            "without motif quality filtering."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MC_DIR,
        metavar="DIR",
        help=(
            "directory for all outputs (default: motifcompendium/bpnet/). Use "
            "this to build a variant compendium without overwriting the "
            "existing one -- the .mc save files alone run to hundreds of MB "
            "per head, so copying the directory first is not a practical "
            "alternative. Note cluster_final ids are not stable across runs, "
            "so a variant compendium needs its own hitcall/launch_link.py "
            "pass before its hits are comparable."
        ),
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="Minimum total reads to include an experiment (default: 0, disabled)",
    )
    parser.add_argument(
        "--blacklist",
        nargs="+",
        default=["ENCSR973QQI"],
        metavar="EXP_ID",
        help="Experiment IDs to exclude",
    )
    parser.add_argument(
        "--head",
        choices=["count", "profile"],
        action="append",
        help="Modisco head to cluster. Repeat to run both. Default: count and profile.",
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "clustering algorithm(s) for both the within- and across-model "
            "stages, passed to mc.cluster. Default: whatever the installed "
            "MotifCompendium defaults to, which is NOT stable across "
            "releases -- v1.0.18 used 'cpm_leiden' alone and v1.0.19 changed "
            "the default to 'cpm_leiden k_centroids', adding an uncapped "
            "k-means refinement. That second stage is ~4 min at count-head "
            "scale and left a profile-head build running 85 h. Pass "
            "'--algorithm cpm_leiden' to restore the pre-v1.0.19 behaviour. "
            "The algorithm actually used is recorded in cluster_metadata.tsv."
        ),
    )
    parser.add_argument(
        "--kmeans-iterations",
        type=int,
        default=None,
        metavar="N",
        help=(
            "bound the k-means refinement instead of removing it. "
            "MotifCompendium's k_centroids defaults to n_iterations=-1 and "
            "exits only when the membership vector is exactly equal between "
            "consecutive iterations, so a 2-cycle oscillation never "
            "terminates. Requires MotifCompendium >= 1.0.19 (algorithm_kwargs "
            "did not exist before it) and an algorithm list containing a "
            "k-means step."
        ),
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
        default=0.90,
        help="Similarity threshold for cross-experiment clustering (default: 0.90)",
    )
    parser.add_argument(
        "--max-cpus",
        type=int,
        default=4,
        help="Maximum CPUs for MotifCompendium (default: 4)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU use in MotifCompendium compute options",
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=1152,
        help="MotifCompendium max_chunk compute option (default: 1152)",
    )
    parser.add_argument(
        "--logo-report-top-n",
        type=int,
        default=500,
        help=(
            "Number of highest-seqlet clusters to include in the logo-heavy HTML "
            "report. Set to 0 to include all clusters (default: 500)."
        ),
    )
    parser.add_argument(
        "--per-cluster-html",
        action="store_true",
        help=(
            "Also write per-cluster motif collection HTML files. These embed many "
            "logos and can be very large, so they are disabled by default."
        ),
    )
    parser.add_argument(
        "--skip-svg-logos",
        action="store_true",
        help="Do not export per-cluster forward/reverse SVG logo files.",
    )
    parser.add_argument(
        "--svg-logo-batch-size",
        type=int,
        default=100,
        help="Number of cluster logos to render per SVG export batch (default: 100).",
    )
    args = parser.parse_args()
    if args.svg_logo_batch_size < 1:
        parser.error("--svg-logo-batch-size must be at least 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    MotifCompendium.set_compute_options(
        max_cpus=args.max_cpus,
        use_gpu=not args.no_gpu,
        max_chunk=args.max_chunk,
        progress_bar=True,
    )

    heads = args.head or ["count", "profile"]
    experiments = load_experiments(args.min_reads, set(args.blacklist))
    print(f"Using {len(experiments)} experiments after experiment-level filtering")

    for head in heads:
        h5_paths = collect_modisco_paths(experiments, head)
        process_head(
            head,
            h5_paths,
            args.within_threshold,
            args.across_threshold,
            args.logo_report_top_n,
            args.per_cluster_html,
            not args.skip_svg_logos,
            args.svg_logo_batch_size,
            out_dir=args.out_dir,
            algorithm=args.algorithm,
            kmeans_iterations=args.kmeans_iterations,
        )


if __name__ == "__main__":
    main()
