import argparse
import html
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import MotifCompendium
import MotifCompendium.utils.analysis as utils_analysis
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
MODISCO_DIR = REPO_ROOT / "modisco" / "bpnet"
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet_all_motifs"
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


def weighted_cluster_on(mc, similarity_threshold, save_name, cluster_on, weight_col):
    cluster_kwargs = {
        "similarity_threshold": similarity_threshold,
        "save_name": save_name,
        "cluster_on": cluster_on,
    }
    if "weight_col" in inspect.signature(mc.cluster).parameters:
        cluster_kwargs["weight_col"] = weight_col
    else:
        cluster_kwargs["cluster_on_weight"] = weight_col
    mc.cluster(**cluster_kwargs)


def write_cluster_metadata(mc, head):
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
        .sort_values("cluster_final")
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

    metadata_path = MC_DIR / f"motifcompendium_{head}_cluster_metadata.tsv"
    agg.to_csv(metadata_path, sep="\t", index=False)
    print(f"{head}: cluster metadata saved to {metadata_path}")
    return agg


def cluster_summary_html(cluster_metadata):
    display_cols = [
        "cluster_final",
        "posneg",
        "n_motifs",
        "total_seqlets",
        "n_experiments",
        "jaspar_name",
        "jaspar_score",
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


def write_cluster_metadata_pdf(pdf_path, title, cluster_metadata):
    display_cols = [
        "cluster_final",
        "posneg",
        "n_motifs",
        "total_seqlets",
        "n_experiments",
        "jaspar_name",
        "jaspar_score",
    ]
    display_cols = [c for c in display_cols if c in cluster_metadata.columns]
    table = cluster_metadata[display_cols].copy()
    if "jaspar_score" in table.columns:
        table["jaspar_score"] = table["jaspar_score"].map(
            lambda x: "" if pd.isna(x) else f"{x:.3f}"
        )

    rows_per_page = 35
    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(table), rows_per_page):
            page = table.iloc[start : start + rows_per_page]
            fig_height = max(4, 0.25 * (len(page) + 5))
            fig, ax = plt.subplots(figsize=(11, fig_height))
            ax.axis("off")
            ax.set_title(title, loc="left", fontsize=14, pad=12)
            ax.text(
                0,
                0.96,
                "All motifs retained. total_seqlets is summed across motifs in each final cluster.",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
            )
            tbl = ax.table(
                cellText=page.astype(str).values,
                colLabels=display_cols,
                cellLoc="left",
                colLoc="left",
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.2)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  PDF cluster report saved to {pdf_path}")


def write_cluster_report_pdf(report_path, pdf_path, title, cluster_metadata):
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(report_path.resolve().as_uri(), wait_until="networkidle")
            page.pdf(path=str(pdf_path), format="Letter", print_background=True)
            browser.close()
        print(f"  PDF cluster report saved to {pdf_path}")
    except Exception as exc:
        print(f"  HTML-to-PDF render failed ({exc}); writing metadata PDF instead")
        write_cluster_metadata_pdf(pdf_path, title, cluster_metadata)


def process_head(head, h5_paths, within_threshold, across_threshold):
    if not h5_paths:
        print(f"{head}: no modisco h5 files found, skipping")
        return

    print(f"{head}: building MotifCompendium from {len(h5_paths)} h5 files")
    mc = MotifCompendium.build_from_modisco(h5_paths)
    mc.save(str(MC_DIR / f"motifcompendium_{head}_all_raw.mc"))

    utils_analysis.plot_similarity_distribution(
        mc, str(MC_DIR / f"motifcompendium_{head}_similarity_distribution.html")
    )

    assign_jaspar_labels(mc)

    mc.cluster(
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
    )
    mc.save(str(MC_DIR / f"motifcompendium_{head}_all_clustered.mc"))

    n_final = max(mc["cluster_final"]) + 1
    print(
        f"{head}: {n_final} final clusters "
        f"(within_threshold={within_threshold}, across_threshold={across_threshold})"
    )

    utils_analysis.export_compendium_clustered_modisco(
        mc,
        "cluster_final",
        str(MC_DIR / f"motifcompendium_{head}_cluster_averages.h5"),
        weight_col="num_seqlets",
    )
    mc_avg = mc.cluster_averages("cluster_final")
    assign_jaspar_labels(mc_avg)
    utils_analysis.export_compendium_meme(
        mc_avg,
        str(MC_DIR / f"motifcompendium_{head}_cluster_averages.meme"),
    )

    cluster_metadata = write_cluster_metadata(mc, head)

    report_path = MC_DIR / f"motifcompendium_{head}_cluster_report.html"
    mc_avg.summary_table_html(str(report_path))
    inject_cluster_summary(
        report_path,
        f"MotifCompendium {head} cluster report, all motifs retained",
        cluster_metadata,
    )
    write_cluster_report_pdf(
        report_path,
        MC_DIR / f"motifcompendium_{head}_cluster_report.pdf",
        f"MotifCompendium {head} cluster report, all motifs retained",
        cluster_metadata,
    )
    print(f"{head}: cluster report saved to {report_path}")

    cluster_html_dir = MC_DIR / f"motifcompendium_{head}_clusters"
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
        "--head",
        choices=["count", "profile"],
        action="append",
        help="Modisco head to cluster. Repeat to run both. Default: count and profile.",
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
    args = parser.parse_args()

    MC_DIR.mkdir(parents=True, exist_ok=True)

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
        process_head(head, h5_paths, args.within_threshold, args.across_threshold)


if __name__ == "__main__":
    main()
