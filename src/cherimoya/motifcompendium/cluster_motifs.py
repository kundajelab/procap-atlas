"""Cluster all Cherimoya modisco motifs across experiments using MotifCompendium.

Cherimoya-specific wrapper around the shared clustering logic in
src/bpnet/motifcompendium/cluster_motifs.py. The only differences from the
bpnet build are the input MoDISco directory (modisco/cherimoya/) and the
output directory (motifcompendium/cherimoya/).

Usage:
    python src/cherimoya/motifcompendium/cluster_motifs.py
    python src/cherimoya/motifcompendium/cluster_motifs.py --head count
    python src/cherimoya/motifcompendium/cluster_motifs.py --head count --head profile
    python src/cherimoya/motifcompendium/cluster_motifs.py --within-threshold 0.95 --across-threshold 0.90
    python src/cherimoya/motifcompendium/cluster_motifs.py --force-merge-threshold 0.93
    python src/cherimoya/motifcompendium/cluster_motifs.py --from-mc motifcompendium/cherimoya/motifcompendium_count_all_clustered.mc --head count --force-merge-threshold 0.93
    python src/cherimoya/motifcompendium/cluster_motifs.py --out-dir motifcompendium/cherimoya_variant/ --head count
"""

import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODISCO_DIR = REPO_ROOT / "modisco" / "cherimoya"
MC_DIR = REPO_ROOT / "motifcompendium" / "cherimoya"

_BPNET_MC_PATH = (
    REPO_ROOT / "src" / "bpnet" / "motifcompendium" / "cluster_motifs.py"
)
_spec = importlib.util.spec_from_file_location(
    "_bpnet_cluster_motifs", _BPNET_MC_PATH
)
_bpnet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bpnet)

load_experiments = _bpnet.load_experiments
process_head = _bpnet.process_head
process_from_mc = _bpnet.process_from_mc
parse_algorithm_kwargs = _bpnet.parse_algorithm_kwargs
MotifCompendium = _bpnet.MotifCompendium


def collect_modisco_paths(experiments, head):
    h5_paths = {}
    for exp_id in experiments:
        h5_path = MODISCO_DIR / f"{exp_id}_{head}.modisco.h5"
        if h5_path.exists():
            h5_paths[exp_id] = str(h5_path)
    return h5_paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster all Cherimoya modisco motifs across experiments using "
            "MotifCompendium without motif quality filtering."
        )
    )
    parser.add_argument(
        "--from-mc",
        type=Path,
        default=None,
        metavar="MC_FILE",
        help=(
            "load a previously clustered .mc file and apply force-merge + "
            "re-export, skipping the build-from-h5 and initial clustering "
            "steps. Requires --head (exactly one) and "
            "--force-merge-threshold."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MC_DIR,
        metavar="DIR",
        help=f"directory for all outputs (default: {MC_DIR.relative_to(REPO_ROOT)}/)",
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
        help="clustering algorithm(s) for mc.cluster (default: installed MotifCompendium default)",
    )
    parser.add_argument(
        "--algorithm-kwarg",
        action="append",
        default=None,
        metavar="ALGORITHM.KEY=VALUE",
        dest="algorithm_kwarg",
        help="per-step clustering argument (e.g. 'k_centroids.tol=-inf')",
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
        "--force-merge-threshold",
        type=float,
        default=None,
        metavar="THRESH",
        help="similarity threshold for post-clustering centroid merging (default: disabled)",
    )
    parser.add_argument(
        "--force-merge-density",
        type=float,
        default=1.0,
        help="DCC density for centroid merging (default: 1.0)",
    )
    parser.add_argument(
        "--force-merge-max-iter",
        type=int,
        default=50,
        help="Maximum iterations for centroid merging convergence (default: 50)",
    )
    parser.add_argument(
        "--force-merge-seed",
        type=int,
        default=100,
        help="Random seed for DCC in centroid merging (default: 100)",
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
        help="Number of highest-seqlet clusters to include in the logo HTML report (default: 500)",
    )
    parser.add_argument(
        "--per-cluster-html",
        action="store_true",
        help="Also write per-cluster motif collection HTML files",
    )
    parser.add_argument(
        "--skip-svg-logos",
        action="store_true",
        help="Do not export per-cluster forward/reverse SVG logo files",
    )
    parser.add_argument(
        "--svg-logo-batch-size",
        type=int,
        default=100,
        help="Number of cluster logos to render per SVG export batch (default: 100)",
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

    if args.from_mc:
        if len(heads) != 1:
            parser.error("--from-mc requires exactly one --head")
        if args.force_merge_threshold is None:
            parser.error("--force-merge-threshold is required with --from-mc")
        process_from_mc(
            args.from_mc,
            heads[0],
            args.force_merge_threshold,
            args.force_merge_density,
            args.force_merge_max_iter,
            args.force_merge_seed,
            args.logo_report_top_n,
            args.per_cluster_html,
            not args.skip_svg_logos,
            args.svg_logo_batch_size,
            out_dir=args.out_dir,
        )
        return

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
            algorithm_kwargs=parse_algorithm_kwargs(args.algorithm_kwarg),
            force_merge_threshold=args.force_merge_threshold,
            force_merge_density=args.force_merge_density,
            force_merge_max_iter=args.force_merge_max_iter,
            force_merge_seed=args.force_merge_seed,
        )


if __name__ == "__main__":
    main()
