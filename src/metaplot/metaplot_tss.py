#!/usr/bin/env python3
"""
Metaplot of + and - strand PRO-cap signal around gene TSSs for each experiment.

Parses the GENCODE GFF3 annotation to extract gene TSS positions, then fetches
strand-specific BigWig signal in a window around each TSS. Windows are oriented
by gene strand so that the sense direction always points right (downstream into
the gene body). The averaged sense and antisense profiles are plotted as a
butterfly/mirror plot and/or as per-TSS heatmaps.

Signal is normalized to RPM using total read counts from configs/n_reads.txt.

Usage:
    python src/metaplot/metaplot_tss.py
    python src/metaplot/metaplot_tss.py --window 1000 --bin-size 5
    python src/metaplot/metaplot_tss.py --experiment ENCSR882DWM
    python src/metaplot/metaplot_tss.py --feature transcript
    python src/metaplot/metaplot_tss.py --plot-type heatmap --max-tss 5000
    python src/metaplot/metaplot_tss.py --plot-type both
"""

import argparse
import gzip
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybigtools
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
DEFAULT_ANNOTATION = REPO_ROOT / "data" / "gencode.v49.annotation.gff3.gz"
CANONICAL_CHROMS = {f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]}


def parse_tss(gff3_path: Path, feature: str = "gene") -> list[tuple[str, int, str]]:
    """Parse TSSs for the given feature type from a GENCODE GFF3 file.

    Returns a list of (chrom, tss_pos, strand) where tss_pos is 0-based.
    TSS = start for + strand genes, end-1 for - strand genes.
    Only canonical chromosomes (chr1–22, X, Y) are included.
    """
    tss_list = []
    opener = gzip.open if str(gff3_path).endswith(".gz") else open
    with opener(gff3_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != feature:
                continue
            chrom = fields[0]
            if chrom not in CANONICAL_CHROMS:
                continue
            start = int(fields[3]) - 1  # GFF3 is 1-based inclusive -> 0-based
            end = int(fields[4])  # GFF3 end is 1-based inclusive -> 0-based exclusive
            strand = fields[6]
            tss = start if strand == "+" else end - 1
            tss_list.append((chrom, tss, strand))
    return tss_list


def collect_windows(
    pl_path: Path,
    mn_path: Path,
    tss_list: list[tuple[str, int, str]],
    window: int,
    bin_size: int,
    total_reads: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract strand-oriented, RPM-normalized signal around every TSS.

    For each TSS a window of [-window, +window) bp is extracted from both
    BigWigs. Windows are flipped for - strand genes so that position 0 is
    always the TSS and positive positions are downstream into the gene body.

    After orientation:
      sense     = pl signal for + strand genes; |mn| for - strand genes
      antisense = |mn| for + strand genes;      pl  for - strand genes

    Signal is normalized to RPM: divided by (total_reads / 1e6).

    Returns (sense, antisense) each of shape (n_tss, n_bins) where
    n_bins = 2 * window // bin_size. TSSs that fall within window bp of a
    chromosome boundary are excluded.
    """
    n_bins = (2 * window) // bin_size
    rpm_scale = 1e6 / total_reads

    sense_rows = []
    antisense_rows = []

    with pybigtools.open(str(pl_path)) as pl_bw, pybigtools.open(str(mn_path)) as mn_bw:
        chrom_sizes = pl_bw.chroms()

        for chrom, tss, strand in tss_list:
            if chrom not in chrom_sizes:
                continue
            left = tss - window
            right = tss + window
            if left < 0 or right > chrom_sizes[chrom]:
                continue

            pl_sig = np.zeros(2 * window)
            mn_sig = np.zeros(2 * window)

            for s, e, v in pl_bw.records(chrom, left, right):
                pl_sig[s - left : e - left] = v
            for s, e, v in mn_bw.records(chrom, left, right):
                mn_sig[s - left : e - left] = abs(v)

            if strand == "-":
                pl_sig = pl_sig[::-1]
                mn_sig = mn_sig[::-1]
                sense = mn_sig
                antisense = pl_sig
            else:
                sense = pl_sig
                antisense = mn_sig

            sense_bin = (
                sense[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
            )
            antisense_bin = (
                antisense[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
            )
            sense_rows.append(sense_bin)
            antisense_rows.append(antisense_bin)

    if not sense_rows:
        empty = np.zeros((0, n_bins))
        return empty, empty

    sense_mat = np.array(sense_rows) * rpm_scale
    antisense_mat = np.array(antisense_rows) * rpm_scale
    return sense_mat, antisense_mat


def plot_metaplot(
    sense: np.ndarray,
    antisense: np.ndarray,
    window: int,
    bin_size: int,
    n_tss: int,
    title: str,
    out_path: Path,
) -> None:
    """Save a butterfly metaplot with mean sense above and antisense below the x-axis."""
    n_bins = 2 * window // bin_size
    positions = np.linspace(-window, window, n_bins, endpoint=False) + bin_size / 2

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.fill_between(positions, sense, color="#e04b4b", alpha=0.8, label="Sense")
    ax.fill_between(
        positions, -antisense, color="#4b7be0", alpha=0.8, label="Antisense"
    )
    ax.plot(positions, sense, color="#c02020", linewidth=0.8)
    ax.plot(positions, -antisense, color="#2050c0", linewidth=0.8)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.6)

    ax.set_xlabel("Position relative to TSS (bp)")
    ax.set_ylabel("Mean signal (RPM)")
    ax.set_title(f"{title}\nn = {n_tss:,} TSSs")
    ax.legend(frameon=False, loc="upper right")

    ymax = max(sense.max(), antisense.max()) * 1.15
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(-window, window)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(
    sense_mat: np.ndarray,
    antisense_mat: np.ndarray,
    window: int,
    bin_size: int,
    max_tss: int,
    title: str,
    out_path: Path,
) -> None:
    """Save per-TSS heatmaps of sense (top) and antisense (bottom) signal.

    Rows are sorted by total sense signal in the central 20% of the window
    (i.e., ±window/10 around TSS). If more than max_tss TSSs are available,
    the top max_tss by that score are shown.

    Color scale runs from 0 to the 99th percentile of signal across both strands.
    """
    n_bins = sense_mat.shape[1]
    center = n_bins // 2
    flank = max(1, n_bins // 10)  # ±10% of window

    sort_scores = sense_mat[:, center - flank : center + flank].sum(axis=1)
    order = np.argsort(sort_scores)[::-1]
    if max_tss < len(order):
        order = order[:max_tss]
    # Re-sort selected rows by score for display (highest at top)
    sense_show = sense_mat[order]
    antisense_show = antisense_mat[order]

    vmax = np.percentile(np.concatenate([sense_show, antisense_show]), 99)
    vmax = max(vmax, 1e-6)  # avoid all-zero color scale

    xtick_bins = np.linspace(0, n_bins - 1, 5)
    xtick_labels = [f"{int(x):+d}" for x in np.linspace(-window, window, 5)]

    n_shown = len(sense_show)
    height_per_row = 0.003  # inches per row
    panel_height = max(2.0, n_shown * height_per_row)
    fig_height = panel_height * 2 + 1.5  # two panels + margins

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6, fig_height),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35},
    )

    for ax, mat, label, cmap in zip(
        axes,
        [sense_show, antisense_show],
        ["Sense", "Antisense"],
        ["Reds", "Blues"],
    ):
        im = ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            origin="upper",
        )
        ax.axvline(n_bins / 2, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(xtick_bins)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Position relative to TSS (bp)")
        ax.set_ylabel("TSS (ranked)")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, label="RPM", fraction=0.03, pad=0.02)

    fig.suptitle(f"{title}\nn = {n_shown:,} TSSs shown", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_n_reads(path: Path) -> dict[str, float]:
    """Load total read counts from n_reads.txt."""
    n_reads = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                n_reads[parts[0]] = float(parts[4])
    return n_reads


def main():
    parser = argparse.ArgumentParser(
        description="Metaplot PRO-cap signal around gene TSSs"
    )
    parser.add_argument(
        "--annotation",
        metavar="GFF3",
        default=DEFAULT_ANNOTATION,
        type=Path,
        help=f"GENCODE GFF3 annotation file (default: {DEFAULT_ANNOTATION.name})",
    )
    parser.add_argument(
        "--feature",
        default="gene",
        help="GFF3 feature type to use as TSS source (default: gene)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        metavar="BP",
        help="half-window size in bp around each TSS (default: 100)",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=1,
        metavar="BP",
        help="bin size in bp for averaging signal (default: 1)",
    )
    parser.add_argument(
        "--plot-type",
        choices=["metaplot", "heatmap", "both"],
        default="metaplot",
        help="output plot type (default: metaplot)",
    )
    parser.add_argument(
        "--max-tss",
        type=int,
        default=5000,
        metavar="N",
        help="maximum number of TSSs to show in heatmap, ranked by signal (default: 5000)",
    )
    parser.add_argument(
        "--experiment",
        metavar="EXP_ID",
        help="run only this experiment (default: all experiments)",
    )
    parser.add_argument(
        "--min-reads",
        type=float,
        default=0,
        metavar="N",
        help="skip experiments with fewer than N total reads (default: 0)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "figures" / "metaplots",
        metavar="DIR",
        help="output directory for PDFs (default: figures/metaplots/)",
    )
    args = parser.parse_args()

    if not args.annotation.exists():
        print(f"ERROR: annotation file not found: {args.annotation}", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    n_reads_map = load_n_reads(N_READS_PATH) if N_READS_PATH.exists() else {}

    print(
        f"Parsing {args.feature} TSSs from {args.annotation.name}...", file=sys.stderr
    )
    tss_list = parse_tss(args.annotation, feature=args.feature)
    print(f"Found {len(tss_list):,} TSSs on canonical chromosomes.", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    experiments = config["experiments"]
    if args.experiment:
        if args.experiment not in experiments:
            print(
                f"ERROR: experiment {args.experiment!r} not in config", file=sys.stderr
            )
            sys.exit(1)
        experiments = {args.experiment: experiments[args.experiment]}

    for exp_id, exp in tqdm(experiments.items(), unit="exp"):
        biosample = re.sub(r"[^\w-]", "_", exp.get("biosample", "unknown")).strip("_")
        total_reads = n_reads_map.get(exp_id)

        if total_reads is None:
            print(
                f"WARNING: {exp_id}: no read count found in n_reads.txt, skipping",
                file=sys.stderr,
            )
            continue
        if total_reads < args.min_reads:
            continue

        processed = exp.get("processed", {})
        pl_path = REPO_ROOT / processed["pl_bigwig"]
        mn_path = REPO_ROOT / processed["mn_bigwig"]

        missing = [p for p in (pl_path, mn_path) if not p.exists()]
        if missing:
            print(
                f"WARNING: {exp_id}: missing {[p.name for p in missing]}, skipping",
                file=sys.stderr,
            )
            continue

        sense_mat, antisense_mat = collect_windows(
            pl_path, mn_path, tss_list, args.window, args.bin_size, total_reads
        )
        n_tss = len(sense_mat)
        if n_tss == 0:
            print(f"WARNING: {exp_id}: no valid TSS windows, skipping", file=sys.stderr)
            continue

        title = f"{exp_id} — {biosample}"

        if args.plot_type in ("metaplot", "both"):
            out_path = args.out_dir / f"{exp_id}_{biosample}_metaplot.pdf"
            plot_metaplot(
                sense_mat.mean(axis=0),
                antisense_mat.mean(axis=0),
                args.window,
                args.bin_size,
                n_tss,
                title,
                out_path,
            )

        if args.plot_type in ("heatmap", "both"):
            out_path = args.out_dir / f"{exp_id}_{biosample}_heatmap.pdf"
            plot_heatmap(
                sense_mat,
                antisense_mat,
                args.window,
                args.bin_size,
                args.max_tss,
                title,
                out_path,
            )

    print(f"\nPlots written to {args.out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
