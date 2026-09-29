#!/usr/bin/env python3
"""
Motif-centered avg. measured/predicted profile panels, in the style of ProCapNet
Fig. 2a, for the top clusters produced by cluster_motifs_all.py.

For each of the top --top-n clusters per head (ranked by number of experiments the
cluster was found in), recovers every seqlet instance's genomic position directly
from the raw modisco h5 files, pulls a measured (RPM-normalized) and a predicted
signal window centered on each instance (oriented by the seqlet's own strand),
averages within each experiment and then across experiments, and writes one
same-sized, title-free SVG panel per (cluster, column) so the panels can be
manually arranged into a figure in a vector editor. Also reuses the forward CWM
logo SVGs already exported by cluster_motifs_all.py and scales a "weight dot"
panel by the cluster's number of experiments.

Must be run after cluster_motifs_all.py for the same head(s).

Usage:
    python src/bpnet/motifcompendium/motif_centered_metaplot.py
    python src/bpnet/motifcompendium/motif_centered_metaplot.py --head count --top-n 20
    python src/bpnet/motifcompendium/motif_centered_metaplot.py --window 200 --bin-size 2
"""

import argparse
import re
import shutil
import sys
from itertools import chain
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import MotifCompendium
import numpy as np
import pandas as pd
import pybigtools
import yaml
from tangermeme.io import extract_loci
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
MODISCO_DIR = REPO_ROOT / "modisco" / "bpnet"
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet_all_motifs"
FASTA = REPO_ROOT / "data" / "hg38.fa"
BLACKLIST = REPO_ROOT / "data" / "hg38.blacklist.bed.gz"
PRED_BIGWIG_DIR = REPO_ROOT / "predictions" / "bpnet" / "bigwigs"
DEFAULT_OUT_DIR = REPO_ROOT / "figures" / "motif_metaplots"

# Attribution/OHE input window (attribute_bpnet.py, save_ohe.py) and the
# tfmodisco-lite `-w` window (src/bpnet/modisco/), both centered on the same
# peak midpoint.
IN_WINDOW = 2114
MODISCO_WINDOW = 1000

PATTERN_IDX_RE = re.compile(r"pattern_(\d+)$")

_warned_missing = set()


def pattern_index_from_name(name):
    match = PATTERN_IDX_RE.search(name)
    if not match:
        raise ValueError(f"could not parse a pattern index from metadata name {name!r}")
    return int(match.group(1))


def get_pattern_seqlets(h5_path, posneg, pattern_idx):
    group_name = f"{posneg}_patterns"
    pattern_name = f"pattern_{pattern_idx}"
    with h5py.File(h5_path, "r") as f:
        if group_name not in f:
            raise KeyError(f"{h5_path}: no group {group_name!r} (top-level keys: {list(f.keys())})")
        grp = f[group_name]
        if pattern_name not in grp:
            raise KeyError(
                f"{h5_path}:{group_name}: no {pattern_name!r} (available: {list(grp.keys())})"
            )
        seqlets = grp[pattern_name]["seqlets"]
        starts = seqlets["start"][:]
        ends = seqlets["end"][:]
        example_idx = seqlets["example_idx"][:]
        is_revcomp = seqlets["is_revcomp"][:].astype(bool)
    return starts, ends, example_idx, is_revcomp


def build_locus_lookup(exp_id, config, all_chroms):
    """Reproduce attribute_bpnet.py's/save_ohe.py's extract_loci() filtering for
    exp_id so example_idx in its modisco h5 can be mapped back to a genomic peak
    midpoint. Returns (chrom array, mid array) indexed by example_idx."""
    processed = config["experiments"].get(exp_id, {}).get("processed", {})
    peaks_path = REPO_ROOT / processed["filtered_peaks"]
    loci = pd.read_csv(
        peaks_path,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )
    loci = loci[loci["chrom"].isin(all_chroms)].reset_index(drop=True)
    _, mask = extract_loci(
        loci=loci,
        sequences=str(FASTA),
        chroms=all_chroms,
        in_window=IN_WINDOW,
        exclusion_lists=[str(BLACKLIST)],
        return_mask=True,
    )
    kept = loci[mask.numpy()].reset_index(drop=True)
    mid = (kept["start"] + (kept["end"] - kept["start"]) // 2).to_numpy()
    chrom = kept["chrom"].to_numpy()
    return chrom, mid


def recover_cluster_instances(mc_metadata, cluster_id, head, config, all_chroms, locus_cache):
    """Return {model: [(chrom, genomic_center, is_revcomp), ...]} for every seqlet
    backing this final cluster, re-derived from the raw modisco h5 files."""
    rows = mc_metadata[mc_metadata["cluster_final"] == cluster_id]
    instances_by_model = {}
    for _, row in rows.iterrows():
        model = row["model"]
        posneg = row["posneg"]
        pattern_idx = pattern_index_from_name(row["name"])
        h5_path = MODISCO_DIR / f"{model}_{head}.modisco.h5"
        starts, ends, example_idx, is_revcomp = get_pattern_seqlets(h5_path, posneg, pattern_idx)
        assert len(starts) == row["num_seqlets"], (
            f"{h5_path}:{posneg}_patterns/pattern_{pattern_idx}: seqlet count "
            f"{len(starts)} != metadata num_seqlets {row['num_seqlets']} "
            "(pattern index parsing from metadata name may be wrong)"
        )

        if model not in locus_cache:
            locus_cache[model] = build_locus_lookup(model, config, all_chroms)
        chrom_arr, mid_arr = locus_cache[model]

        centers = mid_arr[example_idx] - MODISCO_WINDOW // 2 + (starts + ends) // 2
        chroms = chrom_arr[example_idx]
        instances_by_model.setdefault(model, []).extend(zip(chroms, centers, is_revcomp))
    return instances_by_model


def collect_motif_windows(pl_path, mn_path, instances, window, bin_size):
    """Extract strand-oriented signal windows around every motif instance.

    Generalizes metaplot_tss.collect_windows() to center on an arbitrary list of
    (chrom, pos, is_revcomp) instances instead of a fixed per-experiment TSS list,
    orienting by the instance's own strand instead of gene strand. Returns raw
    (unnormalized) (sense, antisense) matrices of shape (n_instances, n_bins).
    """
    n_bins = (2 * window) // bin_size
    sense_rows = []
    antisense_rows = []

    with pybigtools.open(str(pl_path)) as pl_bw, pybigtools.open(str(mn_path)) as mn_bw:
        chrom_sizes = pl_bw.chroms()

        for chrom, center, is_revcomp in instances:
            if chrom not in chrom_sizes:
                continue
            left = int(center) - window
            right = int(center) + window
            if left < 0 or right > chrom_sizes[chrom]:
                continue

            pl_sig = np.zeros(2 * window)
            mn_sig = np.zeros(2 * window)
            for s, e, v in pl_bw.records(chrom, left, right):
                pl_sig[s - left : e - left] = v
            for s, e, v in mn_bw.records(chrom, left, right):
                mn_sig[s - left : e - left] = abs(v)

            if is_revcomp:
                pl_sig = pl_sig[::-1]
                mn_sig = mn_sig[::-1]
                sense, antisense = mn_sig, pl_sig
            else:
                sense, antisense = pl_sig, mn_sig

            sense_rows.append(sense[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1))
            antisense_rows.append(
                antisense[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
            )

    if not sense_rows:
        empty = np.zeros((0, n_bins))
        return empty, empty
    return np.array(sense_rows), np.array(antisense_rows)


def measured_bigwig_paths(model, config):
    processed = config["experiments"].get(model, {}).get("processed", {})
    pl, mn = processed.get("pl_bigwig"), processed.get("mn_bigwig")
    if not pl or not mn:
        return None
    pl_path, mn_path = REPO_ROOT / pl, REPO_ROOT / mn
    if not (pl_path.exists() and mn_path.exists()):
        return None
    return pl_path, mn_path


def predicted_bigwig_paths(model):
    pl_path = PRED_BIGWIG_DIR / f"{model}_pl.bigWig"
    mn_path = PRED_BIGWIG_DIR / f"{model}_mn.bigWig"
    if not (pl_path.exists() and mn_path.exists()):
        return None
    return pl_path, mn_path


def average_profile_for_cluster(instances_by_model, bw_paths_fn, rpm_scale_fn, window, bin_size, warn_label):
    """Average within each experiment, then across experiments (equal weight per
    experiment), for whichever bigwig source bw_paths_fn resolves."""
    per_model_sense = []
    per_model_antisense = []
    for model, instances in instances_by_model.items():
        paths = bw_paths_fn(model)
        if paths is None:
            warn_key = (warn_label, model)
            if warn_key not in _warned_missing:
                print(f"WARNING: {warn_label} bigwigs not found for {model}, skipping", file=sys.stderr)
                _warned_missing.add(warn_key)
            continue
        pl_path, mn_path = paths
        sense_mat, antisense_mat = collect_motif_windows(pl_path, mn_path, instances, window, bin_size)
        if len(sense_mat) == 0:
            continue
        scale = rpm_scale_fn(model) if rpm_scale_fn else 1.0
        per_model_sense.append(sense_mat.mean(axis=0) * scale)
        per_model_antisense.append(antisense_mat.mean(axis=0) * scale)

    n_bins = (2 * window) // bin_size
    if not per_model_sense:
        return np.zeros(n_bins), np.zeros(n_bins), 0
    return np.mean(per_model_sense, axis=0), np.mean(per_model_antisense, axis=0), len(per_model_sense)


def plot_profile_panel(sense, antisense, window, out_path, figsize=(2.4, 1.0)):
    n_bins = len(sense)
    positions = np.linspace(-window, window, n_bins, endpoint=False)

    fig, ax = plt.subplots(figsize=figsize)
    ymax = max(sense.max(), antisense.max(), 1e-9) * 1.15

    ax.fill_between(positions, sense, color="#1a3d7c", alpha=0.9)
    ax.fill_between(positions, -antisense, color="#7fa8e0", alpha=0.9)
    ax.plot(positions, sense, color="#0d234a", linewidth=0.6)
    ax.plot(positions, -antisense, color="#4a76b8", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(-window, window)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(out_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_weight_dot(n_experiments, vmin, vmax, out_path, size_in=0.5):
    fig, ax = plt.subplots(figsize=(size_in, size_in))
    frac = 0.0 if vmax == vmin else (n_experiments - vmin) / (vmax - vmin)
    frac = max(0.0, min(1.0, frac))
    cmap = plt.get_cmap("YlGnBu")
    radius = 0.15 + 0.35 * np.sqrt(frac)

    circle = plt.Circle((0.5, 0.5), radius, color=cmap(frac), transform=ax.transAxes)
    ax.add_patch(circle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.savefig(out_path, format="svg", transparent=True)
    plt.close(fig)


def plot_preview_grid(rows, out_path):
    n = len(rows)
    fig, axes = plt.subplots(
        n, 3, figsize=(6, max(1.2, 0.5 * n)), gridspec_kw={"width_ratios": [0.4, 1, 1]}
    )
    axes = np.atleast_2d(axes)

    for i, row in enumerate(rows):
        ax_label, ax_meas, ax_pred = axes[i]
        ax_label.text(
            0.5,
            0.5,
            f"#{row['rank']} c{row['cluster_final']}\nn={row['n_experiments']}",
            ha="center",
            va="center",
            fontsize=6,
        )
        ax_label.axis("off")
        for ax, sense, antisense, title in (
            (ax_meas, row["measured_sense"], row["measured_antisense"], "measured" if i == 0 else None),
            (ax_pred, row["pred_sense"], row["pred_antisense"], "predicted" if i == 0 else None),
        ):
            ymax = max(sense.max(), antisense.max(), 1e-9) * 1.15
            ax.fill_between(range(len(sense)), sense, color="#1a3d7c")
            ax.fill_between(range(len(antisense)), -antisense, color="#7fa8e0")
            ax.set_ylim(-ymax, ymax)
            ax.set_xticks([])
            ax.set_yticks([])
            if title:
                ax.set_title(title, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_head_metaplots(head, args, config, n_reads_map, all_chroms):
    metadata_path = MC_DIR / f"motifcompendium_{head}_cluster_metadata.tsv"
    logo_paths_path = MC_DIR / f"motifcompendium_{head}_cluster_logo_paths.tsv"
    mc_path = MC_DIR / f"motifcompendium_{head}_all_clustered.mc"
    if not (metadata_path.exists() and logo_paths_path.exists() and mc_path.exists()):
        print(f"{head}: missing cluster_motifs_all.py outputs, skipping", file=sys.stderr)
        return

    cluster_metadata = pd.read_csv(metadata_path, sep="\t")
    logo_paths = pd.read_csv(logo_paths_path, sep="\t")

    top = (
        cluster_metadata.sort_values(["n_experiments", "total_seqlets"], ascending=[False, False])
        .head(args.top_n)
        .reset_index(drop=True)
    )
    top["rank"] = np.arange(1, len(top) + 1)
    top = top.merge(logo_paths[["cluster_final", "logo_fwd_svg"]], on="cluster_final", how="left")

    mc = MotifCompendium.load(str(mc_path))
    mc_metadata = mc.metadata

    out_dir = args.out_dir / head
    dirs = {name: out_dir / name for name in ("logos_fwd", "weight_dots", "measured", "predicted")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    vmin, vmax = top["n_experiments"].min(), top["n_experiments"].max()
    locus_cache = {}
    manifest_rows = []
    preview_rows = []

    for _, row in tqdm(top.iterrows(), total=len(top), desc=f"{head} clusters"):
        cluster_id = int(row["cluster_final"])
        rank = int(row["rank"])
        stem = f"rank_{rank:04d}_cluster_{cluster_id:04d}"

        instances_by_model = recover_cluster_instances(
            mc_metadata, cluster_id, head, config, all_chroms, locus_cache
        )

        measured_sense, measured_antisense, n_meas = average_profile_for_cluster(
            instances_by_model,
            bw_paths_fn=lambda m: measured_bigwig_paths(m, config),
            rpm_scale_fn=lambda m: 1e6 / n_reads_map[m] if n_reads_map.get(m) else 1.0,
            window=args.window,
            bin_size=args.bin_size,
            warn_label="measured",
        )
        pred_sense, pred_antisense, n_pred = average_profile_for_cluster(
            instances_by_model,
            bw_paths_fn=predicted_bigwig_paths,
            rpm_scale_fn=None,
            window=args.window,
            bin_size=args.bin_size,
            warn_label="predicted",
        )

        logo_dst = dirs["logos_fwd"] / f"{stem}_logo.svg"
        logo_rel = row["logo_fwd_svg"]
        if isinstance(logo_rel, str):
            logo_src = MC_DIR / logo_rel
            if logo_src.exists():
                shutil.copyfile(logo_src, logo_dst)

        dot_dst = dirs["weight_dots"] / f"{stem}_dot.svg"
        plot_weight_dot(row["n_experiments"], vmin, vmax, dot_dst)

        measured_dst = dirs["measured"] / f"{stem}_measured.svg"
        plot_profile_panel(measured_sense, measured_antisense, args.window, measured_dst)

        predicted_dst = dirs["predicted"] / f"{stem}_predicted.svg"
        plot_profile_panel(pred_sense, pred_antisense, args.window, predicted_dst)

        manifest_rows.append(
            {
                "rank": rank,
                "cluster_final": cluster_id,
                "n_experiments": row["n_experiments"],
                "total_seqlets": row["total_seqlets"],
                "n_experiments_with_measured_signal": n_meas,
                "n_experiments_with_predicted_signal": n_pred,
                "logo_fwd_svg": str(logo_dst.relative_to(args.out_dir)),
                "weight_dot_svg": str(dot_dst.relative_to(args.out_dir)),
                "measured_profile_svg": str(measured_dst.relative_to(args.out_dir)),
                "predicted_profile_svg": str(predicted_dst.relative_to(args.out_dir)),
            }
        )
        preview_rows.append(
            {
                "rank": rank,
                "cluster_final": cluster_id,
                "n_experiments": row["n_experiments"],
                "measured_sense": measured_sense,
                "measured_antisense": measured_antisense,
                "pred_sense": pred_sense,
                "pred_antisense": pred_antisense,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.out_dir / f"motif_metaplot_{head}_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"{head}: manifest saved to {manifest_path}")

    preview_path = args.out_dir / f"motif_metaplot_{head}_preview.png"
    plot_preview_grid(preview_rows, preview_path)
    print(f"{head}: sanity-check preview grid saved to {preview_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head",
        choices=["count", "profile"],
        action="append",
        help="Modisco head to plot. Repeat to run both. Default: count and profile.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of clusters per head, ranked by n_experiments (default: 50)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=150,
        metavar="BP",
        help="half-window size in bp around each motif instance (default: 150)",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=1,
        metavar="BP",
        help="bin size in bp for averaging signal (default: 1)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help=f"output directory (default: {DEFAULT_OUT_DIR.relative_to(REPO_ROOT)}/)",
    )
    args = parser.parse_args()

    heads = args.head or ["count", "profile"]

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = {int(k): v for k, v in yaml.safe_load(f)["folds"].items()}
    all_chroms = list(chain.from_iterable(chrom_splits.values()))

    n_reads_map = {}
    if N_READS_PATH.exists():
        with open(N_READS_PATH) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    n_reads_map[parts[0]] = float(parts[4])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for head in heads:
        process_head_metaplots(head, args, config, n_reads_map, all_chroms)


if __name__ == "__main__":
    main()
