#!/usr/bin/env python3
"""Plot every motif's real-TSS-relative hit position in one combined figure
-- the classic core-promoter "spacing syntax" view (e.g. TATA sitting at a
sharp -25 to -30bp offset, Inr at 0, DPE/DPR downstream), generalizing
diagnose_hit_summit_distance.py's per-motif print-only summary into a single
comparative plot across the whole motif set.

Reuses diagnose_hit_summit_distance.py's summit-join machinery directly
(load_filtered_peaks/infer_coordinate_mode/build_summit_lookup/
compute_distances) rather than reimplementing it -- same real PRO-cap
summit coordinates (not Fi-NeMo's synthetic peak-midpoint "summit"), same
TSS-relative sign convention (upstream negative regardless of the summit's
own genomic strand).

One violin per motif, sorted by median offset, restricted to motifs with at
least --min-hits matched hits (unmatched/too-rare motifs are silently
dropped, printed with -v). Motifs with a diffuse, non-peaked distribution
(repeat noise, or a motif with no fixed positional relationship to the TSS)
show up as wide/flat violins; real core-promoter elements show up as narrow
violins at a consistent offset.

Defaults to every motif clearing --min-hits, but this is meant to be run
with an explicit --motifs subset for actual figures (paper-facing or
otherwise) -- plotting literally every discovered motif in one figure
produces a cluttered, unreadable plot for any experiment with more than a
handful of motifs. For general/diagnostic all-motif viewing, prefer
diagnose_hit_signal_metaplot.py's --all-motifs mode or
plot_motif_pair_spacing.py instead, which are built for that.

Usage:
    python src/bpnet/hitcall/plot_motif_spacing_syntax.py -e ENCSR220XSM \\
        --motifs pos_patterns.pattern_12 --motifs pos_patterns.pattern_1
    python src/bpnet/hitcall/plot_motif_spacing_syntax.py -e ENCSR220XSM --min-trim-len 6 -v
    python src/bpnet/hitcall/plot_motif_spacing_syntax.py -e ENCSR220XSM --window 150 --min-hits 50
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, resolve_hits_path, trim_suffix
from diagnose_hit_summit_distance import (
    build_summit_lookup,
    compute_distances,
    infer_coordinate_mode,
    load_filtered_peaks,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
DEFAULT_MIN_HITS = 20
DEFAULT_WINDOW = 200


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-e", "--experiment", type=str, required=True)
    parser.add_argument("-m", "--model-dir", type=str, default=None)
    parser.add_argument(
        "--head", type=str, default="profile", choices=["profile", "count"]
    )
    parser.add_argument("--min-trim-len", type=int, default=None, metavar="BP")
    parser.add_argument(
        "--min-hits", type=int, default=DEFAULT_MIN_HITS,
        help=f"skip a motif entirely if it has fewer matched hits than this (default: {DEFAULT_MIN_HITS})",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW, metavar="BP",
        help=f"clip the x-axis to +/- this many bp (default: {DEFAULT_WINDOW}); "
        "distances beyond this are still used for the violin (not dropped), just "
        "not shown past the axis limit",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figures" / "metaplots",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = config["experiments"]
    if args.experiment not in experiments:
        print(f"Error: {args.experiment} not found in config", file=sys.stderr)
        sys.exit(1)
    filtered_peaks_path = REPO_ROOT / experiments[args.experiment]["processed"]["filtered_peaks"]

    model_dir_name = Path(args.model_dir).name if args.model_dir else args.experiment
    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    trim_coords = (
        modisco_dir / f"{args.experiment}_{args.head}_trim_coords_min{args.min_trim_len}bp.tsv"
        if args.min_trim_len is not None
        else None
    )
    suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, trim_coords)
    exp_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{args.head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir

    hits_path = resolve_hits_path(hits_dir, verbose=args.verbose)
    if hits_path is None:
        print(f"Error: no hits found in {hits_dir}", file=sys.stderr)
        sys.exit(1)
    if not filtered_peaks_path.exists():
        print(f"Error: filtered_peaks not found: {filtered_peaks_path}", file=sys.stderr)
        sys.exit(1)

    hits = pd.read_csv(hits_path, sep="\t")
    filtered_peaks_df = load_filtered_peaks(filtered_peaks_path)

    has_any_summit = (
        filtered_peaks_df["summits_pos_list"].apply(len) + filtered_peaks_df["summits_neg_list"].apply(len)
    ) > 0
    mode = infer_coordinate_mode(filtered_peaks_df[has_any_summit], verbose=args.verbose)
    summit_lookup = build_summit_lookup(filtered_peaks_df[has_any_summit], mode)

    signed, matched = compute_distances(hits, summit_lookup)

    rows = []
    for motif_name, group in hits.groupby("motif_name"):
        group_signed = signed[group.index.to_numpy()]
        group_signed = group_signed[~np.isnan(group_signed)]
        if len(group_signed) < args.min_hits:
            if args.verbose:
                print(f"Skipping {motif_name}: only {len(group_signed)} matched hits")
            continue
        rows.append((motif_name, group_signed))

    if not rows:
        print("No motif had enough matched hits to plot.", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: np.median(r[1]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_height = max(3, 0.28 * len(rows))
    fig, ax = plt.subplots(figsize=(8, fig_height))

    data = [np.clip(d, -args.window * 1.5, args.window * 1.5) for _, d in rows]
    labels = [f"{name} (n={len(d)})" for name, d in rows]

    parts = ax.violinplot(data, vert=False, showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("tab:blue")
        body.set_alpha(0.6)

    ax.set_yticks(range(1, len(rows) + 1))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlim(-args.window, args.window)
    ax.set_xlabel("Hit-to-summit distance, bp (upstream negative)")
    ax.set_title(f"{args.experiment} {args.head}: motif spacing relative to real PRO-cap TSS")

    out_path = args.out_dir / f"{args.experiment}_{args.head}_motif_spacing_syntax.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path} ({len(rows)} motifs)")


if __name__ == "__main__":
    main()
