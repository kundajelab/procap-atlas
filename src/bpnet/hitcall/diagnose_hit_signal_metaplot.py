#!/usr/bin/env python3
"""Compare real, observed PRO-cap signal around two categories of a motif's
hits: "normal" (peaks where the motif is called exactly once -- the
uncontested case) vs. "excess" (peaks where it's called several times, the
bulk of the volume flagged for CA-Inr in B-cell/neuron/liver's profile-head
hits -- see filter_low_confidence_hits.py's and diagnose_hit_summit_distance.py's
module docstrings for the investigation this follows on from).

Every per-hit statistic checked so far (hit_correlation, hit_coefficient,
hit_importance, cwm_similarity) fails to distinguish "excess" hits from
"normal" ones -- they look comparably well-supported by the model's own
attributions. None of those are ground truth, though: they all describe how
well a hit matches what the *model* attributes, not whether real
transcription initiation actually happens there. This script checks that
directly, by pulling the actual observed PRO-cap BigWig signal (RPM-
normalized, strand-oriented by each hit's own called strand) around both
groups and plotting them for direct comparison. If "excess" hits sit on
real, if secondary, initiation signal, they should show a real (if smaller)
sense-strand peak at position 0, just like "normal" hits. If they're
model-attribution artifacts uncorrelated with real initiation, the observed
signal at position 0 should look flat/indistinguishable from flanking
background.

Reuses metaplot_tss.py's collect_windows() (BigWig extraction/RPM
normalization/strand-orientation) directly rather than reimplementing it --
same signal-extraction code as every other TSS metaplot in this repo, just
fed motif-hit coordinates instead of GENCODE TSSs.

Usage:
    python src/bpnet/hitcall/diagnose_hit_signal_metaplot.py -e ENCSR342WAR --min-trim-len 6 \\
        --motif-name pos_patterns.pattern_2
    python src/bpnet/hitcall/diagnose_hit_signal_metaplot.py -e ENCSR342WAR --min-trim-len 6 \\
        --motif-name pos_patterns.pattern_2 --excess-min-count 2 --excess-max-count 4
    python src/bpnet/hitcall/diagnose_hit_signal_metaplot.py -e ENCSR220XSM --min-trim-len 6 \\
        --motif-name pos_patterns.pattern_1 --window 300 --bin-size 5
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "metaplot"))
from metaplot_tss import collect_windows  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
DEFAULT_WINDOW = 200
DEFAULT_BIN_SIZE = 5
DEFAULT_NORMAL_COUNT = 1
DEFAULT_EXCESS_MIN_COUNT = 2
DEFAULT_EXCESS_MAX_COUNT = 4


def load_total_reads(experiment):
    with open(N_READS_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] == experiment and len(parts) >= 5:
                return float(parts[4])
    return None


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
        "--motif-name", type=str, required=True,
        help="exact motif_name to compare, e.g. pos_patterns.pattern_2",
    )
    parser.add_argument(
        "--normal-count", type=int, default=DEFAULT_NORMAL_COUNT, metavar="N",
        help=f"peaks with exactly this many hits of the motif form the "
        f"'normal' group (default: {DEFAULT_NORMAL_COUNT})",
    )
    parser.add_argument(
        "--excess-min-count", type=int, default=DEFAULT_EXCESS_MIN_COUNT, metavar="N",
        help=f"peaks with at least this many hits form the 'excess' group "
        f"(default: {DEFAULT_EXCESS_MIN_COUNT})",
    )
    parser.add_argument(
        "--excess-max-count", type=int, default=DEFAULT_EXCESS_MAX_COUNT, metavar="N",
        help=f"peaks with at most this many hits form the 'excess' group -- "
        f"capped by default to the typical bulk case, not the rare extreme "
        f"tail already caught by filter_repeat_density.py (default: {DEFAULT_EXCESS_MAX_COUNT})",
    )
    parser.add_argument(
        "--split-excess-by-strand-mix", action="store_true",
        help=(
            "split the 'excess' group into peaks where its hits fall on both "
            "strands ('mixed-strand', consistent with genuine divergent-"
            "promoter biology -- one Inr per direction) vs. all on the same "
            "strand ('same-strand', not explained by divergent transcription). "
            "Motivated by same-motif hit_correlation/hit_coefficient/"
            "hit_importance being unable to tell excess hits apart from normal "
            "ones -- checks whether real observed signal can, and whether "
            "that split lines up with the divergent-promoter explanation."
        ),
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, metavar="BP")
    parser.add_argument("--bin-size", type=int, default=DEFAULT_BIN_SIZE, metavar="BP")
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
    processed = experiments[args.experiment]["processed"]
    pl_path = REPO_ROOT / processed["pl_bigwig"]
    mn_path = REPO_ROOT / processed["mn_bigwig"]
    total_reads = load_total_reads(args.experiment)
    if total_reads is None:
        print(f"Error: {args.experiment} not found in {N_READS_PATH}", file=sys.stderr)
        sys.exit(1)

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
    if args.verbose:
        print(f"Reading hits from {hits_path}")

    hits = pd.read_csv(hits_path, sep="\t")
    hits = hits[hits["motif_name"] == args.motif_name].copy()
    if hits.empty:
        print(f"Error: no hits for motif {args.motif_name}", file=sys.stderr)
        sys.exit(1)

    counts = hits.groupby("peak_id").size()
    normal_peaks = counts[counts == args.normal_count].index
    excess_peaks = counts[
        (counts >= args.excess_min_count) & (counts <= args.excess_max_count)
    ].index

    groups = {
        f"normal (n={args.normal_count} hit/peak)": hits[hits["peak_id"].isin(normal_peaks)],
    }
    excess_hits = hits[hits["peak_id"].isin(excess_peaks)]
    if args.split_excess_by_strand_mix:
        strand_nunique = excess_hits.groupby("peak_id")["strand"].nunique()
        mixed_strand_peaks = strand_nunique[strand_nunique > 1].index
        same_strand_peaks = strand_nunique[strand_nunique == 1].index
        groups[f"excess mixed-strand ({args.excess_min_count}-{args.excess_max_count}/peak)"] = (
            excess_hits[excess_hits["peak_id"].isin(mixed_strand_peaks)]
        )
        groups[f"excess same-strand ({args.excess_min_count}-{args.excess_max_count}/peak)"] = (
            excess_hits[excess_hits["peak_id"].isin(same_strand_peaks)]
        )
    else:
        groups[f"excess ({args.excess_min_count}-{args.excess_max_count} hits/peak)"] = excess_hits

    results = {}
    for label, group_hits in groups.items():
        tss_list = [
            (row.chr, int((row.start + row.end) // 2), row.strand)
            for row in group_hits.itertuples(index=False)
        ]
        sense, antisense = collect_windows(
            pl_path, mn_path, tss_list, args.window, args.bin_size, total_reads
        )
        if len(sense) == 0:
            print(f"Warning: no windows extracted for '{label}' (n_hits={len(group_hits)})", file=sys.stderr)
            continue
        results[label] = (sense.mean(axis=0), antisense.mean(axis=0), len(sense))
        print(f"{label}: n_hits={len(group_hits)}, n_windows_extracted={len(sense)}")

    if not results:
        print("Error: no windows extracted for either group", file=sys.stderr)
        sys.exit(1)

    # Fi-NeMo's hit `strand` reflects which orientation of this experiment's
    # own, independently-discovered MoDISco pattern matched -- unlike
    # GENCODE gene strand, that has no guaranteed relationship to real
    # transcription direction (a motif can be discovered/labeled "+" in one
    # experiment and get the RC labeled "+" in another). Detect this by
    # checking which of sense/antisense actually peaks at the hit center in
    # the least ambiguous group (the first one, "normal"), and swap both
    # groups' sense/antisense if it's inverted, rather than trusting
    # hit-strand as a real sense/antisense proxy.
    center = len(next(iter(results.values()))[0]) // 2
    ref_sense, ref_antisense, _ = next(iter(results.values()))
    if ref_antisense[center] > ref_sense[center]:
        print("Note: hit-strand looks RC'ed relative to real transcription direction "
              "(antisense > sense at center in the reference group) -- swapping sense/antisense.")
        results = {label: (a, s, n) for label, (s, a, n) in results.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_bins = (2 * args.window) // args.bin_size
    positions = np.linspace(-args.window, args.window, n_bins, endpoint=False) + args.bin_size / 2

    fig, ax = plt.subplots(figsize=(7, 4))
    color_by_prefix = {
        "normal": "tab:green",
        "excess mixed-strand": "tab:orange",
        "excess same-strand": "tab:red",
        "excess": "tab:red",
    }
    for label, (sense, antisense, n) in results.items():
        color = next(c for prefix, c in color_by_prefix.items() if label.startswith(prefix))
        ax.plot(positions, sense, color=color, linewidth=1.5, label=f"{label} sense (n={n:,})")
        ax.plot(positions, -antisense, color=color, linewidth=1.5, linestyle=":", alpha=0.7)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Position relative to hit center (bp)")
    ax.set_ylabel("Mean signal (RPM)")
    subtitle = (
        "observed PRO-cap signal, normal vs. excess (mixed- vs. same-strand)"
        if args.split_excess_by_strand_mix
        else "observed PRO-cap signal, normal vs. excess hits"
    )
    ax.set_title(f"{args.experiment} {args.head}: {args.motif_name}\n{subtitle}")
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    suffix = "_strand_split" if args.split_excess_by_strand_mix else ""
    out_path = (
        args.out_dir
        / f"{args.experiment}_{args.head}_{args.motif_name.replace('.', '_')}_normal_vs_excess{suffix}.png"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
