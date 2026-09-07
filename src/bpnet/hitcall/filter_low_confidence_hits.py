"""Drop the low-confidence mode of a motif's hits, when one is detectable.

Reviewed real per-hit hit_correlation distributions for K562 ENCSR220XSM's
profile head (three motifs failing report_bpnet.py's cwm_similarity QC even
after filter_repeat_density.py): GATA showed a clear bimodal split -- a large
bulk of ambiguous hits plus a distinct, separable population of very
high-confidence hits (correlation trough around the 90th percentile, then
rising again toward the max) -- while TA-Initiator showed no such split at
all (a smooth, monotonically decaying unimodal distribution). That matters:
a motif's aggregate cwm_similarity can be dragged down by a large low-
confidence tail even when a real, legitimate high-confidence subset exists
under the same motif_name, but filtering only helps where that split is
actually there to find.

This is deliberately identity-agnostic (no hardcoded motif names/thresholds,
same reasoning as filter_repeat_density.py): for each motif, it builds a
histogram of --score-column (default hit_correlation), looks for a genuine
local dip-then-rise (trough after the primary mode, followed by a
sufficiently large and sufficiently separated secondary mode), and drops
hits below that trough only when one is found. Motifs with a smooth/unimodal
distribution (e.g. TA-Inr in the case above) are left untouched -- there's no
data-driven cutoff to apply, and filtering them anyway would just be an
arbitrary top-K cut with no principled justification.

Run after filter_repeat_density.py (reads hits_dedensified.tsv if present,
else hits_unique.tsv) and before report_bpnet.py, which prefers this script's
output (hits_confidence_filtered.tsv) when present.

Usage:
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --head count
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --min-trim-len 6
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --score-column hit_similarity
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, trim_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SCORE_COLUMN = "hit_correlation"
DEFAULT_N_BINS = 50
DEFAULT_SMOOTHING_WINDOW = 3
DEFAULT_MIN_RISE_FRAC = 1.15
DEFAULT_MIN_BIN_FRAC = 0.01
DEFAULT_MIN_TOTAL_HITS = 200


def detect_low_confidence_cutoff(
    scores,
    n_bins=DEFAULT_N_BINS,
    smoothing_window=DEFAULT_SMOOTHING_WINDOW,
    min_rise_frac=DEFAULT_MIN_RISE_FRAC,
    min_bin_frac=DEFAULT_MIN_BIN_FRAC,
    min_total_hits=DEFAULT_MIN_TOTAL_HITS,
):
    """Look for a genuine secondary (higher-score) mode past a trough
    following the primary mode of `scores`'s histogram.

    Returns a dict with the cutoff and diagnostic info if found, else None.
    Uses scipy's prominence-based peak finder rather than hand-rolled local-
    or global-extremum logic, which has two failure modes that are hard to
    dodge simultaneously: chasing the *first* local minimum latches onto
    tiny noise wiggles inside a noisy-but-still-descending bulk region, while
    chasing the *global* minimum of the tail gets fooled by the fact that any
    histogram's most extreme bins are sparse and decay toward ~0 anyway, so
    that edge can look like an even deeper "trough" than the real one
    between two modes. Prominence (how much a peak stands out above its
    surrounding valleys, not just its raw height) is robust to both.
    """
    n = len(scores)
    if n < min_total_hits:
        return None

    counts, edges = np.histogram(scores, bins=n_bins)
    counts = counts.astype(float)
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        counts = np.convolve(counts, kernel, mode="same")

    min_prominence = max(min_bin_frac * n, 1.0)
    peak_idxs, _ = find_peaks(counts, prominence=min_prominence)
    if len(peak_idxs) < 2:
        return None  # no distinct secondary mode at all

    primary_idx = peak_idxs[np.argmax(counts[peak_idxs])]
    later_peaks = peak_idxs[peak_idxs > primary_idx]
    if len(later_peaks) == 0:
        return None  # every other mode is at a *lower* score than the primary one
    peak_idx = later_peaks[np.argmax(counts[later_peaks])]

    trough_idx = primary_idx + int(np.argmin(counts[primary_idx : peak_idx + 1]))

    trough_count = max(counts[trough_idx], 1.0)
    peak_count = counts[peak_idx]
    if peak_count < min_bin_frac * n:
        return None  # secondary mode too small relative to this motif's total hits
    if peak_count < trough_count * min_rise_frac:
        return None  # not a large enough rise to trust over histogram noise

    return dict(
        cutoff=float(edges[trough_idx + 1]),
        primary_mode=float(edges[primary_idx]),
        trough=float(edges[trough_idx]),
        secondary_mode=float(edges[peak_idx]),
        trough_count=float(counts[trough_idx]),
        secondary_count=float(peak_count),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        default=None,
        help="override model directory, default derived from config",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="profile",
        choices=["profile", "count"],
        help="attribution/motif head the hits were called against (default: profile)",
    )
    parser.add_argument(
        "--min-trim-len",
        type=int,
        default=None,
        metavar="BP",
        help=(
            "must match the value hitcall/launch.py was run with, if any -- "
            "resolves the same per-experiment trim-coords-suffixed output "
            "directory rather than the plain {model_dir_name}_{head}/ one."
        ),
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default=DEFAULT_SCORE_COLUMN,
        help=f"per-hit Fi-NeMo score column to check for bimodality (default: {DEFAULT_SCORE_COLUMN})",
    )
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument(
        "--min-rise-frac",
        type=float,
        default=DEFAULT_MIN_RISE_FRAC,
        help=(
            "required ratio of secondary-mode bin count to trough bin count "
            f"to treat the rise as real, not histogram noise (default: {DEFAULT_MIN_RISE_FRAC})"
        ),
    )
    parser.add_argument(
        "--min-bin-frac",
        type=float,
        default=DEFAULT_MIN_BIN_FRAC,
        help=(
            "minimum secondary-mode bin count, as a fraction of the motif's "
            f"total hit count, to trust it (default: {DEFAULT_MIN_BIN_FRAC})"
        ),
    )
    parser.add_argument(
        "--min-total-hits",
        type=int,
        default=DEFAULT_MIN_TOTAL_HITS,
        help=(
            "skip bimodality detection entirely for motifs with fewer than "
            f"this many hits (default: {DEFAULT_MIN_TOTAL_HITS}, too few for a "
            "reliable histogram)"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    model_dir_name = Path(args.model_dir).name if args.model_dir else args.experiment

    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    trim_coords = (
        modisco_dir
        / f"{args.experiment}_{args.head}_trim_coords_min{args.min_trim_len}bp.tsv"
        if args.min_trim_len is not None
        else None
    )
    suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, trim_coords)
    exp_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{args.head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir

    hits_dedensified_path = hits_dir / "hits_dedensified.tsv"
    hits_unique_path = hits_dir / "hits_unique.tsv"
    hits_path = (
        hits_dedensified_path if hits_dedensified_path.exists() else hits_unique_path
    )
    if not hits_path.exists():
        print(f"Error: hits not found: {hits_path}", file=sys.stderr)
        print("Run call_hits_bpnet.py first.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Reading hits from {hits_path}")

    hits = pd.read_csv(hits_path, sep="\t")
    if args.score_column not in hits.columns:
        print(
            f"Error: --score-column {args.score_column!r} not found in "
            f"{hits_path} (columns: {list(hits.columns)})",
            file=sys.stderr,
        )
        sys.exit(1)

    keep_mask = np.ones(len(hits), dtype=bool)
    filtered_motifs = []
    unfiltered_motifs = []
    for motif_name, group in hits.groupby("motif_name", sort=False):
        scores = group[args.score_column].to_numpy()
        result = detect_low_confidence_cutoff(
            scores,
            n_bins=args.n_bins,
            smoothing_window=args.smoothing_window,
            min_rise_frac=args.min_rise_frac,
            min_bin_frac=args.min_bin_frac,
            min_total_hits=args.min_total_hits,
        )
        if result is None:
            unfiltered_motifs.append(motif_name)
            continue

        below_cutoff = group.index[group[args.score_column] < result["cutoff"]]
        keep_mask[below_cutoff] = False
        filtered_motifs.append(
            dict(
                motif_name=motif_name,
                n_total=len(group),
                n_dropped=len(below_cutoff),
                **result,
            )
        )

    kept = hits[keep_mask]

    if filtered_motifs:
        print(
            f"Detected a low-confidence mode (by {args.score_column}) in "
            f"{len(filtered_motifs)} motif(s):"
        )
        for r in sorted(filtered_motifs, key=lambda r: -r["n_dropped"]):
            print(
                f"  {r['motif_name']}: cutoff={r['cutoff']:.3f} "
                f"(trough={r['trough']:.3f}/{r['trough_count']:.0f} hits, "
                f"secondary_mode={r['secondary_mode']:.3f}/{r['secondary_count']:.0f} hits) "
                f"-- dropped {r['n_dropped']}/{r['n_total']} hits "
                f"({r['n_dropped'] / r['n_total']:.1%})"
            )
    else:
        print(f"No motif showed a detectable low-confidence mode by {args.score_column}.")

    if args.verbose and unfiltered_motifs:
        print(
            f"\n{len(unfiltered_motifs)} motif(s) left untouched (no secondary "
            f"mode detected): {unfiltered_motifs}"
        )

    out_path = hits_dir / "hits_confidence_filtered.tsv"
    kept.to_csv(out_path, sep="\t", index=False)
    print(
        f"\nKept {len(kept)}/{len(hits)} hits "
        f"({len(hits) - len(kept)} dropped from {len(filtered_motifs)} motifs)"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
