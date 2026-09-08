"""Diagnose whether each motif's hits cluster at a consistent position
relative to the peak's real PRO-cap TSS summit, or are scattered across the
peak window like repeat noise.

filter_by_seqlet_importance.py established that TATA's overcalled hits in
K562 ENCSR220XSM aren't weak-magnitude noise -- they carry real
hit_importance, comparable to genuine discovery seqlets. The problem is CWM
*shape* mismatch (what cwm_similarity/hit_correlation measure), not signal
weakness, and no magnitude-based threshold can fix a problem with no
magnitude gap to exploit. This script checks a different, motif-identity-
agnostic axis: real core-promoter elements (TATA, Inr) sit at a fixed,
narrow offset from the transcription start site (TATA ~-25 to -30bp);
AT-rich repeat-context noise elsewhere in a ~2kb peak window has no reason
to respect that offset. If a motif's real hits are position-constrained and
its noise hits aren't, the per-motif hit-to-summit distance distribution
should show a sharp mode (real) sitting on top of, or instead of, a diffuse
spread (noise) -- checkable without any consensus/JASPAR identity
knowledge, using only each motif's own hits.

Fi-NeMo's hit-calling never sees a real biological summit: call_hits_bpnet's
build_peaks_narrowpeak() feeds it a synthetic narrowPeak whose "summit" is
just the peak window's own midpoint (summit=0 offset from a start already
set to the midpoint), purely so Fi-NeMo's internal peak_start+summit
bookkeeping resolves to something. The real PRO-cap-caller-derived summit
(the actual TSS position(s) within the window) survives untouched in
data/processed/peaks/{experiment}_{biosample}_filtered.bed.gz (columns 6/7:
summit_pos, summit_neg -- whichever strand doesn't apply is "."), which this
script re-reads directly and joins back to hits by peak_name (which Fi-NeMo
hits already carry, encoded by build_peaks_narrowpeak as
f"{chrom}:{start}-{end}" over that same filtered_peaks file's own
coordinates -- no need to touch regions.npz or attributions at all).

The raw summit column's coordinate convention (absolute genomic position,
vs. an ENCODE-narrowPeak-style offset from the peak's own start) isn't
documented anywhere in this repo -- it comes verbatim from whatever the
upstream PRO-cap peak caller wrote. Rather than assume, this script checks
both interpretations against the peak's own [start, end) window and picks
whichever one actually lands inside it, printing the inferred mode and its
support rate so a wrong guess is visible instead of silently corrupting
every distance downstream.

Distance is reported "TSS-relative": upstream of the summit (where TATA
lives) is negative, downstream is positive, regardless of the summit's own
genomic strand -- computed as (hit_center - summit) for a "+"-strand
summit, (summit - hit_center) for a "-"-strand summit, so a real, position-
constrained motif shows up as a concentrated distribution regardless of
which strand's TSS it's associated with. A hit nearer to more than one
candidate summit (bidirectional peaks can carry both a summit_pos and a
summit_neg) is assigned to whichever is nearest.

This is read-only and diagnostic only -- it does not write a filtered hits
file. It does not need regions.npz/attributions, so it's cheap enough to
run directly (no SLURM launcher).

Usage:
    python src/bpnet/hitcall/diagnose_hit_summit_distance.py -e ENCSR220XSM
    python src/bpnet/hitcall/diagnose_hit_summit_distance.py -e ENCSR220XSM --min-trim-len 6 -v
    python src/bpnet/hitcall/diagnose_hit_summit_distance.py -e ENCSR220XSM --concentration-window 50
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from call_hits_bpnet import (
    DEFAULT_CWM_TRIM_THRESHOLD,
    resolve_hits_path,
    trim_suffix,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
DEFAULT_MIN_HITS = 20
DEFAULT_CONCENTRATION_WINDOW = 50
PEAK_NAME_RE = re.compile(r"^(.+):(\d+)-(\d+)$")
FILTERED_PEAKS_COLUMNS = [
    "chrom", "start", "end", "strand", "confidence",
    "uni_or_bidirectional", "summit_pos", "summit_neg",
]


def parse_summit_field(value):
    """"." (absent), a single int, or a comma-separated list of ints
    (the merge script's own comment allows "summit(s)", plural) -> a list
    of ints, possibly empty.
    """
    if value in (".", "", None) or (isinstance(value, float) and np.isnan(value)):
        return []
    return [int(v) for v in str(value).split(",") if v not in (".", "")]


def load_filtered_peaks(filtered_peaks_path):
    df = pd.read_csv(
        filtered_peaks_path,
        sep="\t",
        header=None,
        names=FILTERED_PEAKS_COLUMNS,
        dtype={"chrom": str},
    )
    df["peak_name"] = (
        df["chrom"] + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
    )
    df["summits_pos_list"] = df["summit_pos"].apply(parse_summit_field)
    df["summits_neg_list"] = df["summit_neg"].apply(parse_summit_field)
    return df.set_index("peak_name")


def infer_coordinate_mode(filtered_peaks_df, verbose=False):
    """Decide whether summit_pos/summit_neg are absolute genomic coordinates
    or offsets from the peak's own start, by checking which interpretation
    actually lands inside that peak's own [start, end) window.
    """
    n_abs_in_window = 0
    n_offset_in_window = 0
    n_checked = 0
    for row in filtered_peaks_df.itertuples():
        start, end = row.start, row.end
        for raw in row.summits_pos_list + row.summits_neg_list:
            n_checked += 1
            if start <= raw < end:
                n_abs_in_window += 1
            if start <= start + raw < end:
                n_offset_in_window += 1

    if n_checked == 0:
        print("Error: no peaks with any summit_pos/summit_neg value found.", file=sys.stderr)
        sys.exit(1)

    abs_frac = n_abs_in_window / n_checked
    offset_frac = n_offset_in_window / n_checked
    if verbose:
        print(
            f"Coordinate-mode check (n={n_checked} summit values): "
            f"absolute-in-window={abs_frac:.1%}, offset-from-start-in-window={offset_frac:.1%}"
        )

    mode = "absolute" if abs_frac >= offset_frac else "offset"
    support = max(abs_frac, offset_frac)
    print(f"Inferred summit coordinate mode: {mode} ({support:.1%} land inside the peak's own window)")
    if support < 0.5:
        print(
            "WARNING: neither interpretation confidently lands inside the peak "
            "window -- summit coordinates may use some other convention this "
            "script doesn't handle. Treat results with caution.",
            file=sys.stderr,
        )
    return mode


def build_summit_lookup(filtered_peaks_df, mode):
    """peak_name -> list of (coord, strand) tuples, coords converted to
    absolute genomic positions per the inferred coordinate mode.
    """
    lookup = {}
    for peak_name, row in filtered_peaks_df.iterrows():
        candidates = []
        for raw in row.summits_pos_list:
            coord = raw if mode == "absolute" else row.start + raw
            candidates.append((coord, "+"))
        for raw in row.summits_neg_list:
            coord = raw if mode == "absolute" else row.start + raw
            candidates.append((coord, "-"))
        if candidates:
            lookup[peak_name] = candidates
    return lookup


def nearest_signed_distance(hit_center, candidates):
    """Signed, TSS-relative distance to the nearest candidate summit:
    upstream (where TATA lives) is negative regardless of the summit's own
    genomic strand. Returns (signed, abs) or (None, None) if no candidates.
    """
    best_signed = None
    best_abs = None
    for coord, strand in candidates:
        signed = (hit_center - coord) if strand == "+" else (coord - hit_center)
        a = abs(signed)
        if best_abs is None or a < best_abs:
            best_abs, best_signed = a, signed
    return best_signed, best_abs


def compute_distances(hits, summit_lookup):
    signed = np.full(len(hits), np.nan)
    matched = np.zeros(len(hits), dtype=bool)
    for i, row in enumerate(hits.itertuples(index=False)):
        candidates = summit_lookup.get(row.peak_name)
        if not candidates:
            continue
        hit_center = (row.start + row.end) / 2.0
        s, _ = nearest_signed_distance(hit_center, candidates)
        signed[i] = s
        matched[i] = True
    return signed, matched


def summarize_motif(motif_name, signed_distances, window):
    d = signed_distances[~np.isnan(signed_distances)]
    n = len(d)
    if n == 0:
        return None
    abs_d = np.abs(d)
    frac_in_window = float(np.mean(abs_d <= window))
    return dict(
        motif_name=motif_name,
        n_hits=n,
        median_signed=float(np.median(d)),
        p5=float(np.percentile(d, 5)),
        p95=float(np.percentile(d, 95)),
        median_abs=float(np.median(abs_d)),
        frac_in_window=frac_in_window,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e", "--experiment", type=str, required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
    )
    parser.add_argument(
        "-m", "--model-dir", type=str, default=None,
        help="override model directory, default derived from config",
    )
    parser.add_argument(
        "--head", type=str, default="profile", choices=["profile", "count"],
        help="attribution/motif head the hits were called against (default: profile)",
    )
    parser.add_argument(
        "--min-trim-len", type=int, default=None, metavar="BP",
        help=(
            "must match the value hitcall/launch.py was run with, if any -- "
            "resolves the same per-experiment trim-coords-suffixed output "
            "directory rather than the plain {model_dir_name}_{head}/ one."
        ),
    )
    parser.add_argument(
        "--concentration-window", type=int, default=DEFAULT_CONCENTRATION_WINDOW,
        metavar="BP",
        help=(
            "report the fraction of each motif's hits within +/- this many bp "
            f"of the summit (default: {DEFAULT_CONCENTRATION_WINDOW})"
        ),
    )
    parser.add_argument(
        "--min-hits", type=int, default=DEFAULT_MIN_HITS,
        help=f"skip a motif entirely if it has fewer hits than this (default: {DEFAULT_MIN_HITS})",
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

    if args.verbose:
        print(f"Reading hits from {hits_path}")
        print(f"Reading filtered peaks from {filtered_peaks_path}")

    hits = pd.read_csv(hits_path, sep="\t")
    filtered_peaks_df = load_filtered_peaks(filtered_peaks_path)

    has_any_summit = (
        filtered_peaks_df["summits_pos_list"].apply(len) + filtered_peaks_df["summits_neg_list"].apply(len)
    ) > 0
    mode = infer_coordinate_mode(filtered_peaks_df[has_any_summit], verbose=args.verbose)
    summit_lookup = build_summit_lookup(filtered_peaks_df[has_any_summit], mode)

    signed, matched = compute_distances(hits, summit_lookup)
    n_unmatched = int((~matched).sum())
    if args.verbose or n_unmatched:
        print(
            f"{n_unmatched}/{len(hits)} hits ({n_unmatched / len(hits):.1%}) "
            "could not be matched to any peak with a real summit"
        )

    rows = []
    skipped = []
    for motif_name, group in hits.groupby("motif_name"):
        group_signed = signed[group.index.to_numpy()]
        n_valid = int((~np.isnan(group_signed)).sum())
        if n_valid < args.min_hits:
            skipped.append((motif_name, n_valid))
            continue
        summary = summarize_motif(motif_name, group_signed, args.concentration_window)
        if summary is not None:
            rows.append(summary)

    if not rows:
        print("No motif had enough matched hits to summarize.")
        return

    rows.sort(key=lambda r: -r["frac_in_window"])
    w = args.concentration_window
    print(
        f"\nPer-motif hit-to-summit distance (TSS-relative, upstream negative; "
        f"{len(rows)} motifs with >= {args.min_hits} matched hits):"
    )
    print(
        f"  {'motif_name':<24} {'n_hits':>8} {'median':>8} {'p5':>8} {'p95':>8} "
        f"{f'frac|d|<={w}':>12}"
    )
    for r in rows:
        print(
            f"  {r['motif_name']:<24} {r['n_hits']:>8} {r['median_signed']:>8.1f} "
            f"{r['p5']:>8.1f} {r['p95']:>8.1f} {r['frac_in_window']:>12.1%}"
        )

    if args.verbose and skipped:
        print(f"\n{len(skipped)} motif(s) skipped (fewer than {args.min_hits} matched hits): {skipped}")


if __name__ == "__main__":
    main()
