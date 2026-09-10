#!/usr/bin/env python3
"""For every motif, measure how much more attribution "energy" sits outside
its trimmed core in the hits-averaged CWM than in the original MoDISco
discovery archetype -- a general, identity-agnostic candidate for deciding
which motifs need filter_low_confidence_hits.py's hit_seqlet_confidence
(tangermeme recursive_seqlets) corroboration floor, without relying on
cwm_similarity (which stays >0.9 for CA-Inr regardless of this problem --
its trimmed core is only ~4bp, short enough that almost any hit containing
it scores well against a near-zero-flank archetype, structurally blind to
background contamination -- see filter_low_confidence_hits.py's module
docstring) or on manually identifying a MotifCompendium cluster.

This is the same underlying phenomenon --seqlet-compendium-clusters was
built to address, checked with a different, atlas-wide-safe signal: for
each motif m,

    hits_fc, modisco_fc = the same two CWM arrays report_bpnet.py's
        report/CWMs/*/hits_fc.png and modisco_fc.png plot (built via
        finemo.evaluation.get_cwms and finemo.data_io.load_modisco_motifs,
        called directly here rather than through `finemo report`'s CLI,
        which -- as established while root-causing this -- discards
        --cwm-trim-coords/--min-trim-len for any hits file other than
        hits_unique.tsv and always re-derives a flat, un-floored trim
        window instead)

    background_ratio(cwm) = sum(cwm[:, ~trim_mask]**2) / sum(cwm[:, trim_mask]**2)

    excess = background_ratio(hits_fc) - background_ratio(modisco_fc)

A motif whose real hits are dominated by the same tight core the discovery
seqlets were built from keeps excess near 0 (hits_fc and modisco_fc reflect
the same, usually small, background level). A motif whose hit set includes
a lot of positions carrying real attribution outside the core -- exactly
what hits_fc.png showed for CA-Inr in B-cell/neuron/liver, and NOT what it
showed for K562's own (differently-shaped, RC'ed) Inr -- pushes excess well
above 0. A difference, not a ratio: background_ratio(modisco_fc) is often
genuinely ~0 for a cleanly-discovered motif, and dividing by that blows a
ratio-of-ratios up into a meaningless, unbounded number (found by hand on
synthetic data) -- a difference degrades gracefully instead.

Diagnostic only -- does not filter anything or write output. Motifs are
sorted by ratio descending so the worst offenders are easy to spot; compare
against cwm_similarity (pass --report-tsv to have those printed alongside)
to see directly why cwm_similarity misses this for short-core motifs.

Usage:
    python src/bpnet/hitcall/diagnose_background_energy_ratio.py -e ENCSR342WAR --min-trim-len 6
    python src/bpnet/hitcall/diagnose_background_energy_ratio.py -e ENCSR342WAR --min-trim-len 6 \\
        --report-tsv hitcalls/bpnet/ENCSR342WAR_profile/trimcoords-ENCSR342WAR_profile_trim_coords_min6bp/report/motif_report.tsv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from finemo.data_io import load_mapping_tuple, load_modisco_motifs, load_regions_npz
from finemo.evaluation import get_cwms

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, resolve_hits_path, trim_suffix
from filter_by_seqlet_importance import build_peak_row_index

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MIN_HITS = 20
DEFAULT_MIN_GAP_RATIO = 0.2
DEFAULT_MIN_TOP_EXCESS = 0.05
EPS = 1e-12


def single_elbow_count(values_desc, min_excess=0.0):
    """How many of the top-ranked values sit before the sharpest break in
    the curve: the point of maximum vertical distance from the straight
    chord connecting the first and last (rank, value) point, the standard
    single-line-distance elbow/knee heuristic. `values_desc` must already
    be sorted descending; only values > min_excess are considered at all.
    """
    values = np.asarray(values_desc, dtype=float)
    values = values[values > min_excess]
    n = len(values)
    if n < 3:
        return n
    x = np.arange(n, dtype=float) / (n - 1)
    y_range = values[0] - values[-1]
    if y_range <= 0:
        return 0
    y = (values - values[-1]) / y_range
    dist = y - (1 - x)
    return int(np.argmax(dist)) + 1


def detect_elbow_count(
    values_desc, min_excess=0.0, min_pool=5, max_rounds=6,
    min_gap_ratio=DEFAULT_MIN_GAP_RATIO, min_top_excess=DEFAULT_MIN_TOP_EXCESS,
):
    """Recursively apply single_elbow_count: a single extreme top value
    (e.g. K562's TATA at 0.48 vs. everything else <=0.23) pulls the single-
    pass elbow to cut right after itself, silently missing a second, still-
    real-but-smaller break right after (found by hand: K562's real, hand-
    confirmed positives are TATA + a GC-rich SP/KLF-like repeat-calling
    motif + TA-Inr, ranks 1-3, but a single elbow pass only ever recovers
    rank 1 alone). Each round's cut is kept only if its gap is at least
    `min_gap_ratio` of the *remaining* pool's own spread -- a later, weaker
    break, now large only relative to an already-shrunk scale, is rejected
    rather than chased indefinitely. Validated against two hand-labeled
    real experiments (K562 profile: TATA/GC-rich-SP-KLF-repeat/TA-Inr;
    ENCSR342WAR/neuron profile: CA-Inr + 2 other hand-confirmed-noisy
    motifs), both recovering exactly 3 -- stable across min_gap_ratio
    roughly 0.12-0.25, not a single fragile lucky value.

    min_top_excess is a separate, absolute floor, checked before any of the
    above: every gap/spread check above is purely *relative*, so if every
    motif's excess were tiny noise (e.g. 0.003/0.0025/0.001 -- a genuinely
    clean experiment with nothing real to flag), the relative machinery
    could still find "a gap" among those tiny numbers and return a nonzero
    count. Returns 0 immediately if the single largest excess doesn't clear
    this floor, regardless of what relative structure exists below it. Not
    independently calibrated against a known-clean experiment (no example
    of one on hand) -- 0.05 is a conservative placeholder, well under every
    hand-confirmed real case seen so far (smallest: K562's TA-Inr at 0.12),
    but should be revisited once a genuinely-clean reference experiment is
    available to check it against.
    """
    values = np.asarray(sorted(values_desc, reverse=True), dtype=float)
    values = values[values > min_excess]
    if len(values) == 0 or values[0] < min_top_excess:
        return 0
    total = 0
    remaining = values
    for _ in range(max_rounds):
        if len(remaining) < min_pool:
            break
        c = single_elbow_count(remaining, min_excess=min_excess)
        if c <= 0 or c >= len(remaining):
            break
        gap = remaining[c - 1] - remaining[c]
        local_spread = remaining[0] - remaining[-1]
        if local_spread <= 0 or gap / local_spread < min_gap_ratio:
            break
        total += c
        remaining = remaining[c:]
    return total


def build_positions_df(hits_group, peak_row_index, peak_region_starts):
    peak_idx = hits_group["peak_id"].map(peak_row_index).to_numpy()
    peak_region_start = peak_region_starts[peak_idx]
    return pl.DataFrame(
        {
            "peak_id": peak_idx,
            "start_untrimmed": hits_group["start_untrimmed"].to_numpy(),
            "peak_region_start": peak_region_start,
            "is_revcomp": (hits_group["strand"] == "-").to_numpy(),
        }
    )


def compute_background_excess_by_motif(
    hits, regions, motifs_df, cwms_modisco, trim_masks, peak_row_index, peak_region_starts,
    min_hits=DEFAULT_MIN_HITS,
):
    """Per-motif background_ratio(hits_fc) - background_ratio(modisco_fc)
    (see module docstring for the formula and rationale). Returns a list of
    dicts, one per motif with >= min_hits hits, each with motif_name/
    n_hits/excess/bg_ratio_hits/bg_ratio_modisco -- NOT sorted, callers
    sort as needed (e.g. by 'excess' descending before detect_elbow_count).
    """
    motif_width = cwms_modisco.shape[2]
    rows = []
    for motif_name, group in hits.groupby("motif_name", sort=False):
        if len(group) < min_hits:
            continue
        motif_row = motifs_df.filter(
            (pl.col("motif_name") == motif_name) & (pl.col("strand") == "+")
        )
        if motif_row.height == 0:
            continue
        motif_id = motif_row["motif_id"][0]

        positions_df = build_positions_df(group, peak_row_index, peak_region_starts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hits_fc = get_cwms(regions, positions_df, motif_width)
        modisco_fc = cwms_modisco[motif_id]
        mask = trim_masks[motif_id].astype(bool)

        core_energy_hits = float((hits_fc[:, mask] ** 2).sum())
        bg_energy_hits = float((hits_fc[:, ~mask] ** 2).sum())
        core_energy_modisco = float((modisco_fc[:, mask] ** 2).sum())
        bg_energy_modisco = float((modisco_fc[:, ~mask] ** 2).sum())

        bg_ratio_hits = bg_energy_hits / (core_energy_hits + EPS)
        bg_ratio_modisco = bg_energy_modisco / (core_energy_modisco + EPS)
        excess = bg_ratio_hits - bg_ratio_modisco

        rows.append(dict(
            motif_name=motif_name, n_hits=len(group), excess=excess,
            bg_ratio_hits=bg_ratio_hits, bg_ratio_modisco=bg_ratio_modisco,
        ))
    return rows


def load_and_compute_background_excess(
    experiment, head, min_trim_len=None, model_dir=None, modisco_h5_override=None,
    min_hits=DEFAULT_MIN_HITS, verbose=False,
):
    """Convenience wrapper doing the same file resolution/loading main()
    does, for callers (e.g. filter_low_confidence_hits.py's scoping) that
    just want the computed rows without duplicating that plumbing. Exits
    the process on missing files, matching every other script in this
    directory's error handling -- these are all standalone CLI tools first.
    """
    model_dir_name = Path(model_dir).name if model_dir else experiment
    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    trim_coords_path = (
        modisco_dir / f"{experiment}_{head}_trim_coords_min{min_trim_len}bp.tsv"
        if min_trim_len is not None
        else None
    )
    suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, trim_coords_path)
    exp_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir

    hits_path = resolve_hits_path(hits_dir, verbose=verbose)
    if hits_path is None:
        print(f"Error: no hits found in {hits_dir}", file=sys.stderr)
        sys.exit(1)
    regions_npz = exp_dir / "regions.npz"
    if not regions_npz.exists():
        print(f"Error: regions.npz not found: {regions_npz}", file=sys.stderr)
        sys.exit(1)
    modisco_h5 = (
        Path(modisco_h5_override) if modisco_h5_override
        else modisco_dir / f"{experiment}_{head}.modisco.h5"
    )
    if not modisco_h5.exists():
        print(f"Error: modisco h5 not found: {modisco_h5}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"[background_excess] Reading hits from {hits_path}")
        print(f"[background_excess] Reading regions from {regions_npz}")
        print(f"[background_excess] Reading motif CWMs from {modisco_h5} (trim_coords={trim_coords_path})")

    hits = pd.read_csv(hits_path, sep="\t")
    sequences, contribs, peaks_df, _ = load_regions_npz(str(regions_npz))
    regions = contribs * sequences if contribs.ndim == 3 else contribs[:, None, :] * sequences
    peak_row_index = build_peak_row_index(peaks_df)
    peak_region_starts = peaks_df["peak_region_start"].to_numpy()

    trim_coords = (
        load_mapping_tuple(str(trim_coords_path), int)
        if trim_coords_path and trim_coords_path.exists() else None
    )
    if trim_coords_path is not None and trim_coords is None and verbose:
        print(f"[background_excess] Note: {trim_coords_path} not found, falling back to default cwm_trim_threshold")

    motifs_df, cwms_modisco, trim_masks, _ = load_modisco_motifs(
        str(modisco_h5), trim_coords, None, DEFAULT_CWM_TRIM_THRESHOLD,
        "cwm", None, None, None, 1.0, True,
    )

    return compute_background_excess_by_motif(
        hits, regions, motifs_df, cwms_modisco, trim_masks,
        peak_row_index, peak_region_starts, min_hits=min_hits,
    )


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
    parser.add_argument("--modisco-h5", type=str, default=None)
    parser.add_argument(
        "--min-hits", type=int, default=DEFAULT_MIN_HITS,
        help=f"skip a motif entirely if it has fewer hits than this (default: {DEFAULT_MIN_HITS})",
    )
    parser.add_argument(
        "--report-tsv", type=str, default=None,
        help="optional report_bpnet.py motif_report.tsv to print cwm_similarity alongside, for comparison",
    )
    parser.add_argument(
        "--min-gap-ratio", type=float, default=DEFAULT_MIN_GAP_RATIO,
        help=f"see detect_elbow_count's docstring (default: {DEFAULT_MIN_GAP_RATIO})",
    )
    parser.add_argument(
        "--min-top-excess", type=float, default=DEFAULT_MIN_TOP_EXCESS,
        help=(
            "absolute floor: return 0 in-scope motifs regardless of relative "
            f"gap structure if the single largest excess doesn't clear this "
            f"(default: {DEFAULT_MIN_TOP_EXCESS}, an uncalibrated placeholder -- "
            "see detect_elbow_count's docstring)"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    rows = load_and_compute_background_excess(
        args.experiment, args.head, min_trim_len=args.min_trim_len, model_dir=args.model_dir,
        modisco_h5_override=args.modisco_h5, min_hits=args.min_hits, verbose=args.verbose,
    )
    if not rows:
        print("No motif had enough hits to compute a ratio.", file=sys.stderr)
        sys.exit(1)

    report_df = None
    if args.report_tsv:
        report_path = Path(args.report_tsv)
        if report_path.exists():
            report_df = pd.read_csv(report_path, sep="\t").set_index("motif_name")["cwm_similarity"]
        else:
            print(f"Warning: --report-tsv {report_path} not found, skipping", file=sys.stderr)
    if report_df is not None:
        for r in rows:
            r["cwm_similarity"] = report_df.get(r["motif_name"], float("nan"))

    rows.sort(key=lambda r: -r["excess"])
    n_in_scope = detect_elbow_count(
        [r["excess"] for r in rows],
        min_gap_ratio=args.min_gap_ratio, min_top_excess=args.min_top_excess,
    )

    header = f"{'motif_name':<28} {'n_hits':>8} {'bg_ratio(hits_fc)':>18} {'bg_ratio(modisco_fc)':>21} {'excess':>10}"
    if report_df is not None:
        header += f" {'cwm_similarity':>14}"
    print(header)
    for i, r in enumerate(rows):
        marker = "*" if i < n_in_scope else " "
        line = (
            f"{marker}{r['motif_name']:<27} {r['n_hits']:>8} {r['bg_ratio_hits']:>18.4f} "
            f"{r['bg_ratio_modisco']:>21.4f} {r['excess']:>10.4f}"
        )
        if report_df is not None:
            line += f" {r['cwm_similarity']:>14.4f}"
        print(line)
    print(f"\n{n_in_scope} motif(s) marked '*' as in-scope by detect_elbow_count (min_gap_ratio={args.min_gap_ratio})")


if __name__ == "__main__":
    main()
