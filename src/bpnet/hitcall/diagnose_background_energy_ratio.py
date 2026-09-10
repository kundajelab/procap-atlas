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
EPS = 1e-12


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
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    model_dir_name = Path(args.model_dir).name if args.model_dir else args.experiment
    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    trim_coords_path = (
        modisco_dir / f"{args.experiment}_{args.head}_trim_coords_min{args.min_trim_len}bp.tsv"
        if args.min_trim_len is not None
        else None
    )
    suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, trim_coords_path)
    exp_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{args.head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir

    hits_path = resolve_hits_path(hits_dir, verbose=args.verbose)
    if hits_path is None:
        print(f"Error: no hits found in {hits_dir}", file=sys.stderr)
        sys.exit(1)
    regions_npz = exp_dir / "regions.npz"
    if not regions_npz.exists():
        print(f"Error: regions.npz not found: {regions_npz}", file=sys.stderr)
        sys.exit(1)
    modisco_h5 = Path(args.modisco_h5) if args.modisco_h5 else modisco_dir / f"{args.experiment}_{args.head}.modisco.h5"
    if not modisco_h5.exists():
        print(f"Error: modisco h5 not found: {modisco_h5}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Reading hits from {hits_path}")
        print(f"Reading regions from {regions_npz}")
        print(f"Reading motif CWMs from {modisco_h5} (trim_coords={trim_coords_path})")

    hits = pd.read_csv(hits_path, sep="\t")
    sequences, contribs, peaks_df, _ = load_regions_npz(str(regions_npz))
    if contribs.ndim == 3:
        regions = contribs * sequences
    else:
        regions = contribs[:, None, :] * sequences
    peak_row_index = build_peak_row_index(peaks_df)
    peak_region_starts = peaks_df["peak_region_start"].to_numpy()

    trim_coords = load_mapping_tuple(str(trim_coords_path), int) if trim_coords_path and trim_coords_path.exists() else None
    if trim_coords_path is not None and trim_coords is None and args.verbose:
        print(f"Note: {trim_coords_path} not found, falling back to default cwm_trim_threshold for all motifs")

    motifs_df, cwms_modisco, trim_masks, _ = load_modisco_motifs(
        str(modisco_h5), trim_coords, None, DEFAULT_CWM_TRIM_THRESHOLD,
        "cwm", None, None, None, 1.0, True,
    )
    motif_width = cwms_modisco.shape[2]

    report_df = None
    if args.report_tsv:
        report_path = Path(args.report_tsv)
        if report_path.exists():
            report_df = pd.read_csv(report_path, sep="\t").set_index("motif_name")["cwm_similarity"]
        else:
            print(f"Warning: --report-tsv {report_path} not found, skipping", file=sys.stderr)

    rows = []
    for motif_name, group in hits.groupby("motif_name", sort=False):
        if len(group) < args.min_hits:
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
        # Excess, not a quotient: bg_ratio_modisco is often genuinely ~0 for a
        # cleanly-discovered motif, and dividing by a near-zero denominator
        # (found by hand on synthetic data) blows a ratio-of-ratios up into a
        # meaningless, unbounded number. A difference degrades gracefully --
        # excess ~= bg_ratio_hits when the archetype's own background truly
        # is ~0 (the common case), and nets out a motif's own legitimately
        # nonzero baseline background when it isn't.
        excess = bg_ratio_hits - bg_ratio_modisco

        row = dict(
            motif_name=motif_name, n_hits=len(group), excess=excess,
            bg_ratio_hits=bg_ratio_hits, bg_ratio_modisco=bg_ratio_modisco,
        )
        if report_df is not None:
            row["cwm_similarity"] = report_df.get(motif_name, float("nan"))
        rows.append(row)

    if not rows:
        print("No motif had enough hits to compute a ratio.", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: -r["excess"])
    header = f"{'motif_name':<28} {'n_hits':>8} {'bg_ratio(hits_fc)':>18} {'bg_ratio(modisco_fc)':>21} {'excess':>10}"
    if report_df is not None:
        header += f" {'cwm_similarity':>14}"
    print(header)
    for r in rows:
        line = (
            f"{r['motif_name']:<28} {r['n_hits']:>8} {r['bg_ratio_hits']:>18.4f} "
            f"{r['bg_ratio_modisco']:>21.4f} {r['excess']:>10.4f}"
        )
        if report_df is not None:
            line += f" {r['cwm_similarity']:>14.4f}"
        print(line)


if __name__ == "__main__":
    main()
