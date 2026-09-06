"""Compute a minimum motif-trim-length floor for Fi-NeMo hit calling.

Fi-NeMo's CWM trimming (`-t/--cwm-trim-threshold`) has no minimum-length
safeguard: a motif's trimmed span is bounded only by the outermost positions
whose per-position summed |contribution| clears `trim_threshold *
max(|contribution|)`, with no padding. Kelly Cochran's ProCapNet
run_finemo.py patched this with a `min_len=6` floor specifically because
Initiator (Inr) elements are very short and can otherwise be trimmed down to
just 1-2bp:
https://github.com/kellycochran/procapnet_allscripts/blob/main/GENCODE/src/attributions_genomewide/run_finemo.py

This script replicates Fi-NeMo's own `trim_motif` (imported directly from the
installed `finemo` package, so it always matches whatever trimming algorithm
`finemo call-hits` actually runs), widens -- symmetrically, clamped to the
untrimmed motif width -- any motif trimmed below `--min-len`, and writes a
`-R/--cwm-trim-coords`-compatible TSV containing only the motifs that needed
widening. Everything else is left out of the file, so it keeps using
Fi-NeMo's own default threshold-based trimming.

Pass `-e/--experiment` to compute the floor against that experiment's own
`modisco/bpnet/{experiment}_{head}.modisco.h5` (matching call_hits_bpnet.py's
default per-experiment motif source); without it, defaults to the shared
atlas-wide MotifCompendium cluster-average set instead (for use with
`--modisco-h5` overrides pointed at the compendium).

Usage:
    python src/bpnet/hitcall/compute_trim_floor.py -e ENCSR882DWM --head profile
    python src/bpnet/hitcall/compute_trim_floor.py -e ENCSR882DWM --head count --min-len 8
    python src/bpnet/hitcall/compute_trim_floor.py --head profile  # atlas-wide MotifCompendium set instead
    python src/bpnet/hitcall/compute_trim_floor.py --modisco-h5 modisco/bpnet/ENCSR882DWM_profile.modisco.h5 --head profile
    python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM --cwm-trim-coords modisco/bpnet/ENCSR882DWM_profile_trim_coords_min6bp.tsv
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from finemo.data_io import trim_motif

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODISCO_PATTERN_GROUPS = ["pos_patterns", "neg_patterns"]


def widen_to_min_len(start, end, width, min_len):
    """Symmetrically widen [start, end) to at least min_len, clamped to [0, width).

    Mirrors Kelly Cochran's ProCapNet trim_motif() min_len while-loop.
    """
    while end - start < min_len and (start > 0 or end < width):
        start = max(start - 1, 0)
        end = min(end + 1, width)
    return start, end


def sort_key(pattern_name):
    return (0, int(pattern_name)) if pattern_name.isdigit() else (1, pattern_name)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default=None,
        help=(
            "compute the floor against this experiment's own "
            "modisco/bpnet/{experiment}_{head}.modisco.h5, matching "
            "call_hits_bpnet.py's default per-experiment motif source. "
            "Without this, defaults to the shared atlas-wide MotifCompendium "
            "cluster-average set instead."
        ),
    )
    parser.add_argument(
        "--head",
        type=str,
        default="profile",
        choices=["profile", "count"],
        help="motif head to compute the floor for (default: profile)",
    )
    parser.add_argument(
        "--modisco-h5",
        type=str,
        default=None,
        help=(
            "override the modisco-lite-format h5 of motif CWMs to inspect "
            "directly, taking precedence over --experiment"
        ),
    )
    parser.add_argument(
        "--cwm-trim-threshold",
        type=float,
        default=0.3,
        help="trim threshold to replicate (default: 0.3, matching "
        "call_hits_bpnet.py's default)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=6,
        help="minimum trimmed motif length in bp (default: 6, Kelly Cochran's "
        "ProCapNet Inr floor)",
    )
    parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        default=None,
        help=(
            "output TSV path for the -R/--cwm-trim-coords override; default "
            "is alongside the input h5, "
            "{experiment}_{head}_trim_coords_min{min_len}bp.tsv (per-experiment "
            "mode) or motifcompendium_{head}_trim_coords_min{min_len}bp.tsv "
            "(atlas-wide mode)"
        ),
    )
    args = parser.parse_args()

    if args.modisco_h5:
        modisco_h5 = Path(args.modisco_h5)
        name_stem = f"{args.head}"
    elif args.experiment:
        modisco_h5 = (
            REPO_ROOT
            / "modisco"
            / "bpnet"
            / f"{args.experiment}_{args.head}.modisco.h5"
        )
        name_stem = f"{args.experiment}_{args.head}"
    else:
        modisco_h5 = (
            REPO_ROOT
            / "motifcompendium"
            / "bpnet"
            / f"motifcompendium_{args.head}_cluster_averages.h5"
        )
        name_stem = f"motifcompendium_{args.head}"
    if not modisco_h5.exists():
        print(f"Error: motif CWMs not found: {modisco_h5}", file=sys.stderr)
        sys.exit(1)

    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = (
            modisco_h5.parent / f"{name_stem}_trim_coords_min{args.min_len}bp.tsv"
        )

    widened = []
    with h5py.File(modisco_h5, "r") as f:
        for group_name in MODISCO_PATTERN_GROUPS:
            if group_name not in f:
                continue
            for pattern_name in sorted(f[group_name].keys(), key=sort_key):
                pattern_tag = f"{group_name}.{pattern_name}"
                cwm = f[group_name][pattern_name]["contrib_scores"][:].T
                width = cwm.shape[1]

                start, end = trim_motif(cwm, args.cwm_trim_threshold)
                new_start, new_end = widen_to_min_len(
                    start, end, width, args.min_len
                )
                if (new_start, new_end) != (start, end):
                    widened.append(
                        (pattern_tag, start, end, new_start, new_end, width)
                    )

    with open(out_path, "w") as out_f:
        for pattern_tag, _, _, new_start, new_end, _ in widened:
            out_f.write(f"{pattern_tag}\t{new_start}\t{new_end}\n")

    print(
        f"{args.head}: {len(widened)} motif(s) widened to a {args.min_len}bp "
        f"floor (threshold {args.cwm_trim_threshold}):"
    )
    for pattern_tag, start, end, new_start, new_end, width in widened:
        print(
            f"  {pattern_tag}: [{start}, {end}) len={end - start} -> "
            f"[{new_start}, {new_end}) len={new_end - new_start} "
            f"(untrimmed width {width})"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
