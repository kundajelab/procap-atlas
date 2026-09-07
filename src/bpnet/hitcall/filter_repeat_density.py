"""Drop Fi-NeMo hits that sit in dense same-motif clusters (repeat artifacts).

A real TF/core-promoter-element binding site is called once, maybe twice, per
peak. Some motifs -- most visibly the TATA box and TA-Initiator in K562
ENCSR220XSM's profile-head hits, reviewed directly against real report
output -- get called dozens to 100+ times within a single ~2kb peak, always
in degenerate poly-A/T repeat context. That inflates their hit count 2-3+
orders of magnitude beyond the ~1-2 orders of magnitude increase over
TF-MoDISco seqlet counts Fi-NeMo hit calling normally produces, and (for
motifs that also fail the whole-motif `cwm_similarity` QC gate in
report_bpnet.py) currently means losing every hit for that motif, including
real non-repeat instances, rather than just the repeat-driven noise.

This is deliberately motif-identity-agnostic (no hardcoded pattern names/
consensus/JASPAR matching): it only looks at hit density -- N or more hits of
the *same* motif_name within a `--cluster-window` bp span, inside the same
peak -- and drops all hits in that dense span, following Kelly Cochran's
ProCapNet TATA-repeat filter (drop TATA/TATATA hits when 5+ cluster within
80bp: https://github.com/kellycochran/procapnet_allscripts). Being
identity-agnostic is required here: a compendium/local motif's identity
varies per experiment (different pattern indices, no consistent labeling
available -- JASPAR doesn't cover core-promoter elements like Inr/TATA
anyway), so this has to run the same way for every experiment without being
told in advance which motif is which.

Run on hits_unique.tsv, before report_bpnet.py's cwm_similarity QC (not
after): report_bpnet.py's `finemo report -H` accepts a direct hits TSV path
(deprecated but functional Fi-NeMo behavior) in place of the call-hits output
directory, so pointing it at hits_dedensified.tsv recomputes cwm_similarity
against the cleaned hit set -- letting a motif like TATA/TA-Inr, whose
aggregate cwm_similarity is dragged down by repeat noise, potentially clear
the QC threshold on its remaining real hits instead of being dropped
wholesale.

Usage:
    python src/bpnet/hitcall/filter_repeat_density.py -e ENCSR882DWM
    python src/bpnet/hitcall/filter_repeat_density.py -e ENCSR882DWM --head count
    python src/bpnet/hitcall/filter_repeat_density.py -e ENCSR882DWM --min-trim-len 6
    python src/bpnet/hitcall/filter_repeat_density.py -e ENCSR882DWM --min-cluster-hits 5 --cluster-window 80
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, trim_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MIN_CLUSTER_HITS = 5
DEFAULT_CLUSTER_WINDOW = 80


def find_dense_hits(starts, min_cluster_hits, cluster_window):
    """Mark positions (assumed sorted ascending) that belong to some span of
    `cluster_window` bp containing at least `min_cluster_hits` hits.

    Two-pointer sliding window: as `right` advances, `left` is pulled forward
    just enough to keep the window within `cluster_window` bp; any window
    that reaches `min_cluster_hits` marks every position in it (not just the
    exact triggering window), so a whole dense stretch ends up flagged.
    """
    n = len(starts)
    dense = np.zeros(n, dtype=bool)
    left = 0
    for right in range(n):
        while starts[right] - starts[left] > cluster_window:
            left += 1
        if right - left + 1 >= min_cluster_hits:
            dense[left : right + 1] = True
    return dense


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
        "--min-cluster-hits",
        type=int,
        default=DEFAULT_MIN_CLUSTER_HITS,
        metavar="N",
        help=(
            "drop all hits of a motif within any span of --cluster-window bp "
            "containing this many or more same-motif hits in the same peak "
            "(default: 5, Kelly Cochran's ProCapNet TATA-repeat filter)"
        ),
    )
    parser.add_argument(
        "--cluster-window",
        type=int,
        default=DEFAULT_CLUSTER_WINDOW,
        metavar="BP",
        help="span (bp) hit density is evaluated over (default: 80)",
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

    hits_path = hits_dir / "hits_unique.tsv"
    if not hits_path.exists():
        print(f"Error: hits not found: {hits_path}", file=sys.stderr)
        print("Run call_hits_bpnet.py first.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Reading hits from {hits_path}")

    hits = pd.read_csv(hits_path, sep="\t")
    hits = hits.sort_values(["peak_id", "motif_name", "start"]).reset_index(drop=True)

    drop_mask = np.zeros(len(hits), dtype=bool)
    for _, idx in hits.groupby(["peak_id", "motif_name"], sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) < args.min_cluster_hits:
            continue
        starts = hits.loc[idx, "start"].to_numpy()
        dense = find_dense_hits(starts, args.min_cluster_hits, args.cluster_window)
        drop_mask[idx] = dense

    dropped = hits[drop_mask]
    kept = hits[~drop_mask]

    if len(dropped):
        per_motif = (
            dropped.groupby("motif_name")
            .size()
            .sort_values(ascending=False)
            .rename("n_dropped")
        )
        totals = hits.groupby("motif_name").size().rename("n_total")
        summary = pd.concat([per_motif, totals], axis=1, join="inner")
        summary["frac_dropped"] = summary["n_dropped"] / summary["n_total"]
        print(
            f"Dropped {len(dropped)}/{len(hits)} hits from dense same-motif "
            f"clusters (>= {args.min_cluster_hits} hits within "
            f"{args.cluster_window}bp) across {len(summary)} motif(s):"
        )
        for motif_name, row in summary.iterrows():
            print(
                f"  {motif_name}: {int(row['n_dropped'])}/{int(row['n_total'])} "
                f"hits dropped ({row['frac_dropped']:.1%})"
            )
    else:
        print("No dense same-motif clusters found; nothing dropped.")

    out_path = hits_dir / "hits_dedensified.tsv"
    kept.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {len(kept)} hits to {out_path}")


if __name__ == "__main__":
    main()
