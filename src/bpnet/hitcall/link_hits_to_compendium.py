"""Relabel per-experiment Fi-NeMo hits with their atlas-wide MotifCompendium
cluster identity.

call_hits_bpnet.py calls hits against each experiment's own per-experiment
MoDISco motif set by default, so a hit's `motif_name` (e.g.
`pos_patterns.pattern_3`) is only meaningful within that experiment -- the
same string means a different motif in a different experiment. This script
adds a `compendium_motif_name` column (e.g. `pos_patterns.42`) giving the
atlas-wide MotifCompendium cluster each local motif was assigned to, via the
mapping table `src/bpnet/motifcompendium/cluster_motifs.py` exports
(`motifcompendium_{head}_pattern_to_cluster.tsv`), so hits become
cross-experiment comparable without having had to pay for calling hits
against the full atlas-wide motif set in the first place.

Requires cluster_motifs.py to have already been run for the requested head
(it builds the mapping from every experiment's own per-experiment motifs, so
it must be rerun whenever new experiments are added). Also requires
call_hits_bpnet.py to have been run for this experiment/head with the
default (per-experiment) motif source -- not --modisco-h5 pointed at the
compendium.

Usage:
    python src/bpnet/hitcall/link_hits_to_compendium.py -e ENCSR882DWM
    python src/bpnet/hitcall/link_hits_to_compendium.py -e ENCSR882DWM --head count
    python src/bpnet/hitcall/link_hits_to_compendium.py -e ENCSR882DWM --min-trim-len 6
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, trim_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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
        "--mapping-tsv",
        type=str,
        default=None,
        help=(
            "override the pattern-to-cluster mapping TSV; default is "
            "motifcompendium/bpnet/motifcompendium_{head}_pattern_to_cluster.tsv "
            "(cluster_motifs.py output)"
        ),
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

    # Prefer the cwm_similarity-filtered hits (report_bpnet.py) if they
    # exist; fall back to the raw deduplicated hits otherwise.
    hits_filtered_path = hits_dir / "hits_filtered.tsv"
    hits_unique_path = hits_dir / "hits_unique.tsv"
    if hits_filtered_path.exists():
        hits_path = hits_filtered_path
    else:
        hits_path = hits_unique_path

    if args.mapping_tsv:
        mapping_path = Path(args.mapping_tsv)
    else:
        mapping_path = (
            REPO_ROOT
            / "motifcompendium"
            / "bpnet"
            / f"motifcompendium_{args.head}_pattern_to_cluster.tsv"
        )

    for path, label in [(hits_path, "hits"), (mapping_path, "pattern-to-cluster mapping")]:
        if not path.exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            if label == "hits":
                print("Run call_hits_bpnet.py first.", file=sys.stderr)
            else:
                print(
                    "Run src/bpnet/motifcompendium/cluster_motifs.py first.",
                    file=sys.stderr,
                )
            sys.exit(1)

    if args.verbose:
        print(f"Reading hits from {hits_path}")
        print(f"Reading mapping from {mapping_path}")

    hits = pd.read_csv(hits_path, sep="\t")
    mapping = pd.read_csv(mapping_path, sep="\t")
    mapping = mapping[mapping["experiment"] == args.experiment][
        ["local_motif_name", "compendium_motif_name"]
    ]

    linked = hits.merge(
        mapping,
        left_on="motif_name",
        right_on="local_motif_name",
        how="left",
    ).drop(columns=["local_motif_name"])

    n_unmapped = int(linked["compendium_motif_name"].isna().sum())
    if n_unmapped:
        unmapped_motifs = sorted(
            linked.loc[linked["compendium_motif_name"].isna(), "motif_name"].unique()
        )
        print(
            f"WARNING: {n_unmapped}/{len(linked)} hits have no compendium mapping "
            f"(motif(s) {unmapped_motifs} missing from {mapping_path} for "
            f"experiment={args.experiment} -- cluster_motifs.py may need to be "
            "rerun to include this experiment).",
            file=sys.stderr,
        )

    out_path = hits_dir / "hits_linked.tsv"
    linked.to_csv(out_path, sep="\t", index=False)

    n_compendium_motifs = linked["compendium_motif_name"].nunique()
    print(
        f"Linked {len(linked)} hits ({hits_path.name}) across "
        f"{hits['motif_name'].nunique()} local motifs to "
        f"{n_compendium_motifs} compendium clusters"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
