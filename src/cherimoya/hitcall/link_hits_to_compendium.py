"""Relabel per-experiment Fi-NeMo hits with their atlas-wide MotifCompendium
cluster identity.

Cherimoya-specific counterpart to src/bpnet/hitcall/link_hits_to_compendium.py.
Adds a `compendium_motif_name` column using the mapping table from
src/cherimoya/motifcompendium/cluster_motifs.py
(motifcompendium/cherimoya/motifcompendium_{head}_pattern_to_cluster.tsv).

Requires cluster_motifs.py to have already been run for the requested head,
and call_hits_cherimoya.py to have been run for this experiment/head.

Usage:
    python src/cherimoya/hitcall/link_hits_to_compendium.py -e ENCSR882DWM
    python src/cherimoya/hitcall/link_hits_to_compendium.py -e ENCSR882DWM --head count
    python src/cherimoya/hitcall/link_hits_to_compendium.py -e ENCSR882DWM --min-trim-len 6
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_BPNET_HITCALL_DIR = str(REPO_ROOT / "src" / "bpnet" / "hitcall")
if _BPNET_HITCALL_DIR not in sys.path:
    sys.path.insert(0, _BPNET_HITCALL_DIR)

import compressed_io
from call_hits_bpnet import resolve_hits_path

from call_hits_cherimoya import resolve_experiment_paths


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
            "motifcompendium/cherimoya/motifcompendium_{head}_pattern_to_cluster.tsv"
        ),
    )
    parser.add_argument(
        "--compendium-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="override the compendium directory (default: motifcompendium/cherimoya/)",
    )
    parser.add_argument(
        "--min-trim-len",
        type=int,
        default=None,
        metavar="BP",
        help="must match the value hitcall/launch.py was run with, if any",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _, hits_dir, _, _ = resolve_experiment_paths(
        args.experiment, args.head, args.min_trim_len, args.model_dir
    )

    hits_path = resolve_hits_path(hits_dir, verbose=args.verbose)

    if args.mapping_tsv:
        mapping_path = Path(args.mapping_tsv)
    else:
        mc_dir = args.compendium_dir or (REPO_ROOT / "motifcompendium" / "cherimoya")
        mapping_path = mc_dir / f"motifcompendium_{args.head}_pattern_to_cluster.tsv"

    if hits_path is None:
        print(f"Error: no hits found in {hits_dir}", file=sys.stderr)
        print("Run call_hits_cherimoya.py first.", file=sys.stderr)
        sys.exit(1)
    if not compressed_io.exists(mapping_path):
        print(f"Error: pattern-to-cluster mapping not found: {mapping_path}", file=sys.stderr)
        print("Run src/cherimoya/motifcompendium/cluster_motifs.py first.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Reading hits from {hits_path}")
        print(f"Reading mapping from {mapping_path}")

    hits = pd.read_csv(hits_path, sep="\t")
    mapping = pd.read_csv(compressed_io.resolve(mapping_path), sep="\t")
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
    out_path = compressed_io.write_tsv(linked, out_path)

    n_compendium_motifs = linked["compendium_motif_name"].nunique()
    print(
        f"Linked {len(linked)} hits ({hits_path.name}) across "
        f"{hits['motif_name'].nunique()} local motifs to "
        f"{n_compendium_motifs} compendium clusters"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
