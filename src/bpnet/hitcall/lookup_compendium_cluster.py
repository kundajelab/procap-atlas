#!/usr/bin/env python3
"""Resolve MotifCompendium cluster identity for known (experiment,
local motif name) pairs, and report whether they all land in the same
cluster -- the identification step needed before using
filter_low_confidence_hits.py --seqlet-compendium-clusters, which scopes
the hit_seqlet_confidence corroboration floor by verified compendium
identity rather than per-experiment pattern numbering or a cwm_similarity
threshold (needed for motifs like CA-Inr, whose short trimmed core makes
cwm_similarity structurally blind to its overcalling problem -- see
filter_low_confidence_hits.py's module docstring).

Reads the same motifcompendium_{head}_pattern_to_cluster.tsv mapping table
link_hits_to_compendium.py uses. Read-only, no data written.

Usage:
    python src/bpnet/hitcall/lookup_compendium_cluster.py \\
        --pair ENCSR220XSM:pos_patterns.pattern_1 \\
        --pair ENCSR083AMN:pos_patterns.pattern_3 \\
        --pair ENCSR083AMN:pos_patterns.pattern_6 \\
        --pair ENCSR342WAR:pos_patterns.pattern_2 \\
        --pair ENCSR331UGM:pos_patterns.pattern_1
    python src/bpnet/hitcall/lookup_compendium_cluster.py --head count --pair ENCSR882DWM:pos_patterns.pattern_0
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pair", type=str, action="append", required=True, metavar="EXPERIMENT:LOCAL_MOTIF_NAME",
        help="repeatable; each is an experiment ID and local motif_name joined by ':'",
    )
    parser.add_argument("--head", type=str, default="profile", choices=["profile", "count"])
    parser.add_argument("--mapping-tsv", type=str, default=None)
    args = parser.parse_args()

    mapping_path = (
        Path(args.mapping_tsv)
        if args.mapping_tsv
        else REPO_ROOT / "motifcompendium" / "bpnet" / f"motifcompendium_{args.head}_pattern_to_cluster.tsv"
    )
    if not mapping_path.exists():
        print(f"Error: {mapping_path} not found -- run src/bpnet/motifcompendium/cluster_motifs.py first", file=sys.stderr)
        sys.exit(1)

    mapping = pd.read_csv(mapping_path, sep="\t")

    resolved = []
    clusters_seen = set()
    for pair in args.pair:
        if ":" not in pair:
            print(f"Error: --pair {pair!r} must be EXPERIMENT:LOCAL_MOTIF_NAME", file=sys.stderr)
            sys.exit(1)
        experiment, local_motif_name = pair.split(":", 1)
        match = mapping[
            (mapping["experiment"] == experiment) & (mapping["local_motif_name"] == local_motif_name)
        ]
        if match.empty:
            print(f"{experiment}:{local_motif_name} -> NOT FOUND in {mapping_path}")
            continue
        cluster = match.iloc[0]["compendium_motif_name"]
        clusters_seen.add(cluster)
        resolved.append((experiment, local_motif_name, cluster))
        print(f"{experiment}:{local_motif_name} -> {cluster}")

    print(f"\n{len(resolved)}/{len(args.pair)} pair(s) resolved, {len(clusters_seen)} distinct cluster(s): {sorted(clusters_seen)}")
    if len(clusters_seen) == 1 and resolved:
        print(
            f"All resolved pairs share cluster {next(iter(clusters_seen))} -- "
            f"safe to pass as a single --seqlet-compendium-clusters value."
        )
    elif len(clusters_seen) > 1:
        print(
            "WARNING: these pairs map to different clusters -- MotifCompendium isn't "
            "clustering them together. Either pass all of these cluster IDs to "
            "--seqlet-compendium-clusters (repeatable), or treat this as a sign the "
            "compendium clustering itself needs review for this motif."
        )


if __name__ == "__main__":
    main()
