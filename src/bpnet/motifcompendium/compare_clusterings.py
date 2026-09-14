"""Compare two or more compendium builds' cluster assignments.

Written to answer a specific question: MotifCompendium v1.0.19 changed
`mc.cluster`'s default from `"cpm_leiden"` to `["cpm_leiden", "k_centroids"]`,
silently adding an uncapped k-means refinement stage (see
[`cluster_motifs.py`](cluster_motifs.py) `resolve_algorithm`). Before deciding
whether to drop that stage, bound it, or keep it, we need to know how much it
actually changes the partition.

Builds are compared on `motifcompendium_{head}_pattern_to_cluster.tsv`, not on
`cluster_metadata.tsv`. The metadata aggregates by cluster, and `cluster_final`
ids are **not stable across runs**, so metadata rows cannot be joined. The
pattern mapping keys on `(experiment, local_motif_name)`, which is the same
MoDISco pattern in every build, so partitions become comparable.

Reported:

  * Adjusted Rand Index and Adjusted Mutual Information on the shared
    patterns. These are invariant to cluster relabelling, which is required
    here since ids are arbitrary.
  * The fraction of patterns whose cluster *content* is unchanged, i.e. whose
    co-members are identical in both builds. Readable, but it amplifies, so
    report it next to ARI rather than alone: one pattern moving from cluster
    A to cluster B flags every member of both, so at a mean cluster size of
    ~6 a single reassignment marks up to ~12 patterns as changed. Dividing
    `n_patterns_moved` by the mean size of two clusters gives a lower bound
    on the number of actual reassignments.
  * Merge/split accounting: how many reference clusters map to one versus
    several clusters in the comparison build.

Example, for the three-way count-head comparison:

    python src/bpnet/motifcompendium/compare_clusterings.py --head count \\
        leiden=mc_leiden/motifcompendium_count_pattern_to_cluster.tsv \\
        capped5=mc_capped5/motifcompendium_count_pattern_to_cluster.tsv \\
        uncapped=motifcompendium/bpnet/motifcompendium_count_pattern_to_cluster.tsv
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

KEY = ["experiment", "local_motif_name"]


def load_assignments(path: Path) -> pd.Series:
    """cluster id per (experiment, local_motif_name), from a pattern mapping.

    `compendium_motif_name` is `f"{posneg}_patterns.pattern_{cluster_final}"`,
    so the cluster id is recovered from its suffix. posneg is kept in the key
    rather than discarded: a pos and a neg pattern can carry the same
    `cluster_final`, and collapsing them would silently merge two clusters.
    """
    df = pd.read_csv(path, sep="\t")
    missing = set(KEY + ["compendium_motif_name"]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    name = df["compendium_motif_name"].astype(str)
    posneg = name.str.split("_patterns", n=1).str[0]
    cluster = name.str.rsplit("_", n=1).str[-1]
    df["_cluster"] = posneg + ":" + cluster
    dup = df.duplicated(KEY).sum()
    if dup:
        raise ValueError(f"{path} has {dup} duplicate {KEY} rows")
    return df.set_index(KEY)["_cluster"]


def cluster_content_key(labels: pd.Series) -> pd.Series:
    """Map each pattern to a hashable of its co-members.

    Two builds agree on a pattern when its set of cluster-mates is identical,
    regardless of what the cluster is called. This is stricter than "same
    cluster id" and is the quantity a reader cares about.
    """
    members = labels.groupby(labels).apply(
        lambda s: frozenset(s.index)
    )
    return labels.map(members)


def compare_pair(a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> dict:
    shared = a.index.intersection(b.index)
    a_s, b_s = a.loc[shared], b.loc[shared]
    content_a = cluster_content_key(a_s)
    content_b = cluster_content_key(b_s)
    identical = (content_a == content_b)

    # merge/split accounting, in both directions
    per_a = b_s.groupby(a_s).nunique()
    per_b = a_s.groupby(b_s).nunique()
    return {
        "a": label_a,
        "b": label_b,
        "n_shared_patterns": len(shared),
        "n_clusters_a": a_s.nunique(),
        "n_clusters_b": b_s.nunique(),
        "frac_same_clustermates": round(float(identical.mean()), 4),
        "n_patterns_moved": int((~identical).sum()),
        "ari": round(float(adjusted_rand_score(a_s, b_s)), 4),
        "ami": round(float(adjusted_mutual_info_score(a_s, b_s)), 4),
        "a_clusters_split_in_b": int((per_a > 1).sum()),
        "b_clusters_split_in_a": int((per_b > 1).sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "builds", nargs="+", metavar="LABEL=PATH",
        help="pattern_to_cluster.tsv per build, labelled for the report",
    )
    parser.add_argument("--head", default="count", choices=["count", "profile"])
    parser.add_argument(
        "--out-tsv", type=Path, default=None, metavar="PATH",
        help="also write the pairwise table here",
    )
    args = parser.parse_args()

    if len(args.builds) < 2:
        parser.error("need at least two builds to compare")

    labelled = {}
    for spec in args.builds:
        if "=" not in spec:
            parser.error(f"expected LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        labelled[label] = load_assignments(Path(path))

    print(f"{args.head} head: {len(labelled)} builds")
    for label, series in labelled.items():
        print(f"  {label:12s} {len(series):7,} patterns  "
              f"{series.nunique():6,} clusters")

    rows = [compare_pair(labelled[a], labelled[b], a, b)
            for a, b in itertools.combinations(labelled, 2)]
    table = pd.DataFrame(rows)
    print()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))

    if args.out_tsv is not None:
        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_tsv, sep="\t", index=False)
        print(f"\nSaved {args.out_tsv}")

    mean_size = (
        sum(len(s) / s.nunique() for s in labelled.values()) / len(labelled)
    )
    print(
        "\nfrac_same_clustermates is the fraction of patterns whose set of "
        "cluster-mates is identical in both builds, invariant to cluster "
        "relabelling. It amplifies: moving one pattern between two clusters "
        f"flags every member of both, ~{2 * mean_size:.0f} patterns at this "
        f"mean cluster size of {mean_size:.1f}. So n_patterns_moved / "
        f"{2 * mean_size:.0f} is a lower bound on actual reassignments. Read "
        "it alongside ARI, not instead of it."
    )


if __name__ == "__main__":
    main()
