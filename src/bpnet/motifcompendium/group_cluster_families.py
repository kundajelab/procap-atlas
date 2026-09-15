"""Group compendium clusters into motif families, non-destructively.

The compendium relabels Fi-NeMo hits (`../hitcall/link_hits_to_compendium.py`)
rather than serving as the scan set, so near-duplicate clusters never compete
for a site and no hit is suppressed. What redundancy costs is **prevalence**:
if one experiment's TATA pattern links to cluster 8 and another's to cluster
21, then "cluster 8" is not the same motif atlas-wide, and any count of how
many experiments share a motif is split across the family.

That is the rarefaction analysis, which is **count-head only**. The profile
head is where the redundancy was characterised -- TATA spans at least 8
clusters there and AP-1 at least 7 -- because those are the families that
were spotted by eye, but no manuscript number comes from the profile head.
The count head is where a family roll-up changes a reported figure, and its
threshold has to be derived separately.

Either way this is a sequence-similarity answer to "how many of the clusters
are actually real", replacing the JASPAR-name collapse that answered it
before.

JASPAR names are the wrong granularity for this, rather than simply wrong.
A name is a TF-identity label; redundancy here is a question about CWM
similarity, and the two disagree in both directions. Names **split what
similarity fuses**: the ETV family is cluster 13 (ETV7) plus 35 and 10 (both
ELF2), two names but gaps of +0.030 and +0.007. Names **fuse what similarity
separates**: clusters 4 and 17 are both Hand1::Tcf3 yet sit at gap +0.053,
and are correctly kept apart at t=0.05. Some elements also have no JASPAR
entry at all -- CA-Inr is a core promoter initiator, so its clusters come
back as Hand1::Tcf3, ISL2 and Hand1::Tcf3 at 0.83-0.86 because the lookup
must return some TF. Where a canonical entry exists the names are right (TBP
for TATA, ETS for ETV), and distinguishing e.g. CRE from TRE variants is the
lookup working, not failing -- they are simply not the distinctions that
measure redundancy.

Output is a mapping, not a filter. Every cluster appears in
`cluster_family.tsv`; an analysis that wants non-redundant labels rolls up to
`family_rep`, and one that wants full granularity ignores the column. Nothing
is deleted, because some redundancy is acceptable where the motifs really do
differ.

Four design choices, each forced by a measurement rather than chosen:

  * **Motif-to-motif similarity, never similarity between cluster averages.**
    A label-permutation null on the count head had randomly-composed clusters
    scoring *higher* pairwise than real ones (60.6% vs 58.1% of nearest
    neighbours above 0.90), because averaging blurs toward a bland consensus
    and bland averages resemble each other. Similarity between averages is
    uncalibrated; motif-to-motif similarity is what Leiden optimised.
  * **Each pair judged against its own clusters' internal similarity:**
    `gap = min(within_a, within_b) - cross_ab`. A gap near zero means members
    of A resemble members of B as much as B's members resemble each other.
    Families differ in tightness -- the profile head's TATA clusters run
    within 0.966..0.985 -- so a flat similarity ceiling would merge some and
    miss others.
  * **Greedy representatives, not connected components.** Single-linkage
    percolates on these families, which are continua rather than cliques: on
    the profile head the largest component grew 4 -> 15 -> 48 -> 70 -> 137 as
    the threshold went 0.00 -> 0.05, and the group *count* peaked at 79
    (t=0.03) then fell to 58 (t=0.05) while coverage kept rising -- groups
    fusing into each other. Greedy cannot chain, because an absorbed cluster
    never absorbs others.
  * **Prevalence-descending order.** Greedy is order-dependent, but the two
    principled orders agree to within one cluster (564 by prevalence, 563 by
    size, against 600 for a random order). Prevalence also maximises
    absorption for the same guarantee, since the most reproducible cluster
    sits at a family's centre.

The threshold is head-specific and has to be derived, not assumed. On the
profile head, admissible values are pinned between AP-1's worst internal pair
(+0.046) and CA-Inr's tightest separable pair (+0.053), so 0.05 was used.
Pass `--check NAME=ID,ID,...` with a few families identified by eye to find
that window for another head.

    python src/bpnet/motifcompendium/group_cluster_families.py \\
        --head profile --threshold 0.05 \\
        --check TATA=21,8,48 --check AP-1=22,28,45,39 --check CA-Inr=4,15,17
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet"


def cluster_cross_matrix(similarity, codes, n_clusters):
    """Mean motif-to-motif similarity within and between every cluster pair.

    Args:
        similarity: (N, N) motif similarity with 1.0 on the diagonal.
        codes: (N,) cluster index in [0, n_clusters) per motif.
        n_clusters: number of clusters.

    Returns:
        `(cross, within)`. `cross` is (k, k), the mean over all member pairs.
        `within` is (k,) with each cluster's own N self-similarities removed,
        so it measures genuine internal coherence; a cluster of fewer than
        two motifs is `nan`.

    Computed as `H.T @ S @ H` for a one-hot `H`: one pass over the matrix
    instead of a Python loop over k^2 blocks.

    Removing the diagonal matters more than it looks. Leaving it in inflates
    `within` -- three members at pairwise 0.8 would read
    (6*0.8 + 3*1.0)/9 = 0.867 -- which widens every gap and makes real
    duplicates look separable. A singleton would read a perfect 1.0.
    """
    n_motifs = len(codes)
    onehot = np.zeros((n_motifs, n_clusters))
    onehot[np.arange(n_motifs), codes] = 1.0
    sums = onehot.T @ similarity @ onehot
    sizes = onehot.sum(axis=0)
    cross = sums / np.outer(sizes, sizes)

    diag = np.diagonal(sums).copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        within = (diag - sizes) / (sizes * sizes - sizes)
    within[sizes < 2] = np.nan
    return cross, within


def gap_matrix(cross, within):
    """`min(within_a, within_b) - cross_ab`, diagonal masked.

    The minimum rather than the mean: a loose cluster must not excuse a small
    gap to a tight one. `nan` wherever either cluster has no internal
    similarity to compare against.
    """
    gap = np.minimum.outer(within, within) - cross
    np.fill_diagonal(gap, np.nan)
    return gap


def greedy_representatives(gap, order, threshold):
    """Keep a cluster unless it is within `threshold` of one already kept.

    Args:
        gap: (k, k) gap matrix, diagonal `nan`.
        order: cluster indices, most-preferred representative first.
        threshold: a cluster is absorbed when its gap to some kept cluster is
            `<= threshold`.

    Returns:
        `(kept, rep)`. `kept` is the representative indices in the order they
        were chosen; `rep[i]` is the representative of cluster `i`, equal to
        `i` for representatives themselves.

    Absorbed clusters never absorb others, so unlike single-linkage this
    cannot chain A to C through an intermediate B. Each absorbed cluster goes
    to its *tightest* kept cluster, not the first one found, so the mapping
    does not depend on the iteration order of the kept list.
    """
    n_clusters = gap.shape[0]
    kept: list[int] = []
    rep = np.full(n_clusters, -1, dtype=int)
    for i in order:
        i = int(i)
        if kept:
            gaps = gap[i, kept]
            if not np.all(np.isnan(gaps)):
                nearest = int(np.nanargmin(gaps))
                if gaps[nearest] <= threshold:
                    rep[i] = kept[nearest]
                    continue
        kept.append(i)
        rep[i] = i
    return kept, rep


def tightest_kept_pair(gap, kept):
    """Smallest gap between any two representatives, or nan below two.

    This is the guarantee, and it has to be the *minimum*: an earlier version
    printed the maximum, which reports the most separable pair and is
    satisfied by any input at all.
    """
    if len(kept) < 2:
        return float("nan")
    return float(np.nanmin(gap[np.ix_(kept, kept)]))


def load_head(head, mc_dir, min_prevalence, min_size=2):
    """Clusters eligible for grouping, with their similarity submatrix.

    Clusters below `min_prevalence` are excluded from the analysis rather
    than grouped: a cluster seen in one experiment out of 219 has no
    prevalence to undercount, and on the profile head those are the large
    majority (5,529 clusters, of which 987 reach prevalence 2). They still
    appear in the output, mapped to themselves.
    """
    import MotifCompendium

    mc = MotifCompendium.load(
        str(mc_dir / f"motifcompendium_{head}_all_clustered.mc")
    )
    labels = mc.metadata["cluster_final"].to_numpy()
    prevalence = (
        pd.Series(mc.metadata["model"].to_numpy()).groupby(labels).nunique()
    )
    sizes = pd.Series(labels).value_counts()
    eligible = np.array(sorted(
        c for c in prevalence.index
        if prevalence[c] >= min_prevalence and sizes[c] >= min_size
    ))
    rows = np.flatnonzero(np.isin(labels, eligible))
    ids, codes = np.unique(labels[rows], return_inverse=True)
    similarity = mc.similarity[np.ix_(rows, rows)]
    all_ids = np.array(sorted(prevalence.index))
    return {
        "ids": ids,
        "codes": codes,
        "similarity": similarity,
        "prevalence": prevalence,
        "sizes": sizes,
        "all_ids": all_ids,
    }


def report_family(name, wanted, ids, cross, within, gap, sizes, prevalence):
    """Print within/cross/gap for a named family, plus nearby outsiders.

    The outsider list is what makes a threshold derivable: every family
    picked by eye off the report's first page turned out to undercount, since
    the rest of the family sits further down. TATA went from 3 clusters to at
    least 8 this way.
    """
    position = {int(c): i for i, c in enumerate(ids)}
    print(f"\n=== {name} ===")
    missing = [c for c in wanted if c not in position]
    if missing:
        print(f"  not eligible at this prevalence cut: {missing}")
    present = [c for c in wanted if c in position]
    if len(present) < 2:
        print("  fewer than two eligible clusters; nothing to compare")
        return
    rows = [position[c] for c in present]
    for c, i in zip(present, rows):
        print(f"  cluster {c:>5}  n={int(sizes[c]):>4}  "
              f"prev={int(prevalence[c]):>3}  within={within[i]:.3f}")
    worst = -np.inf
    for a, i in zip(present, rows):
        for b, j in zip(present, rows):
            if b <= a:
                continue
            worst = max(worst, gap[i, j])
            print(f"  {a:>5} vs {b:>5}  cross={cross[i, j]:.3f}  "
                  f"gap={gap[i, j]:+.3f}")
    for c, i in zip(present, rows):
        outside = [
            (int(ids[j]), cross[i, j], gap[i, j])
            for j in np.argsort(np.where(np.isnan(gap[i]), np.inf, gap[i]))
            if j not in rows and not np.isnan(gap[i, j]) and gap[i, j] <= worst
        ][:5]
        if outside:
            print(f"  outside {c}, gap <= {worst:+.3f}: " + ", ".join(
                f"{t}(cross={x:.3f}, gap={g:+.3f})" for t, x, g in outside
            ))


def build_table(data, rep, kept, within, gap):
    """One row per cluster in the build, grouped or not."""
    ids, prevalence, sizes = data["ids"], data["prevalence"], data["sizes"]
    position = {int(c): i for i, c in enumerate(ids)}
    kept_set = set(kept)
    rows = []
    for cluster in data["all_ids"]:
        cluster = int(cluster)
        i = position.get(cluster)
        if i is None:
            rows.append({
                "cluster_final": cluster,
                "n_motifs": int(sizes[cluster]),
                "n_experiments": int(prevalence[cluster]),
                "within": np.nan,
                "family_rep": cluster,
                "gap_to_rep": np.nan,
                "is_representative": True,
                "grouped": False,
            })
            continue
        r = int(rep[i])
        rows.append({
            "cluster_final": cluster,
            "n_motifs": int(sizes[cluster]),
            "n_experiments": int(prevalence[cluster]),
            "within": within[i],
            "family_rep": int(ids[r]),
            "gap_to_rep": np.nan if r == i else gap[i, r],
            "is_representative": i in kept_set,
            "grouped": True,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--head", default="profile",
                        choices=["count", "profile"])
    parser.add_argument("--mc-dir", type=Path, default=MC_DIR)
    parser.add_argument(
        "--threshold", type=float, default=None, metavar="G",
        help=(
            "absorb a cluster when its gap to a representative is <= G. "
            "Head-specific: derive it with --check before trusting a value. "
            "0.05 was used for the profile head, pinned between AP-1 "
            "(+0.046) and CA-Inr (+0.053)"
        ),
    )
    parser.add_argument(
        "--min-prevalence", type=int, default=2, metavar="N",
        help="group only clusters seen in >= N experiments (default: 2)",
    )
    parser.add_argument(
        "--check", action="append", default=None, metavar="NAME=ID,ID,...",
        help=(
            "report within/cross/gap for a family identified by eye, plus "
            "nearby clusters outside it. Repeatable. Use this to find the "
            "admissible threshold window for a head"
        ),
    )
    parser.add_argument(
        "--out-tsv", type=Path, default=None,
        help="default: <mc-dir>/motifcompendium_<head>_cluster_family.tsv",
    )
    args = parser.parse_args()

    data = load_head(args.head, args.mc_dir, args.min_prevalence)
    ids, codes = data["ids"], data["codes"]
    cross, within = cluster_cross_matrix(data["similarity"], codes, len(ids))
    gap = gap_matrix(cross, within)
    print(f"{args.head}: {len(ids)} clusters eligible at prevalence "
          f">= {args.min_prevalence}, {len(codes)} motifs "
          f"({len(data['all_ids'])} clusters in the build)")

    for spec in args.check or []:
        if "=" not in spec:
            parser.error(f"expected NAME=ID,ID,..., got {spec!r}")
        name, raw = spec.split("=", 1)
        report_family(name, [int(x) for x in raw.split(",")], ids, cross,
                      within, gap, data["sizes"], data["prevalence"])

    if args.threshold is None:
        print("\nNo --threshold given, so no families were grouped. Read the "
              "--check output above: the admissible window runs from the "
              "worst gap inside a family that should merge up to the "
              "tightest gap between clusters that should not.")
        return 0

    prevalence_of = data["prevalence"].loc[ids].to_numpy()
    order = np.argsort(-prevalence_of, kind="stable")
    kept, rep = greedy_representatives(gap, order, args.threshold)

    tightest = tightest_kept_pair(gap, kept)
    if not np.isnan(tightest) and tightest <= args.threshold:
        raise AssertionError(
            f"two representatives are {tightest:+.4f} apart, within the "
            f"{args.threshold} threshold; greedy selection is broken"
        )
    absorbed = len(ids) - len(kept)
    print(f"\nthreshold {args.threshold}: {len(kept)} families, "
          f"{absorbed} clusters absorbed ({absorbed / len(ids):.1%})")
    print(f"tightest pair among representatives: {tightest:+.4f} "
          f"(exceeds {args.threshold}, as required)")

    table = build_table(data, rep, kept, within, gap)
    out = args.out_tsv or (
        args.mc_dir / f"motifcompendium_{args.head}_cluster_family.tsv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, sep="\t", index=False)
    print(f"Saved {len(table)} clusters to {out}")

    multi = table[table.grouped].groupby("family_rep").size()
    multi = multi[multi > 1].sort_values(ascending=False)
    print(f"\n{len(multi)} families hold more than one cluster; "
          f"largest {int(multi.iloc[0]) if len(multi) else 0}")
    for family_rep, n in multi.head(10).items():
        members = table.loc[table.family_rep == family_rep, "cluster_final"]
        print(f"  rep {int(family_rep):>5} ({n:>2} clusters): "
              + ", ".join(str(int(c)) for c in members))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
