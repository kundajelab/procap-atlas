#!/usr/bin/env python3
"""How much of the MotifCompendium lexicon is the same motif counted twice?

Every count derived from the compendium -- lexicon size, rarefaction curves,
the number of tissue-restricted clusters -- is inflated if one real motif got
split across several clusters because `cluster_motifs.py --across-threshold`
failed to merge them. Unlike the other caveats in this directory, redundancy
biases in the *flattering* direction, so it is the one a referee is most likely
to find first.

JASPAR name collisions hint at the scale but cannot measure it: on the real
count-head compendium 306 clusters carry only 112 distinct JASPAR names (SP9
claimed by 31 clusters, TBP by 15), yet JASPAR annotation is a nearest-neighbour
lookup, so a bare GC-box and a GC-box with an ETS half-site can both best-match
SP9 while being genuinely different motifs. This script measures redundancy
directly, by comparing the cluster-average CWMs against each other.

Uses TOMTOM from `memelite` (the "tomtom-lite" reimplementation), which is a
Python API with no command-line entry point -- hence a script rather than a
shell command. Self-comparison of the compendium's own MEME export, diagonal
removed.

Redundancy is reported as *excess clusters*: build a graph joining every pair
of clusters whose alignment passes the significance and overlap filters, take
connected components, and report `n_clusters - n_components`. That is the number
of clusters that would disappear if every near-duplicate group collapsed to one
motif.

No single p-value threshold is defensible -- significance depends on motif
length and information content, and family members are genuinely similar
without being duplicates -- so the output is a sweep across thresholds, the
same treatment the abundance floor gets in plot_motif_rarefaction.py. Read the
shape: if excess is flat across several orders of magnitude, redundancy is well
determined; if it climbs steadily, the lexicon size is threshold-dependent and
should be quoted as a range.

One footgun worth knowing: TOMTOM scores columns against a background
estimated from the target set, so near-deterministic PWMs (a one-hot consensus
with epsilon elsewhere) make that background degenerate and can return p = 1.0
for two *identical* motifs. The compendium's cluster-average CWMs are soft
enough that this does not arise in practice, but it does mean synthetic
one-hot fixtures cannot be used to test the p-value path.

`--min-overlap-frac` additionally requires the best alignment to cover that
fraction of the shorter motif, which suppresses the case where a short motif
aligns significantly inside a longer unrelated one.

Outputs (in --out-dir):
  motif_redundancy_{head}_pairs.tsv       every passing pair at the loosest threshold
  motif_redundancy_{head}_summary.tsv     excess clusters per threshold
  motif_redundancy_{head}_components.tsv  cluster -> merged component, at --report-threshold
  motif_redundancy_{head}.{png,pdf}       excess vs threshold

Usage:
    python src/analysis/motif_redundancy.py --head count
    python src/analysis/motif_redundancy.py --head count --report-threshold 1e-6
    python src/analysis/motif_redundancy.py --head count --min-overlap-frac 0.8
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet"
DEFAULT_THRESHOLDS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
NAME_RE = re.compile(r"((?:pos|neg)_patterns\.\d+)")


def parse_motif_name(key: str, name_regex: re.Pattern = NAME_RE) -> str:
    """Recover a cluster identifier from a MEME metadata line.

    `memelite.io.read_meme` keys its dict on the whole metadata line, whose
    exact shape depends on how MotifCompendium's exporter wrote it, so the
    compendium-style identifier is extracted when present and the raw key kept
    otherwise -- never silently dropped.
    """
    match = name_regex.search(key)
    if match:
        return match.group(1)
    return key.replace("MOTIF", "").strip().split()[0] if key.strip() else key


def load_motifs(meme_path: Path, name_regex: re.Pattern) -> tuple[list[str], list[np.ndarray]]:
    from memelite.io import read_meme

    motifs = read_meme(str(meme_path))
    names, pwms = [], []
    for key, pwm in motifs.items():
        names.append(parse_motif_name(key, name_regex))
        arr = np.asarray(pwm, dtype=np.float64)
        # memelite expects (alphabet, length); MEME files are (length, alphabet)
        # in the file itself, but read_meme already returns the torch layout.
        # Guard anyway: the alphabet axis is the short one for real motifs.
        if arr.ndim != 2:
            raise ValueError(f"motif {key!r} is not 2-D: shape {arr.shape}")
        if arr.shape[0] != 4 and arr.shape[1] == 4:
            arr = arr.T
        pwms.append(arr)
    return names, pwms


def self_compare(pwms: list[np.ndarray], n_jobs: int) -> dict[str, np.ndarray]:
    """Run TOMTOM of every cluster against every other, diagonal masked out."""
    from memelite import tomtom

    p, scores, offsets, overlaps, strands = tomtom(pwms, pwms, n_jobs=n_jobs)
    p = np.asarray(p, dtype=np.float64).copy()
    np.fill_diagonal(p, np.inf)  # a motif is not its own duplicate
    return {
        "p": p,
        "scores": np.asarray(scores),
        "offsets": np.asarray(offsets),
        "overlaps": np.asarray(overlaps),
        "strands": np.asarray(strands),
    }


def build_pairs(
    names: list[str],
    pwms: list[np.ndarray],
    res: dict[str, np.ndarray],
    p_threshold: float,
    min_overlap_frac: float,
) -> pd.DataFrame:
    """Upper-triangle pairs passing the p-value and overlap filters."""
    lengths = np.array([m.shape[-1] for m in pwms])
    p = res["p"]
    iu = np.triu_indices(len(names), k=1)
    # Symmetrize on the more conservative (larger) p-value: TOMTOM is not
    # symmetric, since the query's background distribution sets the scale.
    p_sym = np.maximum(p[iu], p.T[iu])
    overlap = res["overlaps"][iu]
    shorter = np.minimum(lengths[iu[0]], lengths[iu[1]])
    frac = np.divide(overlap, shorter, out=np.zeros_like(overlap, dtype=float),
                     where=shorter > 0)
    keep = (p_sym <= p_threshold) & (frac >= min_overlap_frac)
    return pd.DataFrame(
        {
            "motif_a": [names[i] for i in iu[0][keep]],
            "motif_b": [names[j] for j in iu[1][keep]],
            "p_value": p_sym[keep],
            "overlap": overlap[keep],
            "overlap_frac": np.round(frac[keep], 3),
            "len_a": lengths[iu[0][keep]],
            "len_b": lengths[iu[1][keep]],
            "strand": res["strands"][iu][keep],
        }
    ).sort_values("p_value")


def connected_components(names: list[str], pairs: pd.DataFrame) -> dict[str, int]:
    """Union-find over passing pairs; component id per motif name."""
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["motif_a"], pairs["motif_b"]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    roots = {}
    out = {}
    for n in names:
        r = find(n)
        out[n] = roots.setdefault(r, len(roots))
    return out


def sweep(
    names: list[str],
    pwms: list[np.ndarray],
    res: dict[str, np.ndarray],
    thresholds,
    min_overlap_frac: float,
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        pairs = build_pairs(names, pwms, res, t, min_overlap_frac)
        comps = connected_components(names, pairs)
        n_comp = len(set(comps.values()))
        rows.append(
            {
                "p_threshold": t,
                "n_pairs": len(pairs),
                "n_clusters": len(names),
                "n_components": n_comp,
                "excess_clusters": len(names) - n_comp,
                "excess_frac": round((len(names) - n_comp) / len(names), 4),
                "largest_component": (
                    int(pd.Series(list(comps.values())).value_counts().iloc[0])
                    if names else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_sweep(summary: pd.DataFrame, head: str, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(summary["p_threshold"], summary["excess_frac"], "o-", color="#b2182b", lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("TOMTOM p-value threshold")
    ax.set_ylabel("Excess clusters (fraction of lexicon)")
    ax.set_ylim(0, max(0.05, summary["excess_frac"].max() * 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"Compendium redundancy — {head} head\n"
        f"{int(summary['n_clusters'].iloc[0])} clusters",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}", file=sys.stderr)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--head", default="count", choices=["profile", "count"],
        help="which compendium to self-compare (default: count)",
    )
    parser.add_argument(
        "--meme", type=Path, default=None, metavar="PATH",
        help="override motifcompendium_{head}_cluster_averages.meme",
    )
    parser.add_argument(
        "--cluster-metadata", type=Path, default=None, metavar="PATH",
        help="join JASPAR names onto components, to test whether name "
             "collisions correspond to real CWM redundancy",
    )
    parser.add_argument(
        "--p-thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
        metavar="P", help="thresholds to sweep (default: 1e-12 ... 1e-2)",
    )
    parser.add_argument(
        "--report-threshold", type=float, default=1e-6, metavar="P",
        help="threshold whose components and pairs are written out (default: 1e-6)",
    )
    parser.add_argument(
        "--min-overlap-frac", type=float, default=0.7, metavar="F",
        help="require the best alignment to cover this fraction of the shorter "
             "motif, suppressing short-inside-long matches (default: 0.7)",
    )
    parser.add_argument(
        "--name-regex", default=NAME_RE.pattern, metavar="RE",
        help="regex extracting a cluster id from a MEME metadata line",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="TOMTOM parallelism (default: -1)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figures" / "motif_atlas",
        metavar="DIR", help="output directory (default: figures/motif_atlas/)",
    )
    args = parser.parse_args()

    meme_path = args.meme or (MC_DIR / f"motifcompendium_{args.head}_cluster_averages.meme")
    if not meme_path.exists():
        print(f"ERROR: MEME file not found: {meme_path}", file=sys.stderr)
        print(
            "Run src/bpnet/motifcompendium/cluster_motifs.py --head "
            f"{args.head} first.",
            file=sys.stderr,
        )
        sys.exit(1)

    names, pwms = load_motifs(meme_path, re.compile(args.name_regex))
    if len(names) < 2:
        print(f"ERROR: only {len(names)} motif(s) in {meme_path}", file=sys.stderr)
        sys.exit(1)
    widths = np.array([m.shape[-1] for m in pwms])
    print(
        f"{args.head}: {len(names)} clusters, width {widths.min()}-{widths.max()} "
        f"(median {int(np.median(widths))})",
        file=sys.stderr,
    )

    res = self_compare(pwms, args.n_jobs)
    summary = sweep(names, pwms, res, args.p_thresholds, args.min_overlap_frac)

    thresholds = sorted(set(args.p_thresholds) | {args.report_threshold})
    pairs = build_pairs(names, pwms, res, max(thresholds), args.min_overlap_frac)
    comps = connected_components(
        names, build_pairs(names, pwms, res, args.report_threshold, args.min_overlap_frac)
    )
    comp_df = pd.DataFrame(
        {"motif": names, "component": [comps[n] for n in names],
         "width": widths}
    ).sort_values(["component", "motif"])

    if args.cluster_metadata is not None and args.cluster_metadata.exists():
        meta = pd.read_csv(args.cluster_metadata, sep="\t")
        if {"cluster_final", "posneg"} <= set(meta.columns):
            key = (
                meta["posneg"].astype(str) + "_patterns."
                + meta["cluster_final"].astype(int).astype(str)
            )
            join = pd.DataFrame({"motif": key})
            for c in ("jaspar_name", "jaspar_score", "total_seqlets", "n_experiments"):
                if c in meta.columns:
                    join[c] = meta[c].values
            comp_df = comp_df.merge(join, on="motif", how="left")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"motif_redundancy_{args.head}"
    pairs.to_csv(stem.parent / f"{stem.name}_pairs.tsv", sep="\t", index=False)
    summary.to_csv(stem.parent / f"{stem.name}_summary.tsv", sep="\t", index=False)
    comp_df.to_csv(stem.parent / f"{stem.name}_components.tsv", sep="\t", index=False)
    for suffix in ("pairs", "summary", "components"):
        print(f"Saved {stem.parent / f'{stem.name}_{suffix}.tsv'}", file=sys.stderr)
    plot_sweep(summary, args.head, stem)

    with pd.option_context("display.width", 200):
        print("\nRedundancy sweep:", file=sys.stderr)
        print(summary.to_string(index=False), file=sys.stderr)

    at = summary[summary["p_threshold"] == args.report_threshold]
    if not at.empty:
        row = at.iloc[0]
        print(
            f"\nAt p <= {args.report_threshold:g} and overlap >= "
            f"{args.min_overlap_frac:g}: {int(row['n_clusters'])} clusters collapse "
            f"to {int(row['n_components'])} ({int(row['excess_clusters'])} excess, "
            f"{row['excess_frac']:.1%}); largest merged group "
            f"{int(row['largest_component'])}.",
            file=sys.stderr,
        )

    if "jaspar_name" in comp_df.columns:
        named = comp_df.dropna(subset=["jaspar_name"])
        multi = named.groupby("component")["motif"].size()
        merged = named[named["component"].isin(multi[multi > 1].index)]
        if not merged.empty:
            agree = (
                merged.groupby("component")["jaspar_name"].nunique() == 1
            ).mean()
            print(
                f"\nOf merged components with >1 JASPAR-named cluster, "
                f"{agree:.0%} are internally consistent in JASPAR name.\n"
                "  High agreement => name collisions really were redundancy.\n"
                "  Low agreement  => TOMTOM is merging distinct family members, "
                "so loosen the threshold with care.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
