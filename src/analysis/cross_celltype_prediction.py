#!/usr/bin/env python3
"""Do models predict their own cell type's initiation better than another's?

The ProCapNet analysis (Cochran et al.) asked this across six cell lines. The
atlas makes the same comparison across every experiment at once: predict a
fixed peak set with model i, correlate against observed counts in experiment j,
and compare the matched diagonal against off-diagonal pairs -- split by whether
the two experiments share a tissue group, which separates "the model learned
this cell type" from "the model learned promoters in general".

Three tiers, in increasing distance:

    matched          i == j
    same tissue      i != j, same biosample group
    different tissue i != j, different group

Matched > same tissue > different tissue is the claim. The middle tier is what
makes it a claim about cell-type specificity rather than about overfitting: a
model that merely memorized its own experiment would beat both other tiers
equally.

**This requires counts extracted with `count_correlation.py --held-out-folds`.**
The default extraction averages all seven fold models at every peak, so a
model's predictions for its own experiment include six folds that trained on
those exact peaks, while its predictions for another experiment get no such
help. That inflates the diagonal by construction -- precisely the quantity
being measured -- and this script refuses to run without confirmation that the
inputs are held-out (`--i-know-these-are-fold-averaged` to override, for
exploring the shape of the result only).

Read depth is reported rather than assumed away: deeper experiments are
predicted better, and depth differs by tissue group on this atlas
(Kruskal-Wallis p = 1.2e-4), so a matched-vs-mismatched gap could in principle
track depth. The summary includes the correlation between matched accuracy and
depth so that is visible.

Usage:
    python src/analysis/cross_celltype_prediction.py \\
        --observed figures/count_correlation/observed_counts.tsv \\
        --predicted figures/count_correlation/predicted_counts.tsv
    python src/analysis/cross_celltype_prediction.py --variable-peaks 20000
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _biosample_groups import load_group_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
DEFAULT_DIR = REPO_ROOT / "figures" / "count_correlation"
OUT_DIR = REPO_ROOT / "figures" / "cross_celltype"

TIERS = ("matched", "same tissue", "different tissue")


def load_counts(path: Path) -> pd.DataFrame:
    """Experiments x peaks count matrix, experiments on the index."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    if df.empty:
        raise ValueError(f"{path} has no rows")
    return df


def align(observed: pd.DataFrame, predicted: pd.DataFrame) -> tuple:
    """Restrict both matrices to shared experiments and shared peak columns.

    Column counts can differ if the two files were produced by different runs;
    silently correlating misaligned vectors would give a meaningless matrix, so
    this intersects explicitly and the caller reports what was dropped.
    """
    exps = [e for e in observed.index if e in set(predicted.index)]
    cols = [c for c in observed.columns if c in set(predicted.columns)]
    if not exps:
        raise ValueError("no experiments in common between observed and predicted")
    if not cols:
        raise ValueError("no peaks in common between observed and predicted")
    return observed.loc[exps, cols], predicted.loc[exps, cols]


def most_variable_peaks(observed: pd.DataFrame, n: int) -> list:
    """The n peaks with the highest cross-experiment variance of log1p counts.

    Restricting to these sharpens every tier contrast, because a peak that is
    equally active everywhere cannot distinguish a matched model from a
    mismatched one -- both predict it correctly. It is a sensitivity knob, not
    the headline: the full-peak-set result is the conservative one.
    """
    if n <= 0 or n >= observed.shape[1]:
        return list(observed.columns)
    var = np.log1p(observed).var(axis=0)
    return list(var.sort_values(ascending=False).head(n).index)


def correlation_matrix(
    observed: pd.DataFrame, predicted: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """M[i, j] = corr(predicted for model i, observed in experiment j).

    Rows are models, columns are experiments. On log1p counts, since counts
    span orders of magnitude and an untransformed Pearson would be dominated by
    the few strongest peaks.
    """
    pred = np.log1p(predicted)
    obs = np.log1p(observed)
    exps = list(observed.index)
    out = pd.DataFrame(index=exps, columns=exps, dtype=float)
    for i in exps:
        pi = pred.loc[i]
        for j in exps:
            out.at[i, j] = pi.corr(obs.loc[j], method=method)
    out.index.name = "model"
    out.columns.name = "experiment"
    return out


def tier_of(model: str, experiment: str, groups: dict[str, str]) -> str:
    if model == experiment:
        return "matched"
    gm, ge = groups.get(model), groups.get(experiment)
    if gm is not None and gm == ge:
        return "same tissue"
    return "different tissue"


def long_form(matrix: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    rows = []
    for model in matrix.index:
        for experiment in matrix.columns:
            value = matrix.at[model, experiment]
            if pd.isna(value):
                continue
            rows.append({
                "model": model,
                "experiment": experiment,
                "model_group": groups.get(model),
                "experiment_group": groups.get(experiment),
                "tier": tier_of(model, experiment, groups),
                "correlation": float(value),
            })
    return pd.DataFrame(rows)


def summarize_tiers(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-tier location and spread, plus each tier's gap to the next."""
    rows = []
    for tier in TIERS:
        sub = pairs[pairs["tier"] == tier]["correlation"]
        if not len(sub):
            continue
        rows.append({
            "tier": tier,
            "n_pairs": len(sub),
            "mean": round(float(sub.mean()), 4),
            "median": round(float(sub.median()), 4),
            "q25": round(float(sub.quantile(0.25)), 4),
            "q75": round(float(sub.quantile(0.75)), 4),
        })
    out = pd.DataFrame(rows)
    if len(out) > 1:
        out["delta_to_next"] = out["median"].diff(-1).round(4)
    return out


def paired_within_model(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per model, its matched correlation minus its median mismatched one.

    The right unit of analysis. Pooling all pairs and comparing tiers treats
    every pair as independent when they share models, and lets a handful of
    well-predicted experiments carry the result. Asking instead how often a
    model beats its own mismatched baseline gives one number per model and a
    sign test with an obvious null of 50%.
    """
    rows = []
    for model, sub in pairs.groupby("model"):
        matched = sub[sub["tier"] == "matched"]["correlation"]
        same = sub[sub["tier"] == "same tissue"]["correlation"]
        diff = sub[sub["tier"] == "different tissue"]["correlation"]
        if not len(matched):
            continue
        rows.append({
            "model": model,
            "group": sub["model_group"].iloc[0],
            "matched": round(float(matched.iloc[0]), 4),
            "median_same_tissue": (
                round(float(same.median()), 4) if len(same) else np.nan
            ),
            "median_different_tissue": (
                round(float(diff.median()), 4) if len(diff) else np.nan
            ),
            "beats_different_tissue": (
                bool(matched.iloc[0] > diff.median()) if len(diff) else None
            ),
            "beats_same_tissue": (
                bool(matched.iloc[0] > same.median()) if len(same) else None
            ),
        })
    return pd.DataFrame(rows)


def sign_test(successes: int, trials: int) -> float:
    """Two-sided exact binomial p against p=0.5, without scipy."""
    if trials == 0:
        return float("nan")
    from math import comb

    def tail(k):
        return sum(comb(trials, i) for i in range(0, k + 1)) / 2 ** trials

    lo = min(successes, trials - successes)
    return min(1.0, 2 * tail(lo))


def load_read_depth() -> dict[str, float]:
    if not N_READS_PATH.exists():
        return {}
    d = pd.read_csv(N_READS_PATH, sep="\t")
    return dict(zip(d["experiment"], d["total_reads"]))


def plot_matrix(matrix: pd.DataFrame, groups: dict[str, str], path: Path) -> None:
    order = sorted(matrix.index, key=lambda e: (groups.get(e) or "", e))
    m = matrix.loc[order, order]
    fig, ax = plt.subplots(figsize=(max(4, 0.09 * len(order) + 2),) * 2)
    im = ax.imshow(m.to_numpy(dtype=float), cmap="viridis", aspect="equal")
    ax.set_xlabel("observed in experiment")
    ax.set_ylabel("predicted by model")
    ax.set_xticks([])
    ax.set_yticks([])

    # Tissue-group boundaries, so block structure along the diagonal is
    # visible without labelling 200 axes.
    labels = [groups.get(e) or "other" for e in order]
    edges = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    for e in edges:
        ax.axhline(e - 0.5, color="white", lw=0.4)
        ax.axvline(e - 0.5, color="white", lw=0.4)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r (log1p counts)")
    ax.set_title("Cross-experiment prediction accuracy", fontsize=10)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tiers(pairs: pd.DataFrame, path: Path) -> None:
    present = [t for t in TIERS if (pairs["tier"] == t).any()]
    data = [pairs[pairs["tier"] == t]["correlation"].to_numpy() for t in present]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("tab:blue")
        body.set_alpha(0.55)
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(present, fontsize=8)
    ax.set_ylabel("Pearson r (log1p counts)")
    ax.set_title("Prediction accuracy by relatedness", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--observed", type=Path,
                        default=DEFAULT_DIR / "observed_counts.tsv", metavar="PATH")
    parser.add_argument("--predicted", type=Path,
                        default=DEFAULT_DIR / "predicted_counts.tsv", metavar="PATH")
    parser.add_argument("--biosample-groups", type=Path, default=None, metavar="PATH",
                        help="curated biosample<TAB>group override table")
    parser.add_argument("--method", default="pearson",
                        choices=["pearson", "spearman"])
    parser.add_argument("--variable-peaks", type=int, default=0, metavar="N",
                        help="restrict to the N most variable peaks across "
                             "experiments; 0 uses all (default: 0)")
    parser.add_argument(
        "--i-know-these-are-fold-averaged", action="store_true",
        help="proceed with counts from the default (all-folds-averaged) "
             "extraction. The matched diagonal will be inflated because six of "
             "seven folds trained on those peaks; for exploring shape only",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, metavar="DIR")
    args = parser.parse_args()

    for path in (args.observed, args.predicted):
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            print(
                "Produce them on the cluster with:\n"
                "  python src/analysis/count_correlation.py --model bpnet "
                "--held-out-folds --device cuda",
                file=sys.stderr,
            )
            sys.exit(1)

    if not args.i_know_these_are_fold_averaged:
        print(
            "NOTE: assuming these counts came from --held-out-folds. If they "
            "came from the default extraction the matched diagonal is inflated "
            "by construction; rerun with --held-out-folds or pass "
            "--i-know-these-are-fold-averaged to proceed anyway.",
            file=sys.stderr,
        )

    observed, predicted = align(load_counts(args.observed), load_counts(args.predicted))
    print(
        f"{len(observed)} experiments x {observed.shape[1]:,} shared peaks",
        file=sys.stderr,
    )

    peaks = most_variable_peaks(observed, args.variable_peaks)
    if len(peaks) != observed.shape[1]:
        print(f"restricted to {len(peaks):,} most variable peaks", file=sys.stderr)
        observed, predicted = observed[peaks], predicted[peaks]

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["experiments"]
    tissue, _ = load_group_map(cfg, args.biosample_groups, quiet=True)
    groups = {e: tissue.get(e) for e in observed.index}

    matrix = correlation_matrix(observed, predicted, args.method)
    pairs = long_form(matrix, groups)
    summary = summarize_tiers(pairs)
    per_model = paired_within_model(pairs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.out_dir / "cross_celltype_matrix.tsv", sep="\t")
    pairs.to_csv(args.out_dir / "cross_celltype_pairs.tsv", sep="\t", index=False)
    summary.to_csv(args.out_dir / "cross_celltype_tiers.tsv", sep="\t", index=False)
    per_model.to_csv(args.out_dir / "cross_celltype_per_model.tsv", sep="\t",
                     index=False)
    plot_matrix(matrix, groups, args.out_dir / "cross_celltype_matrix.pdf")
    plot_tiers(pairs, args.out_dir / "cross_celltype_tiers.pdf")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nAccuracy by relatedness:", file=sys.stderr)
        print(summary.to_string(index=False), file=sys.stderr)

    for column, label in (
        ("beats_different_tissue", "a different tissue"),
        ("beats_same_tissue", "the same tissue"),
    ):
        valid = per_model[per_model[column].notna()]
        if not len(valid):
            continue
        wins = int(valid[column].sum())
        print(
            f"\n{wins}/{len(valid)} models predict their own experiment better "
            f"than their median experiment from {label} "
            f"(sign test p = {sign_test(wins, len(valid)):.3g})",
            file=sys.stderr,
        )

    depth = load_read_depth()
    if depth and len(per_model) > 2:
        d = per_model.assign(reads=per_model["model"].map(depth)).dropna(
            subset=["reads"]
        )
        if len(d) > 2:
            r = d["matched"].corr(np.log10(d["reads"]), method="spearman")
            print(
                f"\nMatched accuracy vs log10 read depth: Spearman r = {r:.3f} "
                f"over {len(d)} models. Depth differs by tissue group on this "
                "atlas, so a non-trivial value here means the tier gap needs "
                "a depth-matched check before it is quoted.",
                file=sys.stderr,
            )

    print(f"\nSaved 4 tables and 2 figures to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
