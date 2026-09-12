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

TIERS = ("matched", "same biosample", "same tissue", "different tissue")


def load_counts(path: Path) -> pd.DataFrame:
    """Experiments x peaks count matrix, experiments on the index.

    Rejects negative values. Counts cannot be negative -- observed signal is
    RPM over abs()'d strands and predictions are softmax times exp(log_counts)
    -- and a negative entry would otherwise fail silently and catastrophically:
    normalization scales it up, `log1p` of anything at or below -1 is NaN, and
    every correlation involving that row comes back as exactly 0.0 rather than
    as an error. A minus-strand BigWig read without abs() would do it, since
    this repo stores minus-strand coverage as negative values.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    if df.empty:
        raise ValueError(f"{path} has no rows")
    negative = (df < 0).sum().sum()
    if negative:
        worst = float(df.min().min())
        raise ValueError(
            f"{path} has {negative:,} negative value(s), minimum {worst:.4g}. "
            "Counts must be non-negative; check that minus-strand signal was "
            "absolute-valued during extraction."
        )
    return df


def take_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Column subset by integer position rather than by label.

    `df[cols]` and `df.loc[:, cols]` do label lookups, which at ~100,000
    columns cost minutes: pandas builds an indexer entry per label. Resolving
    labels to positions once and slicing with `.iloc` is the same result in
    well under a second.
    """
    if list(cols) == list(df.columns):
        return df
    positions = df.columns.get_indexer(pd.Index(cols))
    if (positions < 0).any():
        missing = pd.Index(cols)[positions < 0]
        raise KeyError(f"{len(missing)} column(s) not present, e.g. {missing[0]!r}")
    return df.iloc[:, positions]


def normalize_within_peaks(df: pd.DataFrame, total: float = 1e6) -> pd.DataFrame:
    """Rescale each row to a common total over the peak set.

    Required, because observed and predicted counts arrive on entirely
    different scales. `count_correlation.py` RPM-normalizes observed signal
    (row sums ~122,000, tightly clustered) but leaves predictions in whatever
    count scale each model was trained on (row sums ~9,065,000, spanning 4x and
    tracking read depth at Spearman 0.879).

    That is not merely a constant factor. With observed values median 0.039 and
    92% of entries below 1, `log1p` is effectively *linear* on the observed
    side, while predicted values median 34 with 87% above 10 put it firmly in
    its *logarithmic* regime. Correlating the two as-is compares a near-linear
    quantity against a log one, and differencing inherits a per-model offset of
    roughly log(scale_i / scale_j), which biases the sign of a differential
    directly -- a correlation is immune to it, sign accuracy is not.

    After this, both sides are counts-per-million within the peak set and
    log1p means the same thing on each.
    """
    sums = df.sum(axis=1).replace(0, np.nan)
    return df.div(sums, axis=0) * total


def align(observed: pd.DataFrame, predicted: pd.DataFrame) -> tuple:
    """Restrict both matrices to shared experiments and shared peak columns.

    Column counts can differ if the two files were produced by different runs;
    silently correlating misaligned vectors would give a meaningless matrix, so
    this intersects explicitly and the caller reports what was dropped.

    The identical-columns case -- both files from one run, which is the normal
    one -- short-circuits before any column indexing happens at all.
    """
    exps = [e for e in observed.index if e in set(predicted.index)]
    if not exps:
        raise ValueError("no experiments in common between observed and predicted")

    if list(observed.columns) == list(predicted.columns):
        cols = observed.columns
    else:
        pred_cols = set(predicted.columns)
        cols = pd.Index([c for c in observed.columns if c in pred_cols])
    if not len(cols):
        raise ValueError("no peaks in common between observed and predicted")

    return (
        take_columns(observed.loc[exps], cols),
        take_columns(predicted.loc[exps], cols),
    )


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


def group_means(observed: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Mean log1p signal per tissue group, groups x peaks.

    Group means rather than experiment means, so a group with 41 experiments
    does not outvote one with 4.
    """
    log_obs = np.log1p(observed)
    labelled = [e for e in log_obs.index if groups.get(e) is not None]
    if len(labelled) < 2:
        return pd.DataFrame()
    return log_obs.loc[labelled].groupby([groups[e] for e in labelled]).mean()


def peak_specificity(
    observed: pd.DataFrame, groups: dict[str, str], index: str = "tau"
) -> pd.Series:
    """Per-peak tissue specificity of observed signal, in [0, 1].

    `tau` (default) is Yanai et al. 2005's tissue-specificity index,

        tau = sum_i (1 - x_i / x_max) / (n - 1)

    over per-group mean log1p signal: 0 when a peak is equally active in every
    tissue group, 1 when all its signal sits in one. Preferred over a bespoke
    measure because it is the standard index in expression analysis and
    benchmarked as the best-performing one (Kryuchkova-Mostacci &
    Robinson-Rechavi 2017), so the threshold is citable rather than invented
    here. Computed on log-transformed signal, as that benchmark recommends.

    `entropy` is the normalized-entropy alternative, `1 - H(q)/log(G)`, the
    same form `motif_hit_density.py` uses for motifs. It is retained for
    consistency with that panel; tau is the more sensitive of the two at the
    specific end.

    Either way it must be quantitative, because peak *calling* is not a usable
    specificity measure at this scale. With 224 experiments a promoter with
    modest lineage-biased activity is still called a peak nearly everywhere, so
    breadth of peak calls is dominated by near-ubiquitous peaks and its narrow
    tail is weak singletons rather than strong lineage-specific promoters --
    the same reason prevalence-1 motif clusters are noise rather than rare
    biology.
    """
    by_group = group_means(observed, groups)
    n_groups = len(by_group)
    if n_groups < 2:
        return pd.Series(np.nan, index=observed.columns)

    if index == "tau":
        peak_max = by_group.max(axis=0)
        # All-zero peaks have no max to normalize by; NaN rather than a
        # fabricated 0 or 1, and stratification drops them.
        ratios = by_group.div(peak_max.replace(0, np.nan), axis=1)
        spec = (1 - ratios).sum(axis=0) / (n_groups - 1)
    elif index == "entropy":
        totals = by_group.sum(axis=0)
        q = by_group.div(totals.replace(0, np.nan), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -(q * np.log(q)).sum(axis=0, skipna=True)
        spec = 1 - entropy / np.log(n_groups)
    else:
        raise ValueError(f"unknown specificity index {index!r}")

    spec[peak_max.isna() if index == "tau" else totals.isna()] = np.nan
    return spec.clip(lower=0, upper=1)


def top_group_signal(observed: pd.DataFrame, groups: dict[str, str]) -> pd.Series:
    """Each peak's strongest tissue-group mean, back in count units."""
    by_group = group_means(observed, groups)
    if by_group.empty:
        return pd.Series(np.nan, index=observed.columns)
    return np.expm1(by_group.max(axis=0))


def stratify_by_specificity(
    spec: pd.Series, quantile: float
) -> dict[str, list]:
    """Top and bottom `quantile` of peaks by specificity.

    Returns the two strata the cross-cell-type gap should differ between: it
    should be large among specific peaks and small among ubiquitous ones. A
    gap of similar size in both would mean the comparison is measuring
    something other than cell-type specificity -- read depth, say, or overall
    model quality.
    """
    valid = spec.dropna()
    if valid.empty or not 0 < quantile < 0.5:
        return {}
    lo = valid.quantile(quantile)
    hi = valid.quantile(1 - quantile)
    return {
        "specific": list(valid[valid >= hi].index),
        "ubiquitous": list(valid[valid <= lo].index),
    }


def _standardize(a: np.ndarray) -> np.ndarray:
    """Row-wise z-scores, with zero-variance rows left as zeros."""
    centered = a - a.mean(axis=1, keepdims=True)
    sd = centered.std(axis=1, keepdims=True)
    return np.divide(centered, sd, out=np.zeros_like(centered), where=sd > 0)


def correlation_matrix(
    observed: pd.DataFrame, predicted: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """M[i, j] = corr(predicted for model i, observed in experiment j).

    Rows are models, columns are experiments. On log1p counts, since counts
    span orders of magnitude and an untransformed Pearson would be dominated by
    the few strongest peaks.

    Computed as a single matrix product of row-standardized values rather than
    pair by pair. At 50 experiments and ~100,000 peaks the pair-by-pair form is
    2,500 pandas `Series.corr` calls -- and the specificity stratification
    needs two more matrices on top -- which takes minutes; standardizing once
    and multiplying takes well under a second for identical results.
    """
    pred = np.log1p(predicted.to_numpy(dtype=float))
    obs = np.log1p(observed.to_numpy(dtype=float))
    if method == "spearman":
        # Spearman is Pearson on ranks, so rank each row and reuse the same
        # product. `argsort().argsort()` gives ordinal ranks; ties are broken
        # arbitrarily rather than averaged, which at 100,000 mostly-distinct
        # values moves the correlation in the fourth decimal at most.
        pred = pred.argsort(axis=1).argsort(axis=1).astype(float)
        obs = obs.argsort(axis=1).argsort(axis=1).astype(float)
    elif method != "pearson":
        raise ValueError(f"unknown method {method!r}")

    n_peaks = pred.shape[1]
    values = _standardize(pred) @ _standardize(obs).T / n_peaks
    out = pd.DataFrame(
        values, index=list(predicted.index), columns=list(observed.index)
    )
    out.index.name = "model"
    out.columns.name = "experiment"
    return out


def dominant_tissue_accuracy(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    groups: dict[str, str],
    specificity: pd.Series,
    quantiles=(0.0, 0.5, 0.8, 0.9, 0.95, 0.99),
    ks=(1, 3, 5),
    min_experiments: int = 2,
) -> pd.DataFrame:
    """Can the models name which tissue a locus is most active in?

    For each peak, rank tissue groups by predicted signal and ask where the
    observed most-active group lands. Top-k accuracy against a chance rate of
    k/G.

    The cleanest framing available here, because it cancels the shared
    component *by construction* rather than by peak selection or differencing:
    every model sees the identical sequence at a given peak, so all variation
    across models at that peak is model-specific. It is also immune to the
    transformation questions that plague the correlations -- only the ordering
    of groups matters, not units, scale, or whether log1p is behaving linearly.

    Groups are restricted to those with `min_experiments` or more, so a group
    mean is not a single experiment.
    """
    obs_g = group_means(observed, groups)
    pred_g = group_means(predicted, groups)
    if obs_g.empty or pred_g.empty:
        return pd.DataFrame()

    counts = pd.Series([groups[e] for e in observed.index if groups.get(e)]).value_counts()
    keep = [g for g in obs_g.index if counts.get(g, 0) >= min_experiments]
    obs_g, pred_g = obs_g.loc[keep], pred_g.loc[keep]
    n_groups = len(keep)
    if n_groups < 2:
        return pd.DataFrame()

    spec = specificity.dropna()
    rows = []
    for q in quantiles:
        threshold = spec.quantile(q) if q > 0 else -np.inf
        cols = [c for c in spec[spec >= threshold].index if c in obs_g.columns]
        if len(cols) < 20:
            continue
        obs_v = obs_g[cols].to_numpy()
        pred_v = pred_g[cols].to_numpy()
        true = obs_v.argmax(axis=0)
        order = np.argsort(-pred_v, axis=0)
        rank = (order == true[None, :]).argmax(axis=0)
        row = {
            "tau_quantile": q,
            "n_peaks": len(cols),
            "n_groups": n_groups,
            "median_rank_of_true": int(np.median(rank)) + 1,
        }
        for k in ks:
            if k <= n_groups:
                row[f"top{k}"] = round(float(np.mean(rank < k)), 4)
                row[f"top{k}_chance"] = round(k / n_groups, 4)
                row[f"top{k}_over_chance"] = round(
                    float(np.mean(rank < k)) / (k / n_groups), 2
                )
        rows.append(row)
    return pd.DataFrame(rows)


def differential_prediction(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    groups: dict[str, str],
    biosamples: dict[str, str] | None,
) -> pd.DataFrame:
    """Does a model predict the *difference* between two cell types?

    For each unordered pair, `corr(pred_i - pred_j, obs_i - obs_j)` across
    peaks. Differencing cancels the shared component analytically instead of
    by peak selection, which matters because a level correlation is dominated
    by sequence-intrinsic promoter strength: two models can correlate at 0.61
    across tissue-specific peaks while the deviations that encode cell type --
    small in variance, but the entire question -- are invisible.

    This is the projection that perturbation and promoter-edit experiments
    measure, since `f(mutant) - f(reference)` within one model cancels the
    baseline exactly. It is why matched-versus-unmatched can be decisive for
    edits while looking modest in a level correlation, and reconciles those two
    literatures rather than choosing between them.

    Its own internal check is the tier ordering: predictability should scale
    with how large the true difference is. On this atlas it does -- replicate
    pairs 0.080, same tissue 0.125, different tissue 0.175 -- since replicate
    pairs differ only by noise and so offer nothing to predict. (Those are the
    `--normalize within-peaks` values; before that rescaling the cross-tissue
    median was 0.146.)
    """
    obs = np.log1p(observed.to_numpy(dtype=float))
    pred = np.log1p(predicted.to_numpy(dtype=float))
    exps = list(observed.index)

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan

    rows = []
    for i in range(len(exps)):
        for j in range(i + 1, len(exps)):
            rows.append({
                "experiment_a": exps[i],
                "experiment_b": exps[j],
                "tier": tier_of(exps[i], exps[j], groups, biosamples),
                "differential_r": round(
                    corr(pred[i] - pred[j], obs[i] - obs[j]), 4
                ),
            })
    return pd.DataFrame(rows)


def summarize_differential(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier in TIERS:
        sub = pairs[pairs["tier"] == tier]["differential_r"].dropna()
        if not len(sub):
            continue
        rows.append({
            "tier": tier,
            "n_pairs": len(sub),
            "median": round(float(sub.median()), 4),
            "q25": round(float(sub.quantile(0.25)), 4),
            "q75": round(float(sub.quantile(0.75)), 4),
            "frac_positive": round(float((sub > 0).mean()), 4),
            "sign_test_p": sign_test(int((sub > 0).sum()), len(sub)),
        })
    return pd.DataFrame(rows)


def homogenization(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    groups: dict[str, str],
    biosamples: dict[str, str] | None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Measured-vs-measured against predicted-vs-predicted, by tier.

    ProCapNet's own cross-cell-type comparison (Cochran et al. 2024), and the
    one that supports its central claim: predictions correlated across
    cell-line pairs at r = 0.8-0.97 while the measurements correlated at only
    r = 0.5-0.71, so the models represent cell types as far more alike than
    they are. That is what "a largely cell-type-agnostic cis-regulatory code
    of initiation" means quantitatively.

    It answers a different question from the matched-vs-mismatched tiers.
    Those ask whether a model carries *any* cell-type-specific information --
    it does. This asks how much of the real cell-type difference the models
    reproduce, and is the more demanding test: a model can beat every
    mismatched competitor while still predicting almost the same profile
    everywhere.

    Stratified by peak specificity it becomes very stark. At tissue-specific
    peaks on this atlas the measurements are uncorrelated across tissue groups
    (median r = -0.009, as expected since the peaks are specific) while the
    predictions still correlate at 0.614.
    """
    rows = []
    for label, left, right in (
        ("measured", observed, observed),
        ("predicted", predicted, predicted),
    ):
        pairs = long_form(
            correlation_matrix(left, right, method), groups, biosamples
        )
        pairs = pairs[pairs["tier"] != "matched"]   # 1.0 by identity
        summary = summarize_tiers(pairs)
        summary.insert(0, "source", label)
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True)
    return out[["source", "tier", "n_pairs", "median", "q25", "q75"]]


def consensus_benchmark(
    observed: pd.DataFrame, predicted: pd.DataFrame
) -> pd.DataFrame:
    """Each matched model against two cell-type-agnostic predictors.

    This is the test of whether a model carries cell-type-specific
    information, and it is self-contained: no appeal to replicate
    reproducibility, which a sequence model could never reach anyway -- a
    replicate shares the entire non-sequence cell state, while the genome is
    identical across every cell type, so the model's only route to
    cell-type-specificity is what training put in its weights.

    Two agnostic predictors, and they answer different questions:

    `consensus_model` is the mean prediction over all models. Beating it means
    a model's own weights encode something cell-type-specific rather than a
    generic promoter program. This is the comparison that licenses the claim.

    `consensus_observed` is the mean observed signal over all experiments,
    i.e. the best guess with no cell-type knowledge at all. It is a practical
    benchmark rather than a test of the model: it tends to win, because
    measuring anything is a strong predictor of measuring something similar.

    Returns one row per experiment.
    """
    obs = np.log1p(observed.to_numpy(dtype=float))
    pred = np.log1p(predicted.to_numpy(dtype=float))
    cons_obs = obs.mean(axis=0)
    cons_pred = pred.mean(axis=0)

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan

    rows = []
    for i, exp_id in enumerate(observed.index):
        rows.append({
            "experiment": exp_id,
            "matched_model": round(corr(pred[i], obs[i]), 4),
            "consensus_model": round(corr(cons_pred, obs[i]), 4),
            "consensus_observed": round(corr(cons_obs, obs[i]), 4),
        })
    out = pd.DataFrame(rows)
    out["beats_consensus_model"] = out["matched_model"] > out["consensus_model"]
    out["beats_consensus_observed"] = (
        out["matched_model"] > out["consensus_observed"]
    )
    return out


def reproducibility_baseline(
    observed: pd.DataFrame,
    groups: dict[str, str],
    biosamples: dict[str, str] | None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Observed-vs-observed correlations, by tier.

    Two reference points a predicted-vs-observed number cannot be read without.

    The **ceiling**: a model predicts from sequence alone, so it can at best
    recover the reproducible part of an experiment's signal. Replicate
    agreement (`same biosample`, observed vs observed) is that ceiling, and it
    differs enormously between strata -- so the same r means different things
    in each.

    The **benchmark**: what you would get by using a *related experiment's
    measurements* instead of a model. If observed same-tissue agreement
    exceeds predicted matched accuracy, then another sample of the same lineage
    is a better predictor than the model, which is worth knowing before
    claiming the model has learned cell-type-specific initiation.
    """
    matrix = correlation_matrix(observed, observed, method)
    pairs = long_form(matrix, groups, biosamples)
    out = pairs[pairs["tier"] != "matched"]      # matched is 1.0 by identity
    return summarize_tiers(out)


def tier_of(
    model: str,
    experiment: str,
    groups: dict[str, str],
    biosamples: dict[str, str] | None = None,
) -> str:
    """Which relatedness tier a (model, experiment) pair falls in.

    `same biosample` is separated from `same tissue` because the atlas is
    heavily replicated -- HCT116 has 16 experiments, the metastatic breast
    biosample 10, PBMC 8 -- so without the split a replicate pair would count
    as "same tissue" and could dominate that tier. The claim of interest is
    that a model transfers to a *different* sample of the same lineage; a
    replicate pair only shows it transfers to a rerun of its own sample, which
    is much weaker and closer to the matched case.
    """
    if model == experiment:
        return "matched"
    if biosamples is not None:
        bm, be = biosamples.get(model), biosamples.get(experiment)
        if bm is not None and bm == be:
            return "same biosample"
    gm, ge = groups.get(model), groups.get(experiment)
    if gm is not None and gm == ge:
        return "same tissue"
    return "different tissue"


def long_form(
    matrix: pd.DataFrame,
    groups: dict[str, str],
    biosamples: dict[str, str] | None = None,
) -> pd.DataFrame:
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
                "model_biosample": (biosamples or {}).get(model),
                "tier": tier_of(model, experiment, groups, biosamples),
                "correlation": float(value),
            })
    return pd.DataFrame(rows)


def summarize_tiers(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-tier location and spread, plus each tier's gap to the next.

    Reports a group-balanced median alongside the pooled one whenever the
    model's tissue group is available. Pooling weights every pair equally,
    which over the full atlas hands the result to the largest groups:
    `blood_immune` is 41 of 198 experiments, so it contributes roughly a
    quarter of all cross-tissue pairs and most within-tissue ones. The
    balanced figure takes each group's median first and then the median across
    groups, giving a 4-experiment lineage the same say as a 41-experiment one.
    """
    rows = []
    for tier in TIERS:
        sub = pairs[pairs["tier"] == tier]
        if not len(sub):
            continue
        values = sub["correlation"]
        row = {
            "tier": tier,
            "n_pairs": len(values),
            "mean": round(float(values.mean()), 4),
            "median": round(float(values.median()), 4),
            "q25": round(float(values.quantile(0.25)), 4),
            "q75": round(float(values.quantile(0.75)), 4),
        }
        if "model_group" in sub.columns and sub["model_group"].notna().any():
            per_group = sub.groupby("model_group")["correlation"].median()
            row["median_group_balanced"] = round(float(per_group.median()), 4)
            row["n_groups"] = int(per_group.notna().sum())
        rows.append(row)
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
        repl = sub[sub["tier"] == "same biosample"]["correlation"]
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
            "median_same_biosample": (
                round(float(repl.median()), 4) if len(repl) else np.nan
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
    parser.add_argument(
        "--normalize", default="within-peaks", choices=["within-peaks", "none"],
        help="rescale each experiment's observed and predicted counts to a "
             "common total over the peak set before comparing (default). "
             "Observed counts are RPM-normalized while predictions are in each "
             "model's own count scale, 74x larger and tracking read depth, "
             "which puts log1p in a linear regime on one side and a "
             "logarithmic one on the other. 'none' reproduces the unnormalized "
             "comparison.",
    )
    parser.add_argument(
        "--min-peak-signal", type=float, default=0.5, metavar="RPM",
        help="before ranking by specificity, drop peaks whose strongest "
             "tissue-group mean is below this (default: 0.5). Tau is inflated "
             "by noise on near-empty peaks -- unfiltered, the top-decile "
             "stratum has 12x less signal than the bottom decile and tau "
             "anti-correlates with signal at rho = -0.45 -- which is why "
             "filtering low-signal features before tau is standard. Raising "
             "it sharpens the contrast rather than weakening it.",
    )
    parser.add_argument(
        "--specificity-index", default="tau", choices=["tau", "entropy"],
        help="tissue-specificity index: Yanai et al. 2005 tau (default), or "
             "normalized entropy as used by motif_hit_density.py",
    )
    parser.add_argument(
        "--specificity-quantile", type=float, default=0.1, metavar="Q",
        help="also report the tiers separately among the top and bottom Q of "
             "peaks by observed tissue specificity (default: 0.1; 0 disables). "
             "The gap should be large among specific peaks and small among "
             "ubiquitous ones -- a similar gap in both would mean the "
             "comparison is tracking something other than cell-type identity.",
    )
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
    if args.normalize == "within-peaks":
        obs_total = float(observed.sum(axis=1).median())
        pred_total = float(predicted.sum(axis=1).median())
        print(
            f"Normalizing within peaks: observed row sums ~{obs_total:,.0f} and "
            f"predicted ~{pred_total:,.0f} rescaled to a common 1e6",
            file=sys.stderr,
        )
        observed = normalize_within_peaks(observed)
        predicted = normalize_within_peaks(predicted)
    print(
        f"{len(observed)} experiments x {observed.shape[1]:,} shared peaks",
        file=sys.stderr,
    )

    peaks = most_variable_peaks(observed, args.variable_peaks)
    if len(peaks) != observed.shape[1]:
        print(f"restricted to {len(peaks):,} most variable peaks", file=sys.stderr)
        observed = take_columns(observed, peaks)
        predicted = take_columns(predicted, peaks)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["experiments"]
    tissue, biosample = load_group_map(cfg, args.biosample_groups, quiet=True)
    groups = {e: tissue.get(e) for e in observed.index}
    biosamples = {e: biosample.get(e) for e in observed.index}

    matrix = correlation_matrix(observed, predicted, args.method)
    pairs = long_form(matrix, groups, biosamples)
    summary = summarize_tiers(pairs)
    per_model = paired_within_model(pairs)
    baseline = reproducibility_baseline(observed, groups, biosamples, args.method)
    diff_pairs = differential_prediction(observed, predicted, groups, biosamples)
    diff_summary = summarize_differential(diff_pairs)
    homog = homogenization(observed, predicted, groups, biosamples, args.method)
    homog.to_csv(args.out_dir / "cross_celltype_homogenization.tsv", sep="\t",
                 index=False) if args.out_dir.exists() else None
    baseline.to_csv(args.out_dir / "cross_celltype_baseline.tsv", sep="\t",
                    index=False) if False else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.out_dir / "cross_celltype_matrix.tsv", sep="\t")
    pairs.to_csv(args.out_dir / "cross_celltype_pairs.tsv", sep="\t", index=False)
    summary.to_csv(args.out_dir / "cross_celltype_tiers.tsv", sep="\t", index=False)
    per_model.to_csv(args.out_dir / "cross_celltype_per_model.tsv", sep="\t",
                     index=False)
    plot_matrix(matrix, groups, args.out_dir / "cross_celltype_matrix.pdf")
    plot_tiers(pairs, args.out_dir / "cross_celltype_tiers.pdf")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(args.out_dir / "cross_celltype_baseline.tsv", sep="\t",
                    index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nAccuracy by relatedness:", file=sys.stderr)
        print(summary.to_string(index=False), file=sys.stderr)
        print(
            "\nObserved-vs-observed for the same tiers -- the ceiling a "
            "sequence model could reach, and what a related experiment's own "
            "measurements would give instead of a model:",
            file=sys.stderr,
        )
        print(
            baseline[["tier", "n_pairs", "median"]].to_string(index=False),
            file=sys.stderr,
        )
        homog.to_csv(
            args.out_dir / "cross_celltype_homogenization.tsv", sep="\t",
            index=False,
        )
        diff_pairs.to_csv(
            args.out_dir / "cross_celltype_differential_pairs.tsv", sep="\t",
            index=False,
        )
        diff_summary.to_csv(
            args.out_dir / "cross_celltype_differential.tsv", sep="\t",
            index=False,
        )
        print(
            "\nDifferential prediction -- corr(delta predicted, delta "
            "observed) per pair. Differencing cancels the shared promoter "
            "program, so this is the projection edit/perturbation experiments "
            "measure, and predictability should scale with how large the true "
            "difference is:",
            file=sys.stderr,
        )
        print(diff_summary.to_string(index=False), file=sys.stderr)
        print(
            "\nHow much cell-type difference the models reproduce "
            "(ProCapNet's comparison): predicted-vs-predicted should be no "
            "more similar across cell types than measured-vs-measured is.",
            file=sys.stderr,
        )
        print(
            homog.pivot(index="tier", columns="source", values="median")
            .reindex([t for t in TIERS if t != "matched"])
            .to_string(),
            file=sys.stderr,
        )

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

    if args.specificity_quantile > 0:
        spec = peak_specificity(observed, groups, args.specificity_index)
        if args.min_peak_signal > 0:
            signal = top_group_signal(observed, groups)
            eligible = signal[signal >= args.min_peak_signal].index
            print(
                f"specificity stratification restricted to {len(eligible):,} of "
                f"{len(spec):,} peaks with a tissue-group mean >= "
                f"{args.min_peak_signal} RPM",
                file=sys.stderr,
            )
            spec = spec.loc[eligible]
        topk = dominant_tissue_accuracy(observed, predicted, groups, spec)
        if len(topk):
            topk.to_csv(args.out_dir / "cross_celltype_topk.tsv", sep="\t",
                        index=False)
            with pd.option_context("display.width", 220):
                print(
                    "\nNaming the most-active tissue per peak, by specificity "
                    "threshold. Every model sees the same sequence at a peak, "
                    "so variation across models there is model-specific -- no "
                    "shared component to cancel and no units to worry about:",
                    file=sys.stderr,
                )
                cols = [c for c in topk.columns if not c.endswith("_chance")]
                print(topk[cols].to_string(index=False), file=sys.stderr)

        strata = stratify_by_specificity(spec, args.specificity_quantile)
        spec.rename("specificity").to_csv(
            args.out_dir / "peak_specificity.tsv", sep="\t"
        )
        rows = []
        for name, cols in strata.items():
            if len(cols) < 2:
                continue
            oc = take_columns(observed, cols)
            bench = consensus_benchmark(oc, take_columns(predicted, cols))
            bench.to_csv(
                args.out_dir / f"cross_celltype_consensus_{name}.tsv",
                sep="\t", index=False,
            )
            wins = int(bench["beats_consensus_model"].sum())
            wins_o = int(bench["beats_consensus_observed"].sum())
            n = len(bench)
            print(
                f"  {name:11s}: matched {bench['matched_model'].median():.3f} vs "
                f"consensus-model {bench['consensus_model'].median():.3f} "
                f"(wins {wins}/{n}, p={sign_test(wins, n):.2g}) vs "
                f"consensus-observed {bench['consensus_observed'].median():.3f} "
                f"(wins {wins_o}/{n}, p={sign_test(wins_o, n):.2g})",
                file=sys.stderr,
            )
            hom = homogenization(
                oc, take_columns(predicted, cols), groups, biosamples, args.method
            )
            hom_med = hom.set_index(["source", "tier"])["median"]
            print(
                f"  {name:11s}: across different tissues, measured "
                f"{hom_med.get(('measured', 'different tissue'), float('nan')):.3f} "
                f"vs predicted "
                f"{hom_med.get(('predicted', 'different tissue'), float('nan')):.3f}"
                "  <- models represent cell types as more alike than they are",
                file=sys.stderr,
            )
            hom.to_csv(
                args.out_dir / f"cross_celltype_homogenization_{name}.tsv",
                sep="\t", index=False,
            )
            base = reproducibility_baseline(oc, groups, biosamples, args.method)
            base_med = dict(zip(base["tier"], base["median"]))
            spread = np.log1p(oc.to_numpy(dtype=float)).std(axis=1)
            print(
                f"  {name:11s}: observed replicate agreement "
                f"{base_med.get('same biosample', float('nan')):.3f}, "
                f"observed same-tissue {base_med.get('same tissue', float('nan')):.3f}, "
                f"observed different-tissue "
                f"{base_med.get('different tissue', float('nan')):.3f}; "
                f"median within-experiment log1p sd {np.median(spread):.3f}",
                file=sys.stderr,
            )
            sub_pairs = long_form(
                correlation_matrix(
                    take_columns(observed, cols), take_columns(predicted, cols),
                    args.method,
                ),
                groups, biosamples,
            )
            summary_s = summarize_tiers(sub_pairs).assign(stratum=name,
                                                          n_peaks=len(cols))
            rows.append(summary_s)
        if rows:
            # Per-stratum sign tests, not just pooled medians. The pooled
            # version treats pairs sharing a model as independent, and the
            # stratum is where the claim actually lives.
            for name, cols in strata.items():
                if len(cols) < 2:
                    continue
                sub_pairs = long_form(
                    correlation_matrix(
                        take_columns(observed, cols),
                        take_columns(predicted, cols), args.method,
                    ),
                    groups, biosamples,
                )
                per = paired_within_model(sub_pairs)
                valid = per[per["beats_different_tissue"].notna()]
                if len(valid):
                    wins = int(valid["beats_different_tissue"].sum())
                    print(
                        f"  {name:11s}: {wins}/{len(valid)} models beat their "
                        f"median different-tissue pair "
                        f"(sign test p = {sign_test(wins, len(valid)):.3g})",
                        file=sys.stderr,
                    )
            strat = pd.concat(rows, ignore_index=True)
            strat.to_csv(args.out_dir / "cross_celltype_by_specificity.tsv",
                         sep="\t", index=False)
            with pd.option_context("display.width", 200):
                print(
                    "\nBy observed peak specificity "
                    f"({args.specificity_index}, top/bottom "
                    f"{args.specificity_quantile:.0%}):",
                    file=sys.stderr,
                )
                print(
                    strat[["stratum", "n_peaks", "tier", "n_pairs", "median"]]
                    .to_string(index=False),
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

    n_tsv = len(list(args.out_dir.glob("*.tsv")))
    n_fig = len(list(args.out_dir.glob("*.pdf")))
    print(f"\nSaved {n_tsv} tables and {n_fig} figures to {args.out_dir}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
