# Atlas Analyses

Atlas-level analysis scripts that operate on processed PRO-cap tracks, union
peaks, read counts, and model outputs.

## Status

These scripts are active analysis utilities. They are not required for model
training, but they produce useful QC summaries and comparison figures.

## Prerequisites

- `configs/experiment_config.yaml`
- `configs/n_reads.txt`
- Processed strand BigWigs
- `data/processed/peaks/union_peaks.bed.gz`
- Optional trained BPNet models for predicted-count correlation analyses

## Count Correlations

Extracts observed counts at union peaks, optionally adds model-predicted counts,
and saves pairwise experiment correlation clustermaps.

```bash
python src/analysis/count_correlation.py
python src/analysis/count_correlation.py --model bpnet
python src/analysis/count_correlation.py --experiment ENCSR882DWM --model bpnet
python src/analysis/count_correlation.py --min-reads 10000000 --device cuda
```

Outputs:

```text
figures/count_correlation/
```

Depending on options, outputs include observed count matrices, predicted count
matrices, and clustermap PNGs.

## BPNet vs. Cherimoya Comparison

Compares BPNet and Cherimoya genome-wide benchmark metrics across experiments
benchmarked for both models, producing a scatterplot (with a Wilcoxon
signed-rank test) and a delta histogram for each shared metric.

```bash
python src/analysis/compare_bpnet_cherimoya.py
python src/analysis/compare_bpnet_cherimoya.py --metrics profile_jsd
python src/analysis/compare_bpnet_cherimoya.py --min-reads 10000000
```

Outputs:

```text
plots/bpnet_vs_cherimoya/bpnet_vs_cherimoya_{metric}.pdf
```

Reads the consolidated TSVs at `performance_metrics/{bpnet,cherimoya}/procap-atlas_performance_metrics.tsv`
(see [`src/cherimoya/benchmark/consolidate_metrics.py`](../cherimoya/benchmark/consolidate_metrics.py))
and inner-joins on experiment, so only experiments benchmarked for both models
are compared. Currently `profile_jsd` and `log_counts_pearson` are the only
metrics present in both TSVs.

## Cherimoya Version Comparison

Compares Cherimoya benchmark metrics across the archived model versions under
`performance_metrics/cherimoya/{version}/` (plus the current run at the top
level of that directory) — see
[`src/cherimoya/README.md`](../cherimoya/README.md)'s Historical Notes for
what each version is. Produces the same scatterplot + delta histogram as the
BPNet comparison above, for every pair of versions.

```bash
python src/analysis/compare_cherimoya_versions.py
python src/analysis/compare_cherimoya_versions.py --metrics profile_jsd
python src/analysis/compare_cherimoya_versions.py --min-reads 10000000
```

Outputs:

```text
plots/cherimoya_versions/{version_a}_vs_{version_b}_{metric}.pdf
```

Unlike the BPNet comparison, all four Cherimoya benchmark metrics
(`profile_pearson`, `profile_jsd`, `log_counts_pearson`, `counts_spearman`)
are compared by default, since they're present in every archived version's
TSV. `compare_bpnet_cherimoya.py` and `compare_cherimoya_versions.py` share
their plotting logic via `_metric_comparison_plots.py`.

## Warning Flags

Generates read-depth, perturbation, uncapped-library, and manual warning flags
for experiments. Perturbation is split into two mutually exclusive flags:
`perturbation_treated` (metadata matches an active-treatment keyword, e.g.
dTAG/auxin induction) and `perturbation_untreated` (metadata matches a
genetic-perturbation keyword, e.g. CRISPR/degron insertion, but no
active-treatment keyword — typically the untreated control for a degron cell
line).

```bash
python src/analysis/generate_warning_flags.py
python src/analysis/generate_warning_flags.py --yellow-read-threshold 20000000 --red-read-threshold 10000000
python src/analysis/generate_warning_flags.py --manual-red-experiment ENCSR000ABC:"failed QC"
python src/analysis/generate_warning_flags.py --perturb-keyword "sirna" --treatment-keyword "auxin"
```

Outputs:

```text
configs/model_warning_flags.tsv
configs/model_warning_flags.json
```

## Notes

- `count_correlation.py` normalizes observed counts to RPM using
  `configs/n_reads.txt`.
- Predicted count analyses require complete fold models for the selected model
  family and experiment.
- Warning flag perturbation detection owns the metadata fields and default
  perturbation/treatment keywords used to produce
  `configs/model_warning_flags.tsv`; the MetaFormer target TSV helper consumes
  that table's `is_perturbation` column, which is true if either the treated
  or untreated flag is true.
