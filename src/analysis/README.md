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

## Warning Flags

Generates read-depth, perturbation, uncapped-library, and manual warning flags
for experiments.

```bash
python src/analysis/generate_warning_flags.py
python src/analysis/generate_warning_flags.py --yellow-read-threshold 20000000 --red-read-threshold 10000000
python src/analysis/generate_warning_flags.py --manual-red-experiment ENCSR000ABC:"failed QC"
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
  exclusion keywords used to produce `configs/model_warning_flags.tsv`; the
  MetaFormer target TSV helper consumes that table.
