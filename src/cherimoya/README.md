# Cherimoya PRO-cap Models

Cherimoya training mirrors the BPNet workflow: one model per experiment and
chromosome fold, using processed PRO-cap BigWigs and peak BED files from
`configs/experiment_config.yaml`.

## Status

Cherimoya training and benchmarking are complete, but Cherimoya models are not
the public deployment path yet. Treat the current scripts as the training and
evaluation workflow, not as a public deployment pipeline.

## Prerequisites

- Completed preprocessing with `configs/experiment_config.yaml`,
  `configs/chrom_splits.yaml`, processed strand BigWigs, filtered peaks, and
  GC-matched negatives
- Python packages from the root `uv` project
- Optional SLURM access for launchers
- Optional Apptainer image for Sherlock launchers; see
  [`apptainer/`](apptainer/README.md)

## Training

```bash
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --background gc:0.1
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --n-filters 196 --batch-size 32
```

The training script uses `torch.optim.Muon` for 2D weight matrices and AdamW for
the remaining parameters, with warmup plus cosine learning-rate schedules.
Background sampling accepts the same repeatable `--background NAME:RATIO`
pattern as BPNet.

Outputs:

```text
models/cherimoya/{experiment}/
models/cherimoya/{experiment}_{background_config}/
```

Submit training jobs through SLURM:

```bash
python src/cherimoya/fit/launch.py --dry-run
python src/cherimoya/fit/launch.py --min-reads 20000000 --fit-args "--max-epochs 100"
```

The launcher defaults are for the Sherlock HPC environment. Apptainer is a
Sherlock-specific workaround for its older compiler and OS stack, not a general
Cherimoya requirement. Review partitions, bind paths, module setup, image paths,
and resource requests before using these launchers on another cluster.

## Benchmarking

```bash
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM --save-output
python src/cherimoya/benchmark/consolidate_metrics.py
```

Outputs:

```text
performance_metrics/cherimoya/{model_dir_name}.json
performance_metrics/cherimoya/procap-atlas_performance_metrics.tsv
predictions/cherimoya/{model_dir_name}.npz
```

Prediction output is written only when `--save-output` is used.

## Architecture Sweep

The `n_filters/` directory runs a Cherimoya filter-count sweep over `16`, `24`,
`36`, `48`, `64`, `96`, `196`, and `256` filters:

```bash
python src/cherimoya/n_filters/launch.py --dry-run
python src/cherimoya/n_filters/launch_benchmark.py --dry-run
python src/cherimoya/n_filters/consolidate_metrics.py
```

Outputs:

```text
models/cherimoya_n_filters/
performance_metrics/cherimoya_n_filters/
```

## Historical Notes

The first Cherimoya models were trained while `cherimoya` was in early
development, using commit `69f16dc7ff48ad094aafd4b93433972181c65d50`. Check out
that commit only if you need to reproduce that initial model set. A second set
of models were trained using `c0cbabe26cabfb5012f4fc5328af832e32f9ed04`.
