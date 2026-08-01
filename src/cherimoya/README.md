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
- A Cherimoya Python environment. Cherimoya's real `torch>=2.9.0`/
  `triton>=3.5.1` requirements have no wheels compatible with Sherlock's
  pip/uv (capped at `torch==2.6.0`), so the root `uv` project keeps them in
  separate, mutually exclusive extras (see the root `pyproject.toml`
  `tool.uv.conflicts`). Use one of:
  - On Sherlock: SRCC's `py-pytorch`/`py-triton` Lmod modules; see
    [`sherlock_native/`](sherlock_native/README.md). The Apptainer image
    under [`apptainer/`](apptainer/README.md) bundles a newer CUDA than
    Sherlock's GPU driver supports and cannot run there — it remains useful
    only for other clusters with old compilers but modern GPU drivers.
  - `uv sync --extra cherimoya` for a native install on other (non-Sherlock,
    modern-glibc) Linux hardware
- Optional SLURM access for launchers

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

On Sherlock, load SRCC's `py-pytorch`/`py-triton` modules (see
[`sherlock_native/`](sherlock_native/README.md)) rather than using the root
`uv` project's `cherimoya` extra or the Apptainer image — neither works on
Sherlock's GPU driver. The default root `uv` environment (no extras, or
`--extra sherlock`) pins `torch==2.6.0` for BPNet/preprocessing and does not
include Cherimoya at all. On other (non-Sherlock) Linux hardware, use
`uv sync --extra cherimoya` instead, which resolves Cherimoya's real
`torch>=2.9.0`/`triton>=3.5.1` requirements directly. The `sherlock` and
`cherimoya` extras are declared mutually exclusive and cannot be installed
together.

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

The launcher defaults are for the Sherlock HPC environment and load SRCC's
`py-pytorch`/`py-triton` modules directly (see
[`sherlock_native/`](sherlock_native/README.md)); review partitions and
resource requests, and swap in the Apptainer image (see
[`apptainer/`](apptainer/README.md)) instead, before using these launchers on
another cluster.

`launch.py` submits one SLURM job per experiment, training that experiment's
folds sequentially within the job (already-trained folds are skipped, and an
experiment with every fold already trained is not submitted at all). `--time`
is a flat budget for the whole job regardless of fold count (default
`48:00:00`, the `owners` partition's per-job cap).

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

Submit benchmarking jobs through SLURM:

```bash
python src/cherimoya/benchmark/launch.py --dry-run
python src/cherimoya/benchmark/launch.py --min-reads 20000000
python src/cherimoya/benchmark/launch.py --local             # run in the foreground, no SLURM
```

`launch.py` submits one SLURM job per experiment via `benchmark_cherimoya.py`,
mirroring [`src/bpnet/benchmark/launch.py`](../bpnet/benchmark/launch.py) but
using SRCC's `py-pytorch`/`py-triton` modules (see
[`sherlock_native/`](sherlock_native/README.md)) instead of the `procap-atlas`
conda environment. Experiments with any missing fold model are skipped, as are
experiments with an existing metrics JSON (override with `--force`). `--local`
runs each experiment directly in the foreground via `uv run --extra
cherimoya`, bypassing SLURM entirely — useful on a GPU box you already have a
shell on (e.g. a lab cluster).

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
