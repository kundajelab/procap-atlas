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
  - On Sherlock: the Cherimoya Apptainer image (see
    [`apptainer/`](apptainer/README.md)) is the default for the SLURM
    launchers below and lets training run with `torch.compile` enabled.
    SRCC's `py-pytorch`/`py-triton` Lmod modules (see
    [`sherlock_native/`](sherlock_native/README.md)) are also confirmed
    working and available as a fallback (`--native`), but that module pair
    disables `torch.compile`.
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

On Sherlock, run the script above via `apptainer exec --nv` against the
Cherimoya Apptainer image (see [`apptainer/`](apptainer/README.md)), or load
SRCC's `py-pytorch`/`py-triton` modules instead (see
[`sherlock_native/`](sherlock_native/README.md)) — the root `uv` project's
`cherimoya` extra does not work there, since the default root `uv`
environment (no extras, or `--extra sherlock`) pins `torch==2.6.0` for
BPNet/preprocessing and does not include Cherimoya at all. On other
(non-Sherlock) Linux hardware, use `uv sync --extra cherimoya` instead,
which resolves Cherimoya's real `torch>=2.9.0`/`triton>=3.5.1` requirements
directly. The `sherlock` and
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

The launcher defaults are for the Sherlock HPC environment and review
partitions and resource requests before using these launchers on another
cluster. By default, jobs run each fold via `apptainer exec --nv` against
the Cherimoya Apptainer image (see [`apptainer/`](apptainer/README.md)).
That image's Python 3.12 + torch 2.13.0 build isn't subject to Sherlock's
native module's Python-3.14 `torch.compile` restriction (see
`fit_cherimoya.py`'s `compile_supported`), so training runs compiled.
`apptainer/check_gpu.py` has confirmed this path works on real Sherlock
hardware across every GPU SKU tested.

`--native` runs each fold using SRCC's `py-pytorch`/`py-triton` Sherlock
modules instead (see [`sherlock_native/`](sherlock_native/README.md)) --
useful as a fallback, but that module pair disables `torch.compile`, so
prefer the Apptainer default unless you have a specific reason not to.

`--local` skips SLURM entirely and runs each fold directly in the
foreground (still via Apptainer by default; add `--native` for a `uv run
--extra cherimoya` invocation instead) -- useful on a GPU box you already
have a shell on. Combined with `--dry-run`, `--local` prints one runnable
command per fold instead of running them, which you can pipe into something
like `simple_gpu_scheduler` to fan a personal multi-GPU box out in
parallel:

```bash
python src/cherimoya/fit/launch.py --local --dry-run | simple_gpu_scheduler --gpus 0 1 2 3
```

`launch.py` submits one SLURM job per experiment, training that experiment's
folds sequentially within the job (folds with a completed model are skipped,
and an experiment with every fold already trained is not submitted at all).
`--time` is a flat budget for the whole job regardless of fold count (default
`48:00:00`, the `owners` partition's per-job cap).

Jobs are submitted with `--requeue`, so SLURM automatically resubmits a
pre-empted job (`akundaje`/`owners` are preemptible) instead of it just
dying. The submitted script re-checks each fold for a completed model at the
start of every run, not only once at submission time, so a requeued job
skips whatever folds finished before pre-emption and only retrains the rest
— cheap, since Cherimoya trains fast (~30s/epoch). "Completed" means a
`{experiment}.fold{fold}.final.torch` file, written exactly once at the very
end of training; the plain `.torch` file (no `.final`) is overwritten
throughout training whenever validation correlation improves, so it can
already exist after a single epoch and would wrongly look "done."

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
python src/cherimoya/benchmark/launch.py --local --dry-run | simple_gpu_scheduler --gpus 0 1 2 3
```

`launch.py` submits one SLURM job per experiment via `benchmark_cherimoya.py`,
mirroring [`src/bpnet/benchmark/launch.py`](../bpnet/benchmark/launch.py). By
default, jobs run via `apptainer exec --nv` against the Cherimoya Apptainer
image (see [`apptainer/`](apptainer/README.md)); `--native` uses SRCC's
`py-pytorch`/`py-triton` modules instead (see
[`sherlock_native/`](sherlock_native/README.md)). Experiments with any missing
fold model are skipped, as are experiments with an existing metrics JSON
(override with `--force`). `--local` bypasses SLURM entirely and runs each
experiment directly in the foreground (still via Apptainer by default; add
`--native` for a `uv run --extra cherimoya` invocation instead) — useful on a
GPU box you already have a shell on. Combined with `--dry-run`, `--local`
prints one runnable command per experiment instead of running them, which you
can pipe into something like `simple_gpu_scheduler` to fan a personal
multi-GPU box out in parallel.

## Architecture Sweep (deprecated)

**Deprecated**: this sweep predates the current Cherimoya v0.2.0 models and
training defaults, and we're not continuing this investigation with them.
`n_filters/` is left in place for reference but is not being run or
maintained going forward.

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
