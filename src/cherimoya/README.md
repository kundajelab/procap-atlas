# Cherimoya PRO-cap models

Cherimoya training mirrors the BPNet workflow: one model per experiment and chromosome fold, using the processed PRO-cap BigWigs and peak BED files from `configs/experiment_config.yaml`.

The production entry point is:

```bash
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --background gc:0.1
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --n-filters 196 --batch-size 32
```

The script uses Muon for 2D weight matrices and AdamW for the remaining parameters, with warmup plus cosine learning-rate schedules. `src/cherimoya/fit/muon.py` is a local fallback for PyTorch versions that do not yet include `torch.optim.Muon`; `src/cherimoya/fit/test_muon.py` checks parity with upstream Muon when available.

Cluster launchers use Apptainer by default:

```bash
python src/cherimoya/fit/launch.py --dry-run
python src/cherimoya/fit/launch.py --min-reads 20000000 --fit-args "--max-epochs 100"
```

Benchmark and consolidation:

```bash
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM --save-output
python src/cherimoya/benchmark/consolidate_metrics.py
```

The `n_filters/` directory runs an architecture sweep over `16, 24, 36, 48, 64, 96, 196, 256` filters:

```bash
python src/cherimoya/n_filters/launch.py --dry-run
python src/cherimoya/n_filters/launch_benchmark.py --dry-run
python src/cherimoya/n_filters/consolidate_metrics.py
```

Historical note: the first Cherimoya models were trained while `cherimoya` was in early development, using commit `69f16dc7ff48ad094aafd4b93433972181c65d50`. Check out that commit only if you need to reproduce that initial model set.
