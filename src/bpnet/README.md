# BPNet

Primary deployed model workflow for the PRO-cap atlas. BPNet models are trained
per experiment across seven chromosome folds, then benchmarked, attributed,
converted to browser tracks, and analyzed for motifs.

## Status

BPNet is the deployable model family for the atlas. The root `config.json`,
`configs/`, and trained BPNet artifacts are uploaded by `model_upload.py` to the
Hugging Face model repo.

## Prerequisites

- Completed preprocessing:
  - `configs/experiment_config.yaml`
  - `configs/chrom_splits.yaml`
  - processed strand BigWigs
  - processed and filtered peaks
  - GC-matched negatives
- Python environment from `environment.yml`
- Optional SLURM access for launchers

## Training

```bash
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0 --background gc:0.1
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0 --background ccre:0.05 --background gc:0.05
```

Background sources use repeatable `--background NAME:RATIO` arguments. Available
names are `gc` and `ccre`. The default is `gc:1/7`, giving one negative example
per seven positives.

Outputs:

```text
models/bpnet/{experiment}/
models/bpnet/{experiment}_{background_config}/
```

Submit all experiment/fold jobs through SLURM:

```bash
python src/bpnet/fit/launch.py
python src/bpnet/fit/launch.py --dry-run
python src/bpnet/fit/launch.py --min-reads 20000000
python src/bpnet/fit/launch.py --fit-args "--background gc:0.1"
```

The launcher contains hard-coded defaults for the Sherlock HPC environment,
including Sherlock/Kundaje-specific partition and module assumptions. Adjust it
before using another cluster.

## Benchmarking

```bash
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_gc0.1
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM --save-output
python src/bpnet/benchmark/launch.py --dry-run
```

Outputs:

```text
performance_metrics/bpnet/{model_dir_name}.json
predictions/bpnet/{model_dir_name}.npz
```

Additional K562 peak-set benchmarking and retraining scripts are retained under
`src/bpnet/fit/`, `src/bpnet/benchmark/`, and `src/bpnet/modisco/` for targeted
cross-peak experiments.

## Upload

```bash
python src/bpnet/model_upload.py
```

Uploads BPNet model artifacts, `configs/`, and the root `config.json` to
`adamyhe/procap-atlas`.

## Attributions

```bash
python src/bpnet/attribute/save_ohe.py -e ENCSR882DWM
python src/bpnet/attribute/run_ohe.py
python src/bpnet/attribute/run_ohe.py -j 8 --min-reads 10000000

python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --head count
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_gc0.1
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --reference-mode dinucleotide

python src/bpnet/attribute/launch.py --dry-run
python src/bpnet/attribute/launch.py --head profile --head count
```

Outputs:

```text
attributions/bpnet/{model_dir_name}_{head}.npz
attributions/bpnet/{experiment}_ohe.npz
```

`attribute_bpnet.py` defaults to the DeepLIFT genomic nucleotide-frequency null:
one soft reference per input sequence with the sequence's observed A/C/G/T
frequencies repeated at every position. Use `--reference-mode dinucleotide` and
`--n-shuffles` to reproduce the previous dinucleotide-shuffle baseline.

Convert observed-nucleotide attribution scores to BigWig:

```bash
python src/bpnet/attribute/attribution_to_bigwig.py -e ENCSR882DWM
python src/bpnet/attribute/attribution_to_bigwig.py -e ENCSR882DWM --head count
python src/bpnet/attribute/launch_bigwig_conversion.py --dry-run
python src/bpnet/attribute/launch_bigwig_conversion.py --head profile --head count
```

Output:

```text
attributions/bpnet/bigwigs/{experiment}_{head}.bigWig
```

## MoDISco

Run motif discovery after attribution and OHE files exist:

```bash
python src/bpnet/modisco/launch.py
python src/bpnet/modisco/launch.py --head profile --head count
python src/bpnet/modisco/launch.py --dry-run
python src/bpnet/modisco/launch.py --min-reads 20000000
python src/bpnet/modisco/launch.py -n 500000 -l 30 -w 500
```

Generate motif reports after `.h5` files are complete:

```bash
python src/bpnet/modisco/launch_report.py
python src/bpnet/modisco/launch_report.py --head profile
python src/bpnet/modisco/launch_report.py --dry-run
```

Recover time-limit failures:

```bash
python src/bpnet/modisco/relaunch_timeout.py --list
python src/bpnet/modisco/relaunch_timeout.py --dry-run
python src/bpnet/modisco/relaunch_timeout.py
python src/bpnet/modisco/relaunch_timeout.py --time 4-00:00:00
```

Outputs:

```text
modisco/bpnet/{experiment}_{head}.modisco.h5
modisco/bpnet/{experiment}_{head}.modisco/
logs/bpnet_modisco/
```

## Motif Clustering

[MotifCompendium](https://github.com/kundajelab/MotifCompendium) uses a separate
environment:

```bash
git clone https://github.com/kundajelab/MotifCompendium.git
cd MotifCompendium
conda env create -f environment_gpu.yml
conda activate motifcompendium-gpu
pip install -e .
```

Run clustering from the PRO-cap atlas repo after MoDISco outputs are available:

```bash
python src/bpnet/motifcompendium/cluster_motifs.py
python src/bpnet/motifcompendium/cluster_motifs_all.py
```

Outputs include similarity distributions and clustered motif reports under
`motifcompendium/`.

## Notes

- Fold `i` is held out for testing and fold `(i + 1) % 7` is used for
  validation.
- Chromosome folds are defined in `configs/chrom_splits.yaml`.
- Attribution launchers skip missing models and existing outputs.
- SLURM launch scripts are operational templates for Sherlock; review resources,
  modules, partitions, paths, and environment setup before porting.
