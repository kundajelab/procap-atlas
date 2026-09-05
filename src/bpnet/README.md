# BPNet

Primary deployed model workflow for the PRO-cap atlas. BPNet models are trained
per experiment across seven chromosome folds, then benchmarked, attributed,
converted to browser tracks, and analyzed for motifs.

## Status

BPNet is the deployable model family for the atlas. `hf/config.json`,
`configs/`, and trained BPNet artifacts are uploaded by `model_upload.py` to the
Hugging Face model repo.

## Prerequisites

- Completed preprocessing:
  - `configs/experiment_config.yaml`
  - `configs/chrom_splits.yaml`
  - processed strand BigWigs
  - processed and filtered peaks
  - GC-matched negatives
- Python packages from the root `uv` project (`pyproject.toml` / `uv.lock`).
  On Linux/Sherlock, this pins Torch to 2.6.0 and `pybigtools` to 0.2.5 for
  install compatibility.
- Optional SLURM access for launchers

For the Sherlock locus diagnostics notebook, including `uv` kernel registration
and Open OnDemand resource settings, see
[`notebooks/README.md`](../../notebooks/README.md).

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
before using another cluster. Generated jobs activate
`${PROCAP_ATLAS_ENV:-procap-atlas}` by default to expose `uv` and command-line
tools, then run Python entrypoints with `uv run --extra sherlock --frozen`
(flags before the command, e.g. `uv run --extra sherlock --frozen python
script.py`, not after); the conda
environment is not the Python dependency source of truth.

`launch.py` submits one SLURM job per (experiment, fold) pair, each trained
with `--requeue`, so a pre-empted job (`akundaje`/`owners` are preemptible)
is automatically resubmitted by SLURM. Since bpnet-lite's `fit()` has no
resume support, a requeued job just retrains that fold from epoch 0, but
first re-checks for a completed model in case the job actually finished just
before being marked pre-empted. "Completed" means a
`{experiment}.fold{fold}.final.torch` file, written exactly once at the very
end of training; the plain `.torch` file (no `.final`) is overwritten
throughout training whenever validation loss improves, so it can already
exist after a single epoch and would wrongly look "done."

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

Plot count-prediction diagnostics from benchmark outputs:

```bash
python src/bpnet/benchmark/plot_jitter.py
python src/bpnet/benchmark/plot_jitter.py --metric profile_jsd --sqrt-values
python src/bpnet/benchmark/generate_profile_jsd_bounds.py -e ENCSR882DWM --save-per-locus
python src/bpnet/benchmark/plot_jitter.py --metric profile_jsd --sqrt-values --bounds-tsv
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM --save-output
python src/bpnet/benchmark/plot_counts_prediction_scatter.py -e ENCSR882DWM
python src/bpnet/benchmark/plot_profile_jsd_cdf.py -e ENCSR882DWM --bounds-npz performance_metrics/bpnet_bounds/per_locus/ENCSR882DWM.npz
```

The jitter plot uses existing `performance_metrics/bpnet/*.json` files and
averages the selected metric across model folds for each experiment. Use
`--sqrt-values` with `--metric profile_jsd` to plot Jensen-Shannon distance
instead of divergence. `generate_profile_jsd_bounds.py` adds replicate
and average-profile JSD bounds for the jitter and CDF plots. The replicate
bound uses actual biological/technical replicate BigWigs, split into two
replicate groups. The scatter and profile JSD CDF plots require a saved
`predictions/bpnet/{experiment}.npz` file from `benchmark_bpnet.py --save-output`.

Additional K562 peak-set benchmarking and retraining scripts are retained under
`src/bpnet/fit/`, `src/bpnet/benchmark/`, and `src/bpnet/modisco/` for targeted
cross-peak experiments.

## Predicted Tracks

Generate final visualization tracks by predicting every filtered peak with every
fold checkpoint, averaging profile logits and log-count predictions across
folds, rescaling the averaged outputs to count-scale signal, and writing
plus/minus BigWigs:

```bash
python src/bpnet/predict/generate_predicted_tracks.py -e ENCSR882DWM
python src/bpnet/predict/generate_predicted_tracks.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_gc0.1
python src/bpnet/predict/launch.py --dry-run
python src/bpnet/predict/launch.py --min-reads 10000000
```

Output:

```text
predictions/bpnet/bigwigs/{model_dir_name}_pl.bigWig
predictions/bpnet/bigwigs/{model_dir_name}_mn.bigWig
```

Default experiment-model outputs (`{experiment}_pl.bigWig` and
`{experiment}_mn.bigWig`) are included by the hub upload workflow as
`predictions/bpnet/{experiment}_{strand}.bigWig`; see
[`src/hub/`](../hub/README.md).

## Upload

```bash
python src/bpnet/model_upload.py
```

Uploads BPNet model artifacts, `configs/`, and `hf/config.json` to
`adamyhe/procap-atlas`.

## Attributions

```bash
python src/bpnet/attribute/save_ohe.py -e ENCSR882DWM
python src/bpnet/attribute/run_ohe.py
python src/bpnet/attribute/run_ohe.py -j 8 --min-reads 10000000

python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --head count
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --head orientation
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_gc0.1
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --reference-mode dinucleotide

python src/bpnet/attribute/launch.py --dry-run
python src/bpnet/attribute/launch.py --head profile --head count --head orientation
```

Outputs:

```text
attributions/bpnet/{model_dir_name}_{head}.npz
attributions/bpnet/{experiment}_ohe.npz
```

`attribute_bpnet.py` defaults to the DeepLIFT genomic nucleotide-frequency null:
one soft PFM reference per input sequence with the sequence's observed A/C/G/T
frequencies repeated at every position. This uses tangermeme's callable
reference interface, so references are generated on the fly for each input batch
without passing a soft tensor through tangermeme's tensor-reference one-hot
validator.

Use `--reference-mode dinucleotide` and `--n-shuffles` to reproduce the previous
dinucleotide-shuffle baseline. In locus diagnostics, dinucleotide shuffles
sometimes produced active reference sequences with cryptic promoter-like
signals; for some loci these shuffled references were as active as, or more
active than, the genomic input, making the baseline reference-sensitive rather
than neutral.

Use `--head orientation` to attribute the profile orientation index,
`max(sum(plus), sum(minus)) / (sum(plus) + sum(minus))`. The wrapper computes
the index from joint profile probabilities using a DeepLIFT-compatible ReLU
form of the binary maximum. Count scaling is not applied because the single
predicted total-count factor cancels from the ratio.

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

By default, `launch.py` submits the 30 experiments with the largest processed
peak sets using the timeout-relaunch SLURM defaults (`--partition akundaje` and
`--time 6-23:00:00`). Remaining jobs use the standard launch defaults
(`--partition normal,akundaje,owners` and `--time 2-00:00:00`). Override the
large-job split with `--large-peak-top-n`, `--large-peak-partition`, or
`--large-peak-time`; use `--large-peak-top-n 0` to disable it.

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

[MotifCompendium](https://github.com/kundajelab/MotifCompendium) collapses the
per-experiment MoDISco motifs produced above into one atlas-wide, deduplicated
motif set, so a motif discovered in many experiments is represented once
rather than once per experiment. Hit Calling below runs Fi-NeMo per experiment
against that experiment's own MoDISco motifs, not this shared set directly
(much faster, and avoids Fi-NeMo's joint sparse regression competitively
reconstructing against cell-type-specific motifs an experiment never
expresses) -- instead, `cluster_motifs.py` also exports a mapping from every
experiment's own motifs to the atlas-wide cluster they were assigned to
(`motifcompendium_{head}_pattern_to_cluster.tsv`), which
`link_hits_to_compendium.py` uses after hit calling to relabel each
experiment's hits with the shared, cross-experiment-comparable cluster
identity.

MotifCompendium uses a separate external research environment and is
intentionally not part of the root `uv` project. Run its scripts with plain
`python` after activating its conda environment below — not through `uv run`,
which has no visibility into that environment:

```bash
git clone https://github.com/kundajelab/MotifCompendium.git
cd MotifCompendium
conda env create -f environment_gpu.yml
conda activate motifcompendium-gpu
pip install -e .
```

Run clustering from the PRO-cap atlas repo after MoDISco outputs are available
for the experiments you want included:

```bash
python src/bpnet/motifcompendium/cluster_motifs.py
python src/bpnet/motifcompendium/cluster_motifs.py --head profile
python src/bpnet/motifcompendium/cluster_motifs.py --min-reads 20000000 --blacklist ENCSR973QQI ENCSR882DWM
python src/bpnet/motifcompendium/cluster_motifs.py --across-threshold 0.85 --logo-report-top-n 0
```

`cluster_motifs_filtered.py` is unused/exploration-only (its entropy-based
motif-quality filter killed too many real motifs) and kept for reference only;
`cluster_motifs.py` is the pipeline in active use, and its pattern-to-cluster
mapping output feeds `link_hits_to_compendium.py` in Hit Calling below.

`cluster_motifs.py` runs one head (`count`/`profile`, both by default) at a
time:

1. Selects experiments from `configs/experiment_config.yaml`, dropping
   `--blacklist` IDs (default: `ENCSR973QQI`), any experiment whose
   `library_construction` metadata contains "uncapped", and any experiment
   below `--min-reads` total reads (default: 10M, read from
   `configs/n_reads.txt`).
2. Loads every surviving experiment's `modisco/bpnet/{experiment}_{head}.modisco.h5`
   into one `MotifCompendium` via `build_from_modisco` — this is also where
   MotifCompendium collapses each source pattern down to a single averaged CWM,
   discarding the individual seqlets underlying it (why hit-quality filtering
   in Hit Calling below falls back to a seqlet-free metric).
3. Clusters motifs in two passes: first `--within-threshold` (default 0.95)
   within each experiment, to collapse near-duplicate MoDISco patterns from the
   same model; then `--across-threshold` (default 0.90) across experiments,
   weighted by `num_seqlets`, to merge the same motif found in different
   experiments into one final cluster. Both thresholds are MotifCompendium
   pattern-similarity cutoffs, not sequence identity.
4. Exports the final per-cluster average CWMs as a real modisco-lite-format h5
   (`motifcompendium_{head}_cluster_averages.h5`, useful for calling hits
   atlas-wide by passing it to `call_hits_bpnet.py --modisco-h5` directly, but
   not required for the default per-experiment hit-calling flow below), plus
   a MEME-format version, cluster metadata/reports, and a per-experiment
   pattern-to-cluster mapping TSV (`motifcompendium_{head}_pattern_to_cluster.tsv`)
   that `link_hits_to_compendium.py` uses to relabel per-experiment hits with
   their atlas-wide cluster identity after the fact.

JASPAR annotation (`JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt`
under `data/`) is applied opportunistically for human-readable labels in the
reports; clustering itself does not depend on it, and it's silently skipped
if the file isn't present. `--no-gpu`/`--max-cpus`/`--max-chunk` forward to
`MotifCompendium.set_compute_options` for machines without a GPU or with
tighter resource limits.

Outputs, all under `motifcompendium/bpnet/`:

```text
motifcompendium_{head}_all_raw.mc                       # MotifCompendium save file, pre-clustering
motifcompendium_{head}_similarity_distribution.html      # pairwise motif similarity histogram
motifcompendium_{head}_all_clustered.mc                  # MotifCompendium save file, post-clustering
motifcompendium_{head}_cluster_averages.h5               # modisco-lite-format cluster-average CWMs (optional atlas-wide hit calling)
motifcompendium_{head}_cluster_averages.meme             # MEME-format version of the same clusters
motifcompendium_{head}_pattern_to_cluster.tsv            # experiment/local_motif_name -> compendium_motif_name, for link_hits_to_compendium.py
motifcompendium_{head}_cluster_metadata.tsv              # per-cluster n_motifs/total_seqlets/experiments/JASPAR label
motifcompendium_{head}_cluster_report.html               # MotifCompendium's own logo-heavy summary table
motifcompendium_{head}_cluster_summary.html              # lightweight all-clusters table, links to SVG logos
motifcompendium_{head}_cluster_logos/{fwd,rev}/*.svg      # per-cluster forward/reverse logos
motifcompendium_{head}_cluster_logo_paths.tsv            # cluster_final -> logo SVG path mapping
motifcompendium_{head}_clusters/{pos,neg}_cluster_NNNN.html  # per-cluster motif collection (opt-in)
```

The pipeline writes a full TSV plus a lightweight summary HTML for every
cluster, with links to exported forward/reverse SVG logo files for every
cluster — no motifs are dropped in `cluster_motifs.py`. To keep the
embedded-logo HTML report manageable, `motifcompendium_{head}_cluster_report.html`
is capped to the top 500 clusters by `total_seqlets` by default; use
`--logo-report-top-n 0` to include all cluster logos there (the plain
`_cluster_summary.html` table always includes every cluster). Per-cluster
motif collection HTML files are disabled by default; use `--per-cluster-html`
to write them. SVG logo export is enabled by default; use `--skip-svg-logos`
to disable it, or `--svg-logo-batch-size` to tune rendering batch size.

## Hit Calling

[Fi-NeMo](https://github.com/kundajelab/Fi-NeMo) calls individual motif
instances from attributions. By default it runs per experiment against that
experiment's own `modisco/bpnet/{experiment}_{head}.modisco.h5` (the same
scale of motif set — tens of motifs — [Kelly Cochran's ProCapNet
run_finemo.py](https://github.com/kellycochran/procapnet_allscripts/blob/main/GENCODE/src/attributions_genomewide/run_finemo.py)
used), not the atlas-wide MotifCompendium compendium: Fi-NeMo's joint sparse
regression makes every motif in the h5 compete for the same residual, so
throwing in every experiment's motifs at once is both far slower and prone to
competitively reconstructing with cell-type-specific motifs a given
experiment never actually expresses. A hit's `motif_name` here (e.g.
`pos_patterns.pattern_3`) is therefore only meaningful within that one
experiment — run `link_hits_to_compendium.py` at the end of this section to
relabel hits with the atlas-wide MotifCompendium cluster identity for
cross-experiment comparability. `finemo` is a regular `uv` project dependency
(Linux only; its `pyBigWig` dependency has no macOS wheel). Run after MoDISco
has produced per-experiment motifs for the desired head:

```bash
python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM
python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM --head count
python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_gc0.1
python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM --modisco-h5 motifcompendium/bpnet/motifcompendium_profile_cluster_averages.h5

python src/bpnet/hitcall/launch.py --dry-run
python src/bpnet/hitcall/launch.py --head profile --head count
python src/bpnet/hitcall/launch.py --min-reads 20000000
python src/bpnet/hitcall/launch.py --min-trim-len 6  # apply compute_trim_floor.py's floor
```

Outputs:

```text
hitcalls/bpnet/{model_dir_name}_{head}/peaks.narrowPeak
hitcalls/bpnet/{model_dir_name}_{head}/regions.npz
hitcalls/bpnet/{model_dir_name}_{head}/hits.tsv
hitcalls/bpnet/{model_dir_name}_{head}/hits_unique.tsv
hitcalls/bpnet/{model_dir_name}_{head}/hits.bed
hitcalls/bpnet/{model_dir_name}_{head}/peaks_qc.tsv
hitcalls/bpnet/{model_dir_name}_{head}/motif_data.tsv
hitcalls/bpnet/{model_dir_name}_{head}/motif_cwms.npy
hitcalls/bpnet/{model_dir_name}_{head}/parameters.json
```

`peaks.narrowPeak`/`regions.npz` depend only on the experiment/head/
`--region-width`, not on trimming, so they're cached directly in
`{model_dir_name}_{head}/` and reused across every trim configuration for
that (experiment, head) rather than being regenerated per trim setting.
`finemo call-hits`'s own output (everything else above) does depend on
trimming, so it moves into a trim-suffixed subdirectory whenever hit calling
used non-default trimming, so a rerun with different trim settings doesn't
silently overwrite a previous run's hits: `{model_dir_name}_{head}/trim{threshold}/`
for a non-default `--cwm-trim-threshold`, `.../trimthresh-{file_stem}/` for
`--cwm-trim-thresholds`, and/or `.../trimcoords-{file_stem}/` for
`--cwm-trim-coords` (e.g. `--min-trim-len 6` through `launch.py` writes hits
to `{model_dir_name}_{head}/trimcoords-motifcompendium_{head}_trim_coords_min6bp/`,
while still reusing that experiment's existing `regions.npz`). Default
trimming keeps the plain flat layout above (`call-hits` output stays directly
in `{model_dir_name}_{head}/`, unchanged). `report_bpnet.py`/`launch_report.py`
take the same `--cwm-trim-threshold`/`--cwm-trim-thresholds`/
`--cwm-trim-coords`/`--min-trim-len` values purely to resolve this same
directory layout, not to re-derive trimming themselves.

`call_hits_bpnet.py` first rebuilds the peak coordinates behind the saved
`{experiment}_ohe.npz`/attribution arrays: `extract_loci` (used by
`save_ohe.py`/`attribute_bpnet.py`) silently drops peaks that fall off a
chromosome end or overlap the blacklist, so `peaks.narrowPeak` is regenerated
from the same filtering rather than reusing `filtered_peaks` directly, keeping
row order aligned with the saved arrays. Default settings (`--region-width
2114`, i.e. the model's full input window rather than Fi-NeMo's own 1000bp
default; `--global-lambda 0.7`; `--cwm-trim-threshold 0.3`; `--batch-size 2000`)
all follow [Kelly Cochran's ProCapNet
run_finemo.py](https://github.com/kellycochran/procapnet_allscripts/blob/main/GENCODE/src/attributions_genomewide/run_finemo.py),
since the default motif source is now the same per-experiment scale her run
used. If you override `--modisco-h5` to point at the atlas-wide
MotifCompendium compendium instead, lower `--batch-size` a lot (e.g. `16`) —
that much larger motif set makes GPU memory per batch far higher, and
2000/500/64 all reliably OOM'd on a 44GB GPU against it.

Use `--cwm-trim-thresholds`/`--cwm-trim-coords` (Fi-NeMo's `-T`/`-R`) to
override trimming for specific motifs if any come out over-trimmed by the
default threshold — short core-promoter motifs (e.g. Initiator elements) are
particularly at risk, which is why Kelly Cochran's ProCapNet run patched
Fi-NeMo's trimming with a minimum-length floor that the current Fi-NeMo
release does not have built in.

Generate a ready-to-use `--cwm-trim-coords` floor file with
`compute_trim_floor.py`: it replicates Fi-NeMo's own `trim_motif` against a
motif h5, symmetrically widens (clamped to the untrimmed motif width) any
motif trimmed below `--min-len` bp, and writes only the widened motifs to the
output TSV — everything else keeps Fi-NeMo's default trimming. Pass
`-e/--experiment` to compute it against that experiment's own per-experiment
motif set, matching `call_hits_bpnet.py`'s default motif source (omit it to
compute against the atlas-wide compendium instead, for use with
`--modisco-h5` overrides):

```bash
python src/bpnet/hitcall/compute_trim_floor.py -e ENCSR882DWM --head profile
python src/bpnet/hitcall/compute_trim_floor.py -e ENCSR882DWM --head count --min-len 8
python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM --cwm-trim-coords modisco/bpnet/ENCSR882DWM_profile_trim_coords_min6bp.tsv
```

`launch.py --min-trim-len BP` wires this in per experiment: for each
(experiment, head), it looks up `compute_trim_floor.py -e`'s output for that
`BP` under `modisco/bpnet/` and passes it as that job's `--cwm-trim-coords`,
skipping (and counting separately) any experiment/head whose floor file
hasn't been generated yet. Run `compute_trim_floor.py -e {experiment} --head
{head} --min-len {BP}` first for every experiment/head you plan to launch.

After `call_hits_bpnet.py`, run `report_bpnet.py` to QC and filter hits by
per-motif CWM similarity, following the same principle as the [Human
Development Multiomic Atlas fetal-atlas
paper](https://github.com/GreenleafLab/HDMA/blob/main/code/03-chrombpnet/02-compendium/06b-reconcile_hits.py):
drop all hits for any motif whose hit-derived CWM correlates poorly with the
reference motif CWM (a real, data-driven quality signal for spurious/noisy
motifs, computed from the hits actually called rather than the motif's shape
at discovery time). This runs `finemo report --no-recall` for consistency
between both motif sources it can run against — the default per-experiment
modisco.h5 does retain TF-MoDISco seqlets, but the atlas-wide MotifCompendium
cluster-average h5 (`--modisco-h5` override) doesn't, so seqlet-recall isn't
always available — and drops hits for any motif at or below
`--cwm-similarity-threshold` (default 0.9, matching HDMA):

```bash
python src/bpnet/hitcall/report_bpnet.py -e ENCSR882DWM
python src/bpnet/hitcall/report_bpnet.py -e ENCSR882DWM --head count
python src/bpnet/hitcall/report_bpnet.py -e ENCSR882DWM --cwm-similarity-threshold 0.85

python src/bpnet/hitcall/launch_report.py --dry-run
python src/bpnet/hitcall/launch_report.py --head profile --head count
python src/bpnet/hitcall/launch_report.py --report-args '--cwm-similarity-threshold 0.85'
```

`launch_report.py` is a separate launcher from `hitcall/launch.py`, mirroring
`modisco/launch.py` vs `modisco/launch_report.py`: `finemo report` doesn't use
a GPU, so it runs as its own cheap CPU-only SLURM job (`-C NO_GPU`) rather
than being folded into hit calling's GPU job, and the `--cwm-similarity-threshold`
QC cutoff stays quick to retune without rerunning hit calling itself.

Outputs:

```text
hitcalls/bpnet/{model_dir_name}_{head}/report/                          # finemo report on the pre-filter hits: motif_report.tsv, motif_occurrences.tsv, CWM logos, report.html
hitcalls/bpnet/{model_dir_name}_{head}/report/cwm_similarity_distribution.png  # cwm_similarity histogram with the drop threshold marked
hitcalls/bpnet/{model_dir_name}_{head}/hits_filtered.tsv                # hits with low-similarity motifs removed
hitcalls/bpnet/{model_dir_name}_{head}/comparison/pre_filter/            # hit-stat/peak-distribution/co-occurrence plots on the pre-filter hits
hitcalls/bpnet/{model_dir_name}_{head}/comparison/post_filter/           # same plots on hits_filtered.tsv, for a direct before/after comparison
```

As above, `{head}` gains a trim-suffixed subdirectory (e.g.
`.../trimcoords-{file_stem}/report/...`) whenever hit calling used
non-default trimming; `report_bpnet.py` writes these outputs alongside
whichever `hits_unique.tsv` it read.

`finemo report`'s own `report.html` only visualizes the pre-filter hit set
(`hits_unique.tsv`); there's no built-in visualization of the post-filter
`hits_filtered.tsv` or a side-by-side comparison. `report_bpnet.py` closes
that gap by re-running Fi-NeMo's own plotting functions
(`plot_hit_stat_distributions`, `plot_hit_peak_distributions`,
`plot_peak_motif_indicator_heatmap`) directly on both the pre- and
post-filter hits, so `comparison/pre_filter/` and `comparison/post_filter/`
are directly comparable panel-by-panel.

If `call_hits_bpnet.py` was run with `--cwm-trim-thresholds`/
`--cwm-trim-coords` overrides, pass the same `--cwm-trim-threshold` here —
`finemo report` only exposes a single global threshold, so per-motif
overrides from `compute_trim_floor.py` can't be exactly reproduced at report
time, and `cwm_similarity` for those specific motifs may be computed against
a slightly different template width than was actually used to call hits.

Finally, run `link_hits_to_compendium.py` to relabel each experiment's
per-experiment hits with the atlas-wide MotifCompendium cluster identity they
belong to (`motifcompendium_{head}_pattern_to_cluster.tsv` from Motif
Clustering above), so `hits_linked.tsv` is directly comparable across
experiments the way calling hits against the shared compendium directly used
to be — without paying the cost/pathologies of calling hits against every
experiment's motifs at once:

```bash
python src/bpnet/hitcall/link_hits_to_compendium.py -e ENCSR882DWM
python src/bpnet/hitcall/link_hits_to_compendium.py -e ENCSR882DWM --head count
python src/bpnet/hitcall/link_hits_to_compendium.py -e ENCSR882DWM --min-trim-len 6

python src/bpnet/hitcall/launch_link.py --dry-run
python src/bpnet/hitcall/launch_link.py --head profile --head count
python src/bpnet/hitcall/launch_link.py --min-trim-len 6
```

It prefers `hits_filtered.tsv` (post `report_bpnet.py` QC) if present, falling
back to `hits_unique.tsv` otherwise, and adds a `compendium_motif_name` column
(e.g. `pos_patterns.42`) alongside the original per-experiment `motif_name`
(e.g. `pos_patterns.pattern_3`) rather than replacing it, so both identities
stay available:

```text
hitcalls/bpnet/{model_dir_name}_{head}/hits_linked.tsv
```

Requires `cluster_motifs.py` to have already been run for the requested head
(it builds the mapping from every experiment's own motifs, so needs rerunning
whenever new experiments are added) and `call_hits_bpnet.py` to have been run
for this experiment/head with the default per-experiment motif source, not
`--modisco-h5` pointed at the compendium. `launch_link.py` mirrors
`launch_report.py`: no GPU needed, so it runs as its own cheap CPU-only SLURM
job.

## Notes

- Fold `i` is held out for testing and fold `(i + 1) % 7` is used for
  validation.
- Chromosome folds are defined in `configs/chrom_splits.yaml`.
- Attribution launchers skip missing models and existing outputs.
- SLURM launch scripts are operational templates for Sherlock; review resources,
  modules, partitions, paths, and environment setup before porting.
