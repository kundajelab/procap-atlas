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
   below `--min-reads` total reads (**default: 0, i.e. no depth filter**, read
   from `configs/n_reads.txt`). The atlas compendium was therefore built over
   all 219 QC-passing experiments; the >10M-read restriction to 198 is applied
   by the downstream analyses in [`src/analysis/`](../analysis/README.md), not
   here.

Use `--out-dir` to build a variant compendium without disturbing the existing
one. Outputs default to `motifcompendium/bpnet/` and a rerun there overwrites
in place, which is not recoverable cheaply -- the `.mc` save files alone run to
hundreds of MB per head (1.5 GB for the profile raw), so copying the directory
first is not a practical alternative:

```bash
python src/bpnet/motifcompendium/cluster_motifs.py --head count \
    --min-reads 10000000 --out-dir motifcompendium/bpnet_198 \
    --skip-svg-logos
```

`--logo-report-top-n 0` is **not** a cheap setting: 0 means *no cap*, so every
cluster's logo gets embedded in the HTML report (24 MB on the real count head).
The default of 500 is already the cheap path, and a small positive number is
cheaper still.

Two things to know about a variant compendium. `cluster_final` ids are **not
stable across runs**, so its hits are not comparable until
`hitcall/launch_link.py` has been rerun against its own
`pattern_to_cluster.tsv`; and the logo paths in its
`cluster_logo_paths.tsv` are relative to its own `--out-dir`, so
`src/analysis/motif_redundancy.py` needs `--logo-root` pointed there to embed
logos.
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

#### Cluster redundancy: measured, not corrected

Compendium clusters include near-duplicate variants of one motif. This is
measured and stated rather than corrected; a similarity-based collapse was
built, tested and rejected. Recorded here so it is not re-litigated.

**What redundancy costs.** The compendium *relabels* Fi-NeMo hits
([`../hitcall/link_hits_to_compendium.py`](../hitcall/link_hits_to_compendium.py))
rather than serving as the scan set, so near-duplicate CWMs never compete for
a site and no hit is suppressed. The cost is to **prevalence**: if one
experiment's TATA links to cluster 8 and another's to cluster 21, "cluster 8"
is not the same motif atlas-wide, and a count of how many experiments share a
motif splits across the family.

**How large it is.** On the profile head (v1.1.0, within 0.95 / across 0.90),
families identified by eye span far more clusters than they appear to. Taking
mean motif-to-motif similarity between clusters and comparing it to each
cluster's own internal similarity
(`gap = min(within_a, within_b) - cross_ab`), TATA spans >= 8 clusters
(gaps -0.001..+0.034 among 8/21/48 alone) and AP-1 >= 7 (+0.005..+0.046).
CA-Inr is the counter-example: clusters 15 and 17 sit at gap +0.155 and are
genuinely separable, so not every visually similar pair is redundant.

**Lowering the thresholds does not fix it.** Sweeping `--across-threshold`
with `--within-threshold` held at 0.95, tracking where each family's motifs
land and how many foreign motifs join them:

| across | clusters | prev>=2 | TATA | AP-1 | ETV | CA-Inr |
|---|---|---|---|---|---|---|
| **0.90** | 5529 | **987** | **3c (+0)** | **4c (+0)** | **3c (+0)** | **3c (+0)** |
| 0.8875 | 5097 | 975 | 5c (+38) | 7c (+35) | 3c (+11) | 5c (+34) |
| 0.875 | 4718 | 945 | 4c (+64) | 6c (+20) | 5c (+86) | 4c (+38) |
| 0.85 | 4016 | 936 | 3c (+37) | 5c (+30) | 3c (+95) | 5c (+51) |

The total falls (5529 -> 4016) but the target families go *up* and
non-monotonically -- TATA 3->5->4->3, AP-1 4->7->6->5 -- while absorbing
foreign motifs and losing reproducible clusters (987 -> 936 at prevalence
>= 2). The threshold only decides which **edges exist**; `cpm_leiden`'s
`resolution_parameter` (fixed at 1.0) decides how aggressively communities
form. Lowering the threshold hands Leiden a different graph, so it returns a
different partition rather than a coarser one. **0.90 is the best of the four
on every measure**, and lowering `--within-threshold` is worse still: it
changes *what* the across stage clusters, so the comparison is confounded
(0.925/0.8875 gave TATA 5c(+107), CA-Inr 6c(+51)).

`--within-threshold` is also close to a no-op at these values: 0.925 leaves
13,956 within-clusters from 14,691 motifs.

**Why not a post-hoc collapse.** One was implemented and discarded:

- Its threshold was **fitted and then validated on the same families**,
  pinned between AP-1's worst internal pair (+0.046) and CA-Inr's tightest
  separable pair (+0.053). "AP-1 merges, CA-Inr does not" at t=0.05 is
  arithmetic, not evidence.
- Single-linkage **percolates** on these families, which are gradients rather
  than cliques: the largest connected component grew 4 -> 15 -> 48 -> 70 ->
  137 as the gap threshold went 0.00 -> 0.05, and the group *count* peaked at
  79 (t=0.03) then fell to 58 (t=0.05) while coverage kept rising, i.e.
  groups fusing. Greedy representative selection avoids chaining but is
  order-dependent.
- Even at its best it only took TATA 7 -> 2, because no threshold in the
  admissible window fuses a gradient without merging something distinct.

**Two traps for anyone revisiting this.** Do not measure redundancy on
similarity between *cluster averages*: a label-permutation null on the count
head had randomly-composed clusters scoring **higher** pairwise than real
ones (60.6% vs 58.1% of nearest neighbours above 0.90), because averaging
blurs toward a bland consensus and bland averages resemble each other. And do
not group candidate duplicates by JASPAR name; see below.

**What is reported instead.** Raw cluster counts, with JASPAR-name collapse
as a **lower bound** -- names can only merge clusters, never split them, so
the distinct-name count bounds the number of distinct motifs from below and
the raw count from above. Rarefaction is count-head only. Manual
refinement of annotations into a consensus set, for both the report and
Fi-NeMo labelling, is deferred until after the preprint.

JASPAR names are the wrong granularity for *measuring* redundancy, which is
why they bound rather than correct it. A name is a TF-identity label, and the
two disagree in both directions: names **split what similarity fuses** (the
ETV family is cluster 13/ETV7 plus 35 and 10, both ELF2, at gaps +0.030 and
+0.007) and **fuse what similarity separates** (clusters 4 and 17 are both
Hand1::Tcf3 at gap +0.053). Some elements have no JASPAR entry at all --
CA-Inr is a core promoter initiator, so its clusters come back as
Hand1::Tcf3, ISL2 and Hand1::Tcf3 at 0.83-0.86, since the lookup must return
some TF. Where a canonical entry exists the names are right (TBP for TATA,
ETS for ETV), and distinguishing CRE from TRE variants is the lookup working
rather than failing.

#### Comparing two builds' clusterings

`compare_clusterings.py` quantifies how much two builds actually disagree.
Written to decide what to do about the v1.0.19 `k_centroids` default, but it
applies to any pair of builds.

It compares `pattern_to_cluster.tsv`, **not** `cluster_metadata.tsv`.
`cluster_final` ids are not stable across runs, so metadata rows cannot be
joined at all; the pattern mapping keys on `(experiment,
local_motif_name)`, which is the same MoDISco pattern in every build.

```bash
python src/bpnet/motifcompendium/compare_clusterings.py --head count \
    leiden=mc_leiden/motifcompendium_count_pattern_to_cluster.tsv \
    capped5=mc_capped5/motifcompendium_count_pattern_to_cluster.tsv \
    uncapped=motifcompendium/bpnet/motifcompendium_count_pattern_to_cluster.tsv
```

`frac_same_clustermates` is the fraction of patterns whose set of
cluster-mates is identical in both builds. It is invariant to relabelling and
directly readable, but **read it alongside ARI, not instead of it**: it
amplifies. One pattern moving from cluster A to cluster B flags every member
of both, so at the count head's mean cluster size of ~5.9 (5,639 patterns /
950 clusters) a single reassignment marks up to ~12 patterns as changed.
Dividing `n_patterns_moved` by the mean size of two clusters gives a lower
bound on the number of actual reassignments. Merge/split counts are reported
directionally, so a cluster splitting in two is distinguishable from two
merging.

`posneg` is **not** in the cluster key. `cluster_final` is a global label —
clustering runs `cluster_within="model"` then `cluster_on`, neither of which
stratifies on `posneg` — so a cluster may hold both pos and neg motifs. An
earlier version keyed on `f"{posneg}:{cluster_final}"` and split those,
reporting the count head as 950 clusters where `cluster_metadata.tsv` has
945 rows (exactly the 5 mixed clusters). Every cluster count this README
attributes to `compare_clusterings.py` output — the three-way table, the
capped25 table, the v1.1.0 table — is therefore 950 where the build's own
metadata says 945. The ARI/AMI figures compared both builds under the same
inflated key, so their ordering stands, but the counts do not.

#### The three-way count-head comparison

The controlled experiment for the `k_centroids` question. Clustering is ~4 min
per run at count-head scale, and `--skip-svg-logos --logo-report-top-n 0`
would be wrong here (0 means *no cap*, not "skip"); pass a small cap instead,
or skip the logos and accept the report:

```bash
# Run against MotifCompendium v1.0.19; --kmeans-iterations has since been
# removed (see below), so reproducing steps 2 and 3 needs `git show 30adbab`.
MC=src/bpnet/motifcompendium/cluster_motifs.py
# 1. pre-v1.0.19 behaviour: Leiden only
python $MC --head count --algorithm cpm_leiden \
    --skip-svg-logos --logo-report-top-n 1 --out-dir mc_leiden
# 2. Leiden + bounded refinement
python $MC --head count --kmeans-iterations 5 \
    --skip-svg-logos --logo-report-top-n 1 --out-dir mc_capped5
# 3. the current uncapped build already exists in motifcompendium/bpnet/

python src/bpnet/motifcompendium/compare_clusterings.py --head count \
    leiden=mc_leiden/motifcompendium_count_pattern_to_cluster.tsv \
    capped5=mc_capped5/motifcompendium_count_pattern_to_cluster.tsv \
    uncapped=motifcompendium/bpnet/motifcompendium_count_pattern_to_cluster.tsv
```

Measured, 2026-09-14 (count head, 5,639 patterns, `--across-threshold` 0.90):

| a | b | frac_same_clustermates | n_patterns_moved | ARI | AMI |
|---|---|---|---|---|---|
| leiden | capped5 | 0.2451 | 4,257 | 0.7773 | 0.9013 |
| leiden | uncapped | 0.2444 | 4,261 | 0.7657 | 0.8976 |
| capped5 | uncapped | 0.7349 | 1,495 | 0.9601 | 0.9757 |

**All three builds produced exactly 950 clusters.** Chained after
`cpm_leiden`, `k_centroids` receives the Leiden partition as
`init_membership`, and `clustering.py` then sets `k` from it:

```python
_, init_membership = np.unique(init_membership, return_inverse=True)
k = len(np.unique(init_membership))  # Set k
seeds = [None]      # No random seed, run once
init_method = None  # No initialization method
```

So `k` is inherited, not chosen — and `kmeans++`/`maximin` init and the
two-seed default (`seeds=[100, 200]`) are all bypassed on the chained path.
`k` is an **upper bound that can only shrink**: each iteration re-runs
`_remap_membership` first, so a cluster that gets fully emptied disappears and
the next round builds one fewer centroid. Assignment is `argmax` over the
surviving centroids, so nothing can ever split. Two consequences:

- The refinement is doing real work — Leiden-only and refined partitions
  differ at ARI 0.77, roughly 350+ actual reassignments — and **five
  iterations capture nearly all of it.** `leiden`-vs-`capped5` and
  `leiden`-vs-`uncapped` are indistinguishable (0.2451 vs 0.2444), so by
  iteration 5 the partition has already moved as far from Leiden as it ever
  gets. The residual `capped5`-vs-`uncapped` difference (ARI 0.96, >= ~125
  reassignments) is the slow tail.
- It **cannot** explain the count head going from 944 to 946 clusters, which
  this README previously attributed to it. The stage is monotone
  non-increasing in `k`, so it can never raise a cluster count; here it
  emptied nothing and left all 950 standing. That drift has some other cause
  (most likely the `--across-threshold` 0.85 -> 0.90 change, or a differing
  build set); it is still unexplained.

How to read the result:

- **capped ~= uncapped** -> cap it and rebuild profile with a bounded
  refinement. It is doing real work and converges quickly; only the tail is
  pathological. (This is what happened; v1.1.0 then made the bound
  unnecessary.)
- **leiden ~= uncapped** -> `k_centroids` changes nothing here, and
  `--algorithm cpm_leiden` is both the cheapest and the safest option.
- **all three differ materially** -> this is a question about which partition
  is correct, not which is faster, and the cluster-average logos are the
  evidence to look at (`k_centroids` exists to make members resemble the
  average that represents them, which is exactly what Figure 2c draws).

Whichever is chosen, **both heads must use the same setting** or a
count-vs-profile contrast confounds head with clustering algorithm.

#### Why `k_centroids` explodes at profile-head scale

Traced through MotifCompendium `7e9d1c2`. Two defects, both in the per-iteration
centroid step, and the cost model explains the count/profile gap exactly.

**1. Every iteration recomputes a full k x k similarity that is never read.**
`k_centroids_clustering` calls `mc.cluster_averages(...)` per iteration
(`utils/clustering.py:871`), which ends in `build(cluster_motif_avgs, metadata,
safe=False)` (`MotifCompendium.py:1889`) — and `build` unconditionally computes
all-pairs similarity on what it is given (`MotifCompendium.py:220`):

```python
similarity, alignment_rc, alignment_h = utils_similarity.compute_similarities(
    [motifs], [(0, 0)]
)[0]
```

The loop passes `compute_quality_stats=False`, so that k x k matrix is
discarded unused. The only matrix the iteration needs is the N x k
motif-to-centroid block computed right after.

Counting motif-pair alignments, with `r = k/N`, per-iteration cost is
`N*k + k^2 = N^2 * (r + r^2)` against a one-time `N^2` build:

| head | N | k | r | per-iteration | as % of full build |
|---|---|---|---|---|---|
| count | 5,639 | 950 | 0.168 | 6.3 M | **19.7%** |
| profile | 14,691 | 5,527 | 0.376 | 111.7 M | **51.8%** |

A profile iteration is **17.9x** a count iteration, and each one costs about
half of building the entire compendium from scratch. The profile head is
punished twice: N^2 is 6.8x larger *and* its motifs are more distinct, so
`r` more than doubles and the wasted `k^2` term quadruples relative to the
useful term. Twenty iterations is ten full compendium builds.

**2. The centroid's alignment frame is the cluster's lowest-index member, so
the centroid step does not optimise the assignment step's objective.**
`cluster_averages` takes the alignment vectors from row 0 of the cluster's
submatrix (`MotifCompendium.py:1840-1846`):

```python
alignment_rc_c = self.alignment_rc[c_idxs, :][:, c_idxs][0, :]
alignment_h_c  = self.alignment_h[c_idxs, :][:, c_idxs][0, :]
```

`average_motifs` then aligns every member to that member before averaging. So
when a cluster's lowest-index member leaves, the survivors are re-averaged in
a **different frame** and the centroid jumps for reasons unrelated to the
membership change. Lloyd's algorithm only converges because the centroid step
minimises the same objective the assignment step evaluates; that guarantee is
void here, which is what permits a limit cycle rather than convergence.
`n_iterations=-1` exits only on `np.array_equal(membership_old,
membership_new)` (`utils/clustering.py:890`), so a cycle never terminates, and
`score_old` is assigned but never read — there is no objective-based stopping
rule to fall back on.

Incidentally that indexing is itself quadratic: it materialises a `(|c|, N)`
gather and then a `(|c|, |c|)` slice to read one row. Summed over clusters
that is `N^2` gathered elements per iteration, per matrix, where
`self.alignment_rc[c_idxs[0], c_idxs]` would be `O(|c|)`.

Both are worth reporting upstream. Neither is worked around by `--algorithm
cpm_leiden` alone (that drops the stage, and the measured comparison shows the
stage changes the partition materially), so the cap below stays the
near-term answer.

#### Choosing the iteration cap

**Superseded by v1.1.0**, which makes the refinement bound itself and led to
`--kmeans-iterations` being removed. Kept because the measurement below is
what established that the pre-fix refinement converges on the count head but
not on the profile head — the evidence for the alignment-frame diagnosis.

The three-way result is "capped is close to uncapped but not equal", so the
open question is where the refinement actually converges. The count head is
the cheap place to find out — uncapped converges there in ~4 min, so a cap can
be compared against the converged partition directly:

```bash
python $MC --head count --kmeans-iterations 25 \
    --skip-svg-logos --logo-report-top-n 1 --out-dir mc_capped25
python src/bpnet/motifcompendium/compare_clusterings.py --head count \
    capped25=mc_capped25/motifcompendium_count_pattern_to_cluster.tsv \
    uncapped=motifcompendium/bpnet/motifcompendium_count_pattern_to_cluster.tsv
```

`k` needs no flag: on the chained path it is always inherited from Leiden
(see above), and `--kmeans-iterations` bounded the refinement without
touching it.

Measured, 2026-09-14:

| a | b | frac_same_clustermates | n_patterns_moved | ARI | AMI |
|---|---|---|---|---|---|
| capped25 | uncapped | **1.0** | **0** | 1.0 | 1.0 |

**25 iterations reproduce the converged count-head partition exactly** — on
that head the cap was not an approximation at all, but byte-identical to the
uncapped output. This was the setting for both heads until v1.1.0 removed the
need for a cap.

The count head therefore converges in <= 25 iterations while the profile head
had not converged after 24 h on an A100. Per-iteration cost only accounts for
17.9x of that gap (see above), so the remainder is iteration *count*: the
profile head is running hundreds of iterations where the count head needs
tens. That is the predicted signature of the alignment-frame defect, and it
scales with cluster count because more clusters means more chances for a
cluster's lowest-index member to leave and re-frame the survivors.

Convergence at 25 is established **only for the count head**. For the profile
head the cap is a deliberate approximation until shown otherwise, so either
say so in the methods, or confirm it by running the profile head at 25 and
again at 50 and comparing — identical partitions mean iterations 26-50 changed
nothing. Do the 25 run first and let its wall time decide whether the
confirmation run is affordable.

#### MotifCompendium v1.1.0 (`next_version`): the k_centroids fix

Commit
[`ddd279f`](https://github.com/kundajelab/MotifCompendium/commit/ddd279f) on
the `next_version` branch fixes all three defects traced above: it adds
`select_alignments()` so a cluster's frame comes from its membership rather
than its row order, aligns each motif to the centroid it was assigned to
(making the averaging step the exact spherical-k-means M-step), drops the
throwaway `MotifCompendium` and its unread k x k similarity, and adds a
`_ConvergenceTracker` that stops on a repeated membership, a stalled
objective, or a `max_iterations=100` cap that applies even at
`n_iterations=-1`. An unbounded run is no longer possible.

**It is a version jump, not a patch.** `next_version` is v1.1.0 against
v1.0.19: 1,448 insertions across five files, including a new
`utils/composite.py` and 646 changed lines in `MotifCompendium.py`, plus a
serial-clustering bug fix and reworked clustering-quality calculations. The
branch is also a moving target, so **pin the commit**, do not track the
branch.

Two independent reasons "rerun with defaults" does not reproduce prior
outputs:

1. **The partition changes.** That is the fix working as intended.
2. **`cluster_averages`' alignment frame changes even at a fixed partition.**
   v1.1.0 adds `reference: str = "medoid"`; before it, a cluster average was
   always framed on the cluster's lowest-indexed member.
   `cluster_average_with_metadata` does not pass `reference`, so the default
   applies — which moves `cluster_averages.h5`, `cluster_averages.meme`, the
   cluster logos, Figure 2c and `select_motif_exemplars.py`. Passing
   `reference="first"` would reproduce the old framing, but the new default is
   the better one; the point is that it must be recorded, not avoided.

**`--kmeans-iterations` has been removed.** It existed to bound a loop that is
now self-bounding, and a cap would mask whether the fix actually converges —
which is the thing worth learning. `--algorithm` stays, since reproducing the
pre-v1.0.19 Leiden-only partition is still worth being able to do, and
`--algorithm-kwarg ALGORITHM.KEY=VALUE` (repeatable) replaces it as a
*diagnostic* escape hatch onto any per-step clustering argument — it sets no
defaults, so omitting it leaves the library's own defaults alone. It targets
a step by name and raises if that step is not in the algorithm list, since a
silently ignored kwarg looks like it applied. The resulting setting lands in
`cluster_metadata.tsv`'s `cluster_algorithm`, so a tuned build is not
indistinguishable from an untuned one. The `mc_capped5`/`mc_capped25`
builds are therefore no longer reproducible from the CLI; their outputs on
disk are the record, and `git show 30adbab` has the flag if it is ever needed
again. Watch stderr for
`membership is cycling`, `objective stopped improving`, and `did not converge
within 100 iterations`; the first two are now warnings rather than silent
behaviour.

One thing to watch: `_ConvergenceTracker.should_stop` stops the run at the
*first* iteration that fails to improve by more than `tol=1e-9`. Since
averaging is lossy the objective is not guaranteed monotone, so a run can dip
and recover; this rule would stop at the dip. It returns the best-scoring
iteration, so the result is sound, but it may under-iterate relative to true
convergence. Compare against a `max_iterations`-only run if the count head
comes back materially different from `mc_capped25`.

Measured, 2026-09-14 (count head, v1.1.0 defaults, no cap):

| a | b | frac_same_clustermates | n_patterns_moved | ARI | AMI |
|---|---|---|---|---|---|
| capped25 | v110 | 0.8837 | 656 | 0.9963 | 0.9950 |

**The fix is a refinement, not a re-partitioning.** 950 clusters again, ARI
0.9963, and ~55 actual reassignments (656 / ~12 amplification) out of 5,639
patterns, with splits near-symmetric in both directions (16 vs 15) — local
reshuffling, not systematic merging or splitting. In context on the
count head:

| comparison | ARI |
|---|---|
| leiden vs uncapped v1.0.19 | 0.766 |
| capped5 vs uncapped v1.0.19 | 0.960 |
| capped25 vs uncapped v1.0.19 | 1.000 |
| capped25 vs v1.1.0 | 0.996 |

v1.1.0's correction is roughly **a tenth the size of adding `k_centroids` at
all**, which is what a genuine bug fix should look like rather than a
different algorithm. Count-head downstream numbers should move very little.

**That run stopped early, so the comparison above confounds two changes.**
It emitted `k_centroids: objective stopped improving; returning the
best-scoring iteration`. In the v1.1.0 loop the membership-equality check
runs *before* `should_stop`, so a true fixed point breaks without warning:

```python
if np.array_equal(membership, membership_new):
    break                      # true fixed point, no warning
membership_next = _remap_membership(membership_new)
if tracker.should_stop(membership_next, score):
    break                      # the stall warning fires here
```

The warning therefore means membership was **still changing** when the run
quit. The v1.1.0 count-head partition is `tracker.best_membership` — the
best-scoring iteration — not a fixed point, whereas `mc_capped25` was a
genuine one (it equals the uncapped v1.0.19 build exactly). So the ARI 0.9963
gap mixes the alignment-frame fix with early stopping and cannot be
attributed to either.

The stall rule is too aggressive, and by the authors' own rationale: the
tracker exists *because* lossy averaging makes the objective non-monotone, so
a run can dip and recover — but `should_stop` exits at the **first** iteration
failing to improve by more than `tol=1e-9`. Knowing the objective can dip
calls for a patience counter (stop after *p* consecutive non-improvements),
not an immediate exit. Worth reporting upstream.

**One printed warning does not say how much stalled.** `mc.cluster` invokes
`k_centroids` once per `cluster_within` group — one per experiment, so 219
times for the within-model stage — plus once for the `cluster_on` stage, and
`warnings.warn`'s default filter prints a given warning once per code
location. A single line therefore means "at least one of ~220 calls", at an
unknown stage.

The stage is what matters. `cluster_on` produces `cluster_final` directly, so
a stall there moves the atlas partition; a stall inside one `cluster_within`
group perturbs only that experiment's own pre-clustering. `cluster_motifs.py`
now counts them per stage with `simplefilter("always")`, prints

```
count: clustering convergence -- within-model: 3 stalled; across-model: converged
```

and stamps that string into `cluster_metadata.tsv`'s `cluster_convergence`,
so a build records whether its own partition converged. Unrelated warnings
are re-emitted at their original location rather than swallowed.

To separate the two effects, re-run with the stall rule disabled and only the
cycle and `max_iterations` guards active. `tol=-inf` makes
`score <= previous + tol` unsatisfiable for any finite score:

```bash
python $MC --head count --algorithm-kwarg k_centroids.tol=-inf \
    --skip-svg-logos --logo-report-top-n 1 --out-dir mc_v110_notol
python src/bpnet/motifcompendium/compare_clusterings.py --head count \
    v110=mc_v110/motifcompendium_count_pattern_to_cluster.tsv \
    v110_notol=mc_v110_notol/motifcompendium_count_pattern_to_cluster.tsv
```

If those agree, early stopping cost nothing and ARI 0.9963 is the frame fix
alone. If they differ, `mc_v110` is under-iterated and the no-stall build is
the one to keep.

Note also that this run does not yet demonstrate the 85 h problem is fixed:
the count head converged before the fix too (`capped25 == uncapped`). **The
profile head is the discriminating test.**

**It passed.** On v1.1.0 the profile head cleared clustering and reached
`cluster_averages.h5` and report generation in a fraction of the time, against
85 h on an L40S and 24 h on an A100 without finishing under v1.0.19. Per the
cost model, removing the unread `k x k` similarity only accounts for ~27% of
a per-iteration saving, so the rest is iteration *count* — which confirms the
alignment-frame defect, not raw scale, was what the profile head was stuck on.
That asymmetry between heads is the strongest evidence for the diagnosis and
belongs in the upstream report: 950-cluster count head converged either way,
5,527-cluster profile head only after the fix.

Note that this understates the change to the *outputs*, because
`cluster_averages`' frame moved from row-0 to medoid for **all 950** clusters,
including the ones whose membership is unchanged. The cluster-average h5, the
MEME export and the Figure 2c logos therefore change more broadly than the
partition does.

Order of operations: rebuild the **count** head first and diff it against the
`mc_capped25` baseline with `compare_clusterings.py`. That is ~4 min and
sizes the change before committing the profile head to it. Then rebuild both
heads on the same pinned commit, and re-run everything downstream
(`plot_motif_rarefaction.py`, `motif_group_concentration.py`,
`select_motif_exemplars.py`, Figure 2, hit calling) — cluster ids are not
stable across builds, so nothing downstream carries over.

#### What a build's settings were, after the fact

Nothing in a compendium's outputs records how it was produced. The cluster
metadata, MEME export, cluster-average h5 and HTML reports are identical in
shape regardless of thresholds, and the only 0.85/0.90 strings in the reports
are JASPAR match scores. That matters because several inputs have changed over
the project's life, and every one of them moves the partition:

- **`--across-threshold`'s default changed from 0.85 to 0.90** on 2026-08-21,
  in commit `31d4f24`. A build predating that commit was clustered at 0.85
  unless overridden. The manuscript methods described 0.85 because they were
  written against a pre-August build.
- **MotifCompendium was updated mid-project, and one update changed
  clustering behaviour through a default.** v1.0.19 (commit
  [`7e9d1c2`](https://github.com/kundajelab/MotifCompendium/commit/7e9d1c2),
  2026-08-19) changed `mc.cluster`'s signature:

  ```python
  -        algorithm: str = "cpm_leiden",
  +        algorithm: list[str] | str = ["cpm_leiden", "k_centroids"],
  ```

  `cluster_motifs.py` passed no `algorithm`, so the update silently added a
  second stage: an **uncapped** k-means refinement (`n_iterations=-1`, exiting
  only when the membership vector is *exactly* equal between consecutive
  iterations, so an oscillation never terminates). Each iteration rebuilds
  every centroid and recomputes an N x k float64 similarity on the GPU.

  This is why a profile-head build ran for 85 h having previously completed on
  v1.0.18. The cost is invisible at count-head scale (5,639 motifs, ~4 min of
  clustering) and ruinous at profile-head scale (14,691 motifs). Note that the
  raw `.mc` sizes — 1.5 GB profile against 223 MB count — are *not* a linear
  read on motif count: the `.mc` holds three N x N matrices, so its size goes
  as n^2. 14,691 motifs predict 863 + 216 + 432 = 1,511 MB and 5,639 predict
  127 + 32 + 64 = 223 MB, both matching what is on disk.

  It was **not** the cause of the count head going from 944 to 946 clusters,
  which an earlier version of this README proposed. The three-way comparison
  above shows `k_centroids` inherits `k` from Leiden and can only reduce it.

  `--algorithm cpm_leiden` restores the pre-v1.0.19 behaviour. **Both heads
  must use the same setting** or a count-vs-profile contrast confounds head
  with clustering algorithm.

  Since Sep 2026 `cluster_metadata.tsv` carries five provenance columns —
  `mc_version`, `cluster_algorithm`, `cluster_reference`,
  `cluster_convergence`, `within_threshold`, `across_threshold` —
  so this class of change leaves a trace in the outputs. `mc_version` reads
  the installed distribution metadata, not `MotifCompendium.__version__`,
  which does not exist on any branch — stamping it via `getattr` recorded
  `"unknown"` on every build until this was fixed. `cluster_reference`
  records `cluster_averages`' alignment frame, which v1.1.0 made
  configurable. They are constant
  down each column and additive; the analysis scripts that read this table
  (`motif_group_concentration.py`, `plot_motif_rarefaction.py`,
  `select_motif_exemplars.py`) were verified to give byte-identical results
  with and without them, and compendia built before the change simply lack
  the columns.
- **The build set is `filters INTERSECT h5 files present when the job ran`.**
  `collect_modisco_paths` silently skips a selected experiment whose MoDISco
  output is not on disk yet, so the experiment set can change with no change to
  `--min-reads` or `--blacklist`. At the default `--min-reads 0` this is the
  only thing that can move it.
- **Compute options are partition inputs, not just performance knobs.** GPU
  float reductions are not order-deterministic and `--max-chunk` changes that
  order, so they can flip pairs sitting on the across-threshold boundary.

Clustering is also **not known to be reproducible**: no seed is passed to
`mc.cluster`, Leiden is a randomized algorithm, and the GPU path adds its own
nondeterminism. Two builds on identical inputs are not guaranteed to agree,
and that has never been tested here. Do not treat a small difference in
cluster count between two builds as evidence of anything until it is.

To compare two existing builds, use their metadata TSVs. `sum(n_motifs)` is
the number of input MoDISco patterns and is invariant to the clustering
threshold, while the row count is what the threshold moves, so equal input
motifs with materially different cluster counts points at the threshold or the
library version. The `experiments` column gives each build's experiment set
directly, so whether the inputs changed is a lookup rather than an inference.
Comparing the set of `(posneg, experiments)` signatures measures how much of
the partition actually agrees, which the row count alone cannot -- two clusters
merging while another splits nearly cancels in the total.

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

### Outputs are compressed

Hit-call tables are stored compressed: **gzip for `.tsv`, bgzip for `.bed`
and `.narrowPeak`**. Hit calling runs over all 224 experiments (198 is the
minimum-quality filtered set used for the integrative analyses, not the
hit-call set), with one output set per experiment x head x trim
configuration, so this is a large amount of disk.

`compressed_io.py` holds the three things every script needs:

- `resolve(path)` — takes the *logical* (uncompressed) path and returns
  whichever of it and its `.gz` sibling exists, preferring the compressed
  one. It warns when both exist, since that state follows an interrupted
  compression pass and the two can disagree; it raises naming **both**
  candidates when neither does.
- `write_tsv(frame, path)` — always writes `path.gz`, returning the path
  actually written so the caller's log line names the real file.
- `write_bgzip(frame, path)` — bgzip via the `bgzip` binary (`htslib`, already
  in `environment.yml`). It **raises** rather than falling back to plain gzip:
  a plain-gzip file named `.bed.gz` is indistinguishable until something tries
  to tabix-index it, and then fails far from here.

**Reads accept either form**, so a partially compressed tree still works —
builds predating compression, Fi-NeMo output compressed after the fact, and
freshly written `.gz` all coexist. `resolve_hits_path` resolves every stage
through it, which covers most consumers; `cleanup_hitcalls.py` compares
against `logical_name(path)` so its name sets still match compressed files.

This is the hazard `cleanup_hitcalls.py` originally called out: it compressed
only `hits.bed` because nothing read that back by exact filename, "so
gzipping it can't break any downstream script's file resolution the way
gzipping `hits_unique.tsv` would." `resolve()` is what removes that
constraint.

Two notes:

- **bgzip is a valid gzip stream**, so `gzip`, `pandas` and `polars` all read
  it transparently. Nothing tabix-indexes these yet; bgzip is used so that
  compressing them now does not foreclose it.
- **`peaks.narrowPeak` is ours but is *input* to `finemo extract-regions`**,
  which reads it with `polars.scan_csv`. That handles gzip (verified on
  polars 1.44.2), so a compressed cache works — but confirm the cluster's
  polars is recent, because the failure mode is breakage at region
  extraction rather than at write time.

Fi-NeMo's own writers are untouched: it writes `hits.tsv`, `hits_unique.tsv`,
`hits.bed` and `motif_report.tsv` uncompressed, and those are compressed
afterward.

### Calling hits

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

If `regions.npz` itself goes missing (e.g. deleted directly, or removed by a
disk cleanup) while `hits.tsv` and every later-stage file are still there,
don't rerun `call_hits_bpnet.py` to get it back: `finemo call-hits` has no
skip-if-unchanged logic, so that would unconditionally recall hits and
require redoing every downstream filter/report stage just to regenerate one
cache file. `extract_regions_bpnet.py` calls `call_hits_bpnet.py`'s own
`ensure_regions_npz` directly instead, rebuilding only `peaks.narrowPeak`/
`regions.npz` from the experiment's filtered peaks and saved OHE/attribution
arrays, with `finemo call-hits` never invoked:

```bash
python src/bpnet/hitcall/extract_regions_bpnet.py -e ENCSR882DWM
python src/bpnet/hitcall/extract_regions_bpnet.py -e ENCSR882DWM --force  # rebuild even if a valid regions.npz is already there
```

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

python src/bpnet/hitcall/launch_trim_floor.py --dry-run
python src/bpnet/hitcall/launch_trim_floor.py --head profile --head count
python src/bpnet/hitcall/launch_trim_floor.py --min-len 8
```

Pass either the plain `.tsv` or a manually-gzipped `.tsv.gz` — `--cwm-trim-thresholds`/`--cwm-trim-coords`
accept both. `call_hits_bpnet.py` checks existence via `compressed_io.exists()`
and, unlike `peaks.narrowPeak` (read by `finemo extract-regions` via
`polars.scan_csv`, which decompresses gzip transparently), decompresses a
`.gz` mapping file to a temp file first (`compressed_io.ensure_plain()`)
before handing it to Fi-NeMo's own `-T`/`-R` CLI args, since that reader
isn't confirmed to handle gzip itself. `diagnose_background_energy_ratio.py`
does the same before its own direct `load_mapping_tuple` call, and
`report_bpnet.py` does it for its deprecated single-file `-H` mode too --
that dispatches on a literal `.tsv` suffix, so a `.tsv.gz` hits path (e.g.
`hits_dedensified.tsv.gz` after `filter_repeat_density.py`) was silently
misread as a directory instead of failing loudly.

`launch_trim_floor.py` submits one cheap CPU-only job per (experiment, head)
to generate these atlas-wide, skipping any experiment/head whose per-experiment
modisco.h5 is missing or whose floor TSV already exists at that `--min-len`.
Run it (or `compute_trim_floor.py -e` per experiment manually) before using
`--min-trim-len` below.

`launch.py --min-trim-len BP` wires this in per experiment: for each
(experiment, head), it looks up `compute_trim_floor.py -e`'s output for that
`BP` under `modisco/bpnet/` and passes it as that job's `--cwm-trim-coords`,
skipping (and counting separately) any experiment/head whose floor file
hasn't been generated yet.

Some motifs get called far more than a real per-peak binding site count
would suggest — a normal Fi-NeMo run already produces roughly 1-2 orders of
magnitude more hits than TF-MoDISco seqlets, but some motifs go 2-3+ orders
of magnitude beyond that, always concentrated as dozens to 100+ hits of the
*same* motif inside one ~2kb peak: a repeat-artifact signature, not real
biology. Reviewed directly in a K562 (ENCSR220XSM) profile-head run: a clean
`TATAAA` motif and a `TA`-Initiator both showed this, always in degenerate
poly-A/T repeat context, and it's profile-head-specific (the count head
didn't show it). Run `filter_repeat_density.py` before `report_bpnet.py`
below to drop these hits, identity-agnostically (no hardcoded motif
names/consensus — a motif's identity varies by experiment, and JASPAR
doesn't cover core-promoter elements like Inr/TATA anyway): it drops all
hits of a motif within any `--cluster-window` bp span containing
`--min-cluster-hits` or more same-motif hits in the same peak, following
Kelly Cochran's ProCapNet TATA-repeat filter (5+ hits within 80bp):

```bash
python src/bpnet/hitcall/filter_repeat_density.py -e ENCSR882DWM
python src/bpnet/hitcall/filter_repeat_density.py -e ENCSR882DWM --head count
python src/bpnet/hitcall/filter_repeat_density.py -e ENCSR882DWM --min-trim-len 6
python src/bpnet/hitcall/filter_repeat_density.py -e ENCSR882DWM --min-cluster-hits 5 --cluster-window 80
```

Atlas-wide, this runs as part of `launch_post_hoc_pipeline.py` below rather than
its own separate launcher.

Output:

```text
hitcalls/bpnet/{model_dir_name}_{head}/hits_dedensified.tsv
```

Running this before `report_bpnet.py` matters, not just after: `finemo
report`'s own hits-directory mode always reads `hits.tsv` and ignores any
filtering, so `report_bpnet.py` instead passes `hits_dedensified.tsv`
directly as `-H` when it exists (a deprecated-but-functional Fi-NeMo mode)
to recompute `cwm_similarity` against the cleaned-up hits. That lets a motif
whose aggregate `cwm_similarity` is dragged down by repeat noise clear the
QC threshold on its remaining real hits, instead of every hit for that motif
being dropped wholesale by the `cwm_similarity` filter below.

Repeat density isn't the only reason a motif's aggregate `cwm_similarity`
can stay low even after that filter. Reviewed real per-hit `hit_correlation`
distributions for the same K562 (ENCSR220XSM) run: a GATA motif showed a
clear bimodal split — a large bulk of ambiguous hits plus a distinct,
separable population of very high-confidence hits (correlation rising again
past a trough near the top of the range) — while a TA-Initiator motif in the
same run showed no such split at all, just a smooth, monotonically decaying
distribution. A motif can be dragged down by a low-confidence tail even when
a real, legitimate high-confidence subset exists under the same
`motif_name`, but per-hit filtering only helps where that split actually
exists to find — forcing a cutoff on a motif like the TA-Inr above would
just be an arbitrary top-K cut with no data behind it. Run
`filter_low_confidence_hits.py` after `filter_repeat_density.py` (reads its
output if present, else `hits_unique.tsv`) and before `report_bpnet.py`: for
each motif, it builds a histogram of `--score-column` (default
`hit_correlation`), and drops hits below a detected low-confidence mode only
when scipy's prominence-based peak finder confirms one exists (a real second
mode past a real trough, both large enough to trust over histogram noise —
see the module docstring for why prominence, not first/global local
extrema, is the robust choice here). Motifs with a smooth/unimodal
distribution are left untouched entirely:

```bash
python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM
python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --head count
python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --min-trim-len 6
python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --score-column hit_similarity
python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --score-column hit_importance --log-scale --min-bin-frac 0.001
```

Atlas-wide, this also runs as part of `launch_post_hoc_pipeline.py` below.

`--score-column` is deliberately not fixed to `hit_correlation`: that column
is a scale-invariant *shape* match to the motif template, which a
coincidentally motif-shaped stretch of degenerate sequence can satisfy just
as well as a real site. `hit_importance` instead measures actual
attribution magnitude -- how much the model's own output genuinely depends
on that position -- which a real, functionally-used site should show and a
spurious shape-only match usually shouldn't, independent of how well it
correlates. `hit_importance` is heavy right-skewed and effectively
unbounded (unlike the roughly-bounded `hit_correlation`/`hit_similarity`),
so use `--log-scale` with it -- and expect to need a much smaller
`--min-bin-frac` than the default, since a secondary mode spread across more
bins in log-space dilutes per-bin height much faster than a narrow one with
the same total mass (see `--min-bin-frac`'s help text).

Output:

```text
hitcalls/bpnet/{model_dir_name}_{head}/hits_confidence_filtered.tsv
```

`report_bpnet.py` prefers this over `hits_dedensified.tsv` (which it prefers
over raw `hits_unique.tsv`) the same way and for the same reason described
above: passing it directly as `finemo report`'s `-H` argument gets
`cwm_similarity` recomputed against the further-cleaned hits. This
preference is staleness-aware (`call_hits_bpnet.py`'s `resolve_hits_path`):
if you rerun an earlier stage with different settings (e.g.
`filter_repeat_density.py` with a new `--cluster-window`) after a later
stage already ran, the later stage's file is now older than the one it was
built from, so it gets skipped in favor of the rerun's output rather than
silently reporting on out-of-date hits.

Neither of the above fixes every motif: some show no internal bimodal
structure at all by `hit_correlation`/`hit_similarity` (e.g. TATA in the
same K562 run stayed at ~50% implausible peak prevalence through both
filters), so there's nothing for `filter_low_confidence_hits.py` to find.
Two more targeted approaches were tried against exactly this and both came
back null results on real K562 ENCSR220XSM data; both have since been
deleted from the codebase (their small still-useful helpers moved to
`region_utils.py`), but the negative results are worth recording so they
aren't re-attempted:

- **Anchoring a floor to TF-MoDISco discovery seqlets** (`hit_importance`,
  or `hit_correlation * hit_importance`, below the weakest ~1% of a motif's
  own seqlets -- Kelly Cochran's ProCapNet notebook's fixed-constant
  approach made data-driven): dropped 0/414643 hits across all 45 motifs
  on real data, including TATA. TATA's overcalled hits carry real
  `hit_importance` comparable to genuine seqlets -- the model confidently
  attributes importance to the AT-rich sequence, so there's no magnitude
  gap for a magnitude-anchored floor to exploit.
- **Extending each hit to its full untrimmed CWM window and scoring cosine
  similarity against the motif's full CWM** (the per-instance analog of
  `cwm_similarity`, floor again anchored to seqlets): also null, for a
  different reason -- seqlets score systematically *lower* than hits on
  this metric for every motif checked, including healthy ones. MoDISco
  seqlets are an intentionally diverse cluster of variant/degenerate
  instances averaged into one consensus CWM, while Fi-NeMo hit-calling's
  sparse regression explicitly searches for the single best-fitting
  window, so seqlets are the wrong reference population for this score.

The actual distinguishing property, found by direct visual review of real
hit logo plots: real core-promoter hits sit on a genuine local spike in the
attribution track, while spurious same-shape hits sit in generally
noisy/repeat-dense regions with no local prominence at all -- and both
types occur at every distance from the real PRO-cap TSS summit, which is
why every position-based and magnitude-based score above ends up averaging
the signal away. Fi-NeMo's own optimizer has no mechanism that checks this
(sparsity comes only from a global per-motif L1 penalty and a global
correlation floor, confirmed directly from source, never a comparison to a
hit's own local neighborhood).

`--score-column hit_seqlet_confidence` tests local prominence directly via
`tangermeme.seqlet.recursive_seqlets`, an independent seqlet caller whose
"recursive" property requires every internal sub-span to also
independently pass a p-value threshold. This matches the CLIPNET paper's
own published, validated approach for calling these exact motifs ("High
importance profile motifs such as the TATA box and initiator elements were
called using the recursive_seqlets approach" at `threshold=0.05`, not
tangermeme's own stricter default of `0.01`, which calls essentially
nothing on real data due to the compounding recursive requirement).
`hit_seqlet_confidence` is `-log10(p)` of the best call overlapping a hit's
trimmed span, or exactly `0.0` if none overlaps -- a real, meaningful
negative signal, not missing data. Its magnitude carries almost no
information (corroborated hits cluster tightly near the threshold
regardless of position); all the signal is in whether a call exists at
all, so this bypasses `detect_low_confidence_cutoff`'s bimodality search
(built for a dip-then-rise shape, not this score's spike-then-decay shape)
and applies a direct `hit_seqlet_confidence > 0` floor instead.

Applying that floor identity-agnostically to every motif is too blunt: on
real data it touched 28/45 motifs at 0%-89% drop rates, including motifs
with no known contamination problem, and pushed at least one borderline
motif under QC as a side effect. `--seqlet-low-similarity-only` scopes the
floor to only motifs already failing `cwm_similarity` QC (reads
`report/motif_report.tsv` from a prior `report_bpnet.py` run against hits
*before* this filter), leaving everything else untouched:

```bash
python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR220XSM --score-column hit_seqlet_confidence --seqlet-low-similarity-only
python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR220XSM --score-column hit_seqlet_confidence --seqlet-low-similarity-only --seqlet-similarity-threshold 0.85
```

This substantially improves but doesn't fully resolve these motifs past
`cwm_similarity` 0.9 (K562 ENCSR220XSM TATA: 0.765 -> 0.853;
`neg_patterns.pattern_8`: 0.869 -> 0.895). Looser/stricter
`--seqlet-threshold`/`--seqlet-additional-flanks` were also tried and made
things worse or were a no-op -- see `filter_low_confidence_hits.py`'s
module docstring for the full comparison. `report_bpnet.py`'s
`--cwm-similarity-threshold` default below was lowered to retain these
substantially-improved motifs instead of dropping them wholesale at 0.9.

After `call_hits_bpnet.py` (and `filter_repeat_density.py`/
`filter_low_confidence_hits.py`, if used), run `report_bpnet.py` to QC and
filter hits by per-motif CWM similarity, following the same principle as
the [Human
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
`--cwm-similarity-threshold` (default 0.8, not HDMA's 0.9 -- see above:
K562 ENCSR220XSM's TATA box and other core-promoter motifs plateau around
0.85-0.9 even after `hit_seqlet_confidence`, so 0.9 would drop them
wholesale despite the substantial improvement):

```bash
python src/bpnet/hitcall/report_bpnet.py -e ENCSR882DWM
python src/bpnet/hitcall/report_bpnet.py -e ENCSR882DWM --head count
python src/bpnet/hitcall/report_bpnet.py -e ENCSR882DWM --cwm-similarity-threshold 0.9
```

Atlas-wide, this runs (twice -- baseline and final pass) as part of
`launch_post_hoc_pipeline.py` below rather than its own separate launcher:
`finemo report` doesn't use a GPU, so that consolidated job stays CPU-only
(`-C NO_GPU`) rather than needing hit calling's GPU job, and the
`--cwm-similarity-threshold` QC cutoff (via `--report-args`) stays quick to
retune without rerunning hit calling itself.

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

`extract_regions_bpnet.py` (rebuilds `regions.npz` only if missing/corrupt,
without recalling hits) -> `filter_repeat_density.py` -> `report_bpnet.py`
(baseline, needed for `--seqlet-low-similarity-only`'s scoping) ->
`filter_low_confidence_hits.py` -> `report_bpnet.py` (final) is the whole
locked-in post-hoc pipeline, and all five stages are fully self-contained
per experiment (each only ever reads/writes that one experiment's own
files) and individually fast. `launch_post_hoc_pipeline.py` consolidates
them into one SLURM job per experiment that runs all five in sequence, so
there's no need to submit each stage separately and wait for it to finish
across the whole atlas before starting the next one:

```bash
python src/bpnet/hitcall/launch_post_hoc_pipeline.py --dry-run
python src/bpnet/hitcall/launch_post_hoc_pipeline.py --min-trim-len 6
python src/bpnet/hitcall/launch_post_hoc_pipeline.py --low-confidence-args '--score-column hit_seqlet_confidence --seqlet-low-similarity-only --seqlet-similarity-threshold 0.85'
python src/bpnet/hitcall/launch_post_hoc_pipeline.py --min-trim-len 6 --force  # reprocess with CA_INR_COMPENDIUM_ARGS added below
```

For profile head only, `filter_low_confidence_hits.py` also always gets
`--seqlet-compendium-clusters` for CA-Inr's 8 hand-identified MotifCompendium
clusters (`CA_INR_COMPENDIUM_ARGS`, both `pos_patterns.N`/`neg_patterns.N`
since `cluster_final` doesn't stratify by posneg) -- independent of
`--low-confidence-args`, so overriding that flag for an unrelated reason
(e.g. a different `--seqlet-similarity-threshold`) can't silently drop CA-Inr
coverage. This exists because `cwm_similarity` is structurally blind to
CA-Inr's overcalling (~4bp trimmed core scores >0.9 regardless of real
background contamination), so `--seqlet-low-similarity-only`'s QC-failure
scoping never brings it into scope at any threshold, and neither
`lookup_compendium_cluster.py`'s automatic JASPAR-based identity check nor
`diagnose_background_energy_ratio.py`'s identity-agnostic elbow detection
reliably substitutes for it (see `filter_low_confidence_hits.py`'s module
docstring). Count head is untouched: these cluster ids come from
`motifcompendium_profile_pattern_to_cluster.tsv` specifically and would be
meaningless -- or wrongly matched to an unrelated motif -- against count
head's separate clustering. Already-complete experiments need `--force` to
be reprocessed with this added.

Jobs are submitted with `--requeue` (the default `--partition` includes
`owners`, which is preemptible -- `normal`/`akundaje`/`gpu` are not), which
is safe here since there's no per-stage
skip logic inside the job itself -- a requeued job just reruns all five
stages from scratch, and each one overwrites its own output
deterministically, so redoing an already-succeeded stage can't corrupt
anything. `--requeue` doesn't help with a genuine failure though (a real
bug/bad data, OOM, hitting `--time`), so
`check_post_hoc_pipeline_failures.py` reports jobs that started but never
reached a genuinely-complete state, without needing to check SLURM job
states or scan `.err` logs by hand -- naive non-empty-stderr scanning
isn't reliable for this pipeline specifically, since finemo/numpy/
matplotlib routinely print non-fatal warnings to stderr even on success:

```bash
python src/bpnet/hitcall/check_post_hoc_pipeline_failures.py --min-trim-len 6
```

Rerunning `launch_post_hoc_pipeline.py` with the same arguments afterward
resubmits only the flagged experiments, since it uses the same completion
check.

`link_hits_to_compendium.py` below is deliberately excluded from this
consolidated job: unlike the five stages above, it depends on the
atlas-wide MotifCompendium cluster-average h5, built separately by
aggregating motifs across *every* experiment, so it isn't safe to fold
into each experiment's own independent job -- run it as its own later,
atlas-scope step once the compendium is up to date.

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

It prefers the most-processed hits available (same staleness-aware
resolution as `report_bpnet.py` above): `hits_filtered.tsv` (post
`report_bpnet.py` QC) if present and not stale, else
`hits_confidence_filtered.tsv` (post `filter_low_confidence_hits.py`), else
`hits_dedensified.tsv` (post `filter_repeat_density.py`), else raw
`hits_unique.tsv`. It adds a `compendium_motif_name` column
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
`--modisco-h5` pointed at the compendium. No GPU needed, so `launch_link.py`
runs as its own cheap CPU-only SLURM job.

### Consolidated QC

`report_bpnet.py`'s final pass writes one `motif_report.tsv` per experiment
(`motif_name`, `num_hits_total`, `num_hits_restricted`, `cwm_similarity`), but
nothing else in this codebase aggregates that across the atlas -- every other
script here reads one experiment's own report at a time.
`consolidate_motif_reports.py` sweeps every experiment/head the same way
`launch_post_hoc_pipeline.py` does and concatenates them into one table, plus
prints atlas-wide totals (summed hit counts, median `cwm_similarity`, fraction
of motif rows at or below the 0.8 QC threshold):

```bash
python src/bpnet/hitcall/consolidate_motif_reports.py
python src/bpnet/hitcall/consolidate_motif_reports.py --head profile --head count
python src/bpnet/hitcall/consolidate_motif_reports.py --min-trim-len 6
```

Reads the *final* pass's report (after both `filter_repeat_density.py` and
`filter_low_confidence_hits.py`), not the baseline pass `report_bpnet.py` also
writes for `--seqlet-low-similarity-only` scoping -- that one reflects
pre-corroboration-filter hit counts and would overstate what actually
survived. Output: `figures/motif_atlas/hitcall_motif_report_consolidated.tsv`.

### Hit-Call Diagnostics

Read-only investigation scripts, plus one plotting script. None of them filter
hits or write into the pipeline's own output files, so they are safe to run at
any point. Each one's module docstring carries the full reasoning for why it
exists; this is the index.

Most of them were written while root-causing motif-specific overcalling
(TATA/TA-Inr in K562, then CA-Inr in B-cell/neuron/liver), and the lesson that
produced the whole set is worth repeating: **an identity-agnostic filter plus
two clean spot-checks is not validation.** `hit_seqlet_confidence` looked safe
that way and turned out to touch 28/45 motifs at 0–89% drop rates, including
motifs with no contamination problem at all. Run the full per-motif breakdown
before trusting a filter atlas-wide.

Positional evidence — does a motif sit at a fixed offset from the real TSS?

```bash
python src/bpnet/hitcall/diagnose_hit_summit_distance.py -e ENCSR220XSM --min-trim-len 6
python src/bpnet/hitcall/diagnose_hit_summit_distance.py -e ENCSR220XSM --plot-motifs pos_patterns.pattern_2
python src/bpnet/hitcall/plot_motif_spacing_syntax.py -e ENCSR220XSM --motifs pos_patterns.pattern_12 --motifs pos_patterns.pattern_1
```

`diagnose_hit_summit_distance.py` prints a per-motif summary over every motif
and writes individual histograms for a named subset. It joins hits back to the
*real* PRO-cap summit from
`data/processed/peaks/{experiment}_{biosample}_filtered.bed.gz`, because
Fi-NeMo never sees one — `call_hits_bpnet.py` feeds it a synthetic narrowPeak
whose "summit" is just the peak window's midpoint. That file's summit
coordinate convention is undocumented upstream, so the script tests both
interpretations against the peak window and prints which one it inferred.
`plot_motif_spacing_syntax.py` turns the same machinery into one comparative
violin plot (the core-promoter spacing-syntax view: TATA at -25 to -30, Inr at
0); pass an explicit `--motifs` subset for anything paper-facing, since
plotting every motif is unreadable.

Is a suspicious hit set real? These two answer it with evidence that is not
downstream of Fi-NeMo's own thresholding:

```bash
python src/bpnet/hitcall/diagnose_hit_signal_metaplot.py -e ENCSR342WAR --min-trim-len 6 --motif-name pos_patterns.pattern_2
python src/bpnet/hitcall/diagnose_seqlet_confidence_by_group.py -e ENCSR342WAR --min-trim-len 6 --motif-name pos_patterns.pattern_2
```

Both split a motif's hits into "normal" (peaks with exactly one call) and
"excess" (peaks with several), the shape of the overcalling problem. The first
pulls real observed PRO-cap signal around each group (reusing
`metaplot_tss.py`'s own extraction), which is the only ground truth available:
every per-hit statistic Fi-NeMo reports describes agreement with the *model's*
attributions, not whether initiation actually happens there. The second checks
`tangermeme.seqlet.recursive_seqlets` corroboration rates per group, the signal
that originally exposed the TATA defect.

Which motifs need the corroboration floor, atlas-wide, without naming any of
them:

```bash
python src/bpnet/hitcall/diagnose_background_energy_ratio.py -e ENCSR220XSM --min-trim-len 6
python src/bpnet/hitcall/diagnose_background_energy_ratio.py -e ENCSR220XSM --out-tsv tmp/bg_energy_ENCSR220XSM.tsv
python src/bpnet/hitcall/launch_background_energy_ratio.py --head profile --head count --min-trim-len 6
```

Measures how much more attribution energy sits outside a motif's trimmed core
in the hits-averaged CWM than in its MoDISco archetype. This is the scoping
signal behind `filter_low_confidence_hits.py --seqlet-background-excess-only`,
and it exists because `cwm_similarity` is structurally blind to the problem:
CA-Inr's trimmed core is only ~4bp, short enough that almost any hit
containing it scores >0.9 against a near-zero-flank archetype. Reported as a
difference, not a ratio — `background_ratio(modisco_fc)` is often ~0 for a
cleanly discovered motif, and dividing by it blows the number up. The launcher
runs it across every experiment/head so `detect_elbow_count`'s cutoffs can be
reviewed by hand before being trusted atlas-wide.

How hard did a filter actually hit each motif, and what is a motif's shared
identity?

```bash
python src/bpnet/hitcall/diagnose_repeat_density_impact.py -e ENCSR220XSM --min-trim-len 6
python src/bpnet/hitcall/lookup_compendium_cluster.py --pair ENCSR220XSM:pos_patterns.pattern_1 --pair ENCSR342WAR:pos_patterns.pattern_2
```

`diagnose_repeat_density_impact.py` diffs `hits_unique.tsv` against
`hits_dedensified.tsv` per motif — no recomputation, the data is already
there — giving `filter_repeat_density.py` the full per-motif breakdown that
`hit_seqlet_confidence` should have had.
`lookup_compendium_cluster.py` resolves `(experiment, local motif name)` pairs
to their MotifCompendium cluster and reports whether they all agree, which is
the identification step required before scoping a filter with
`--seqlet-compendium-clusters`.

### Housekeeping

`call_hits_bpnet.py` and the post-hoc pipeline leave large redundant files
behind. `cleanup_hitcalls.py` finds and removes them, in increasing order of
judgment: always-safe deletes (`hits.tsv`, whose deduplicated
`hits_unique.tsv` is what every downstream script actually reads; abandoned
`hits_flank_filtered.tsv`/`hits_seqlet_filtered.tsv` stages; orphaned
`regions.tmp.npz`), gzip-in-place (`hits.bed`), and opt-in removal of
call-hits output superseded by a sibling trim-suffixed directory:

```bash
python src/bpnet/hitcall/cleanup_hitcalls.py                              # dry-run report (default)
python src/bpnet/hitcall/cleanup_hitcalls.py --execute
python src/bpnet/hitcall/cleanup_hitcalls.py --execute --include-abandoned-trim-dirs
```

It reports without touching anything unless `--execute` is passed, and skips
anything modified in the last `--min-age-hours` (default 24) so an in-flight
job's outputs are never removed underneath it.

The abandoned-stage deletes are not merely about disk: `HITS_FILE_STAGES` in
`call_hits_bpnet.py` still ranks those filenames *ahead* of the current
outputs for staleness detection, so a leftover file that happens to be newer
than the real current output would make `resolve_hits_path` silently prefer
the stale abandoned one.

## Notes

- Fold `i` is held out for testing and fold `(i + 1) % 7` is used for
  validation.
- Chromosome folds are defined in `configs/chrom_splits.yaml`.
- Attribution launchers skip missing models and existing outputs.
- SLURM launch scripts are operational templates for Sherlock; review resources,
  modules, partitions, paths, and environment setup before porting.
