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

## Motif Atlas Panels

Cross-experiment motif analyses behind the manuscript's motif-lexicon figure.
Both scripts are atlas-scope (they read every experiment at once), unlike the
per-experiment scripts under [`src/bpnet/hitcall/`](../bpnet/hitcall/README.md),
and both share the biosample-to-tissue grouping in `_biosample_groups.py`.

That grouping is keyword-based curation, not computation. Write it out, edit
it, and pass it back so the groups are explicit rather than implicit:

```bash
python src/analysis/motif_hit_density.py --write-group-tsv configs/biosample_groups.tsv
python src/analysis/motif_hit_density.py --biosample-groups configs/biosample_groups.tsv
```

Biosamples matching no rule land in `other` and are always reported to stderr.
Metastatic biosamples are matched before any organ rule, since they are named
for the organ they spread *to* (e.g. "Metastatic Breast Carcinoma in the
Brain" is not a neural sample).

### Motif Lexicon Rarefaction

How the size of the deduplicated MotifCompendium lexicon grows as experiments
are added, and whether that growth is driven by experiment count or by
biosample diversity. Reads only
`motifcompendium/bpnet/motifcompendium_{head}_cluster_metadata.tsv`, whose
`experiments` column is already a cluster x experiment presence matrix — no
attributions or hit calls needed.

```bash
python src/analysis/plot_motif_rarefaction.py
python src/analysis/plot_motif_rarefaction.py --head count
python src/analysis/plot_motif_rarefaction.py --min-cluster-experiments 2
python src/analysis/plot_motif_rarefaction.py --annotation-tsv configs/motif_classes.tsv
```

Outputs:

```text
figures/motif_atlas/motif_rarefaction_{head}.tsv        # long-form curves (k, scheme, motif_class, mean, lo, hi)
figures/motif_atlas/motif_rarefaction_{head}.{png,pdf}
figures/motif_atlas/motif_prevalence_{head}.tsv         # per-cluster prevalence and group breadth
```

Three sampling schemes are compared at each subset size: `uniform` (random
experiments), `diverse` (round-robin across biosample groups), and `redundant`
(one biosample group exhausted before starting the next). `diverse` above
`redundant` is the panel's claim — that tissue diversity, not experiment
count, is what recovers the lexicon.

The `uniform` mean is computed in closed form, not sampled: a cluster present
in `p` of `N` experiments is detected by a random size-`k` subset with
probability `1 - C(N-p, k)/C(N, k)`. That closed form is also why the script
deliberately has **no permutation null**. The obvious one — hold each cluster's
prevalence fixed but randomize which experiments it appears in — is provably
vacuous, since the expectation depends only on `p` and never on which
experiments, so it reproduces the observed `uniform` curve exactly. Only a
structured sampling scheme can see structure here; do not re-add a
uniform-subsampling null.

If `cluster_metadata.tsv` does not exist yet, the script falls back to
`motifcompendium_{head}_pattern_to_cluster.tsv` automatically (or pass
`--pattern-to-cluster` explicitly). `cluster_motifs.py` writes the mapping at
line 361, right after clustering, but the metadata only at line 386 — after
the cluster-average h5 export, JASPAR annotation of the averages,
forward/reverse logo generation, MEME export and per-cluster SVG logo
rendering, which it waits on solely to merge the logo paths in. On a full
atlas run those stages take hours, and this panel needs none of them:
grouping the mapping's `experiment` column by `compendium_motif_name`
recovers exactly the same presence sets (tested against the metadata loader
in `tests/test_motif_atlas_panels.py`). Only `total_seqlets`, `n_motifs` and
`jaspar_name` are lost, so stratified curves collapse to a single class
unless `--annotation-tsv` is supplied.

To skip those stages on future runs, `cluster_motifs.py` takes
`--skip-svg-logos` and `--logo-report-top-n 0`.

Two things to set deliberately:

- `--min-reads` (default 10M, matching `cluster_motifs.py`) holds discovery
  power roughly fixed. Motif discovery scales with library size, so a curve
  over all experiments partly measures read depth rather than biology.
- `--min-cluster-experiments 2` drops single-experiment clusters. Singletons
  are both the least reproducible clusters and, being numerous, the dominant
  contribution to the all-motifs curve's slope — they make the lexicon look
  unsaturated on their own. Prefer this setting for the figure.

#### Abundance thresholds and the sensitivity sweep

The obvious noise filter — drop clusters with few seqlets — is prevalence
confounded. `total_seqlets` is summed over a cluster's contributing motifs and
`n_motifs` tracks prevalence closely (within-model clustering collapses each
experiment to ~one motif per cluster), so `total_seqlets` is largely a
prevalence proxy: thresholding it preferentially deletes tissue-restricted
clusters, which is exactly backwards for this panel.

`--min-seqlets-per-motif` uses the normalized ratio `total_seqlets /
n_motifs` instead, but that is only *partly* decoupled — measured
`r = 0.40` against prevalence on the real count-head compendium. So it
defaults to 0, **and on the real data it should stay there** (see below).
Raising it makes the surviving lexicon more ubiquitous, so a small experiment
sample recovers a larger fraction of it; on real count-head data the k=5
fraction rises monotonically 20.7% → 44.8% from no floor to ≥1000
seqlets/motif.

Note what that rise does *not* establish. Two hypotheses predict it
identically: low-abundance clusters may be real but tissue-restricted (hence
low-prevalence), or they may be spurious (and spurious clusters also appear in
few experiments). The sweep cannot distinguish them, so do not read the rise
as evidence either way.

`--sweep` reports that sensitivity instead of hiding it:

```bash
python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2 --sweep
python src/analysis/plot_motif_rarefaction.py --sweep-thresholds 0 50 100 500 --sweep-mark 5
```

Outputs:

```text
figures/motif_atlas/motif_rarefaction_sweep_{head}.tsv          # long-form (threshold, k, mean, fraction, n_clusters)
figures/motif_atlas/motif_rarefaction_sweep_{head}_summary.tsv  # fraction recovered at each --sweep-marks k
figures/motif_atlas/motif_rarefaction_sweep_{head}.{png,pdf}
```

Both absolute counts and fractions are plotted, because neither is honest
alone: absolute counts keep a fixed meaning across thresholds but each curve
ends at a different total, while fractions read directly as "what a
k-experiment study sees" but have a denominator that moves with the threshold.
Fractions are therefore **not comparable across thresholds**, and no single row
of the summary is "the" answer.

What the sweep does support is the weakest-form claim, which the run prints:
the largest fraction recovered at a given k over every threshold tested is an
upper bound on what a k-experiment study can see. On real count-head data that
is *at most 45% at k=5*, so at least 55% of the lexicon is missed by a
five-experiment study regardless of abundance threshold. Quote that rather
than any single-threshold number.

The sweep uses the closed-form uniform expectation only — it is about
abundance sensitivity, not sampling scheme, and the structured schemes would
add Monte-Carlo noise to a comparison that is exact without it. It requires
`cluster_metadata.tsv`; the pattern-to-cluster fallback carries no seqlet
counts and the sweep refuses rather than silently sweeping nothing.

#### What discriminates real from spurious clusters

Abundance does not; group concentration does — see
[Discovery Concentration](#discovery-concentration) below for the script.
Measured on the real count-head compendium, by `seqlets_per_motif` band:

```text
band       n   median_prevalence  concentration  jaspar_rate  median_jaspar_score
<25       20                 2.0           1.05         0.80                0.840
25-50    105                 3.0           0.96         0.91                0.870
50-100    46                 4.0           0.96         0.91                0.873
100-500   99                 6.0           0.86         0.83                0.862
>=500     73                15.0           0.82         0.96                0.959
```

The useful conclusion here is the second column: low-abundance clusters carry
solid JASPAR matches (median 0.84–0.87; the lower means reflect the ~20% with
no match at all, not weak matches), so they are real motifs that happen to be
under-discovered rather than noise. That is why no abundance floor is applied —
it would delete real biology without removing anything spurious.

**Do not read the `concentration` column above as evidence about tissue
structure.** It is a per-band *median*, and the median is near-useless here: at
prevalence 2–4 the ratio takes only a couple of distinct values (with `p = 2`,
expected is 1.90 and observed can only be 1 or 2, so the ratio is 0.53 or
1.05). Pooling classes together by abundance band dilutes it further. Measured
properly — split by `motif_class`, using pooled concentration and the
single-group test — discovery *is* strongly tissue-concentrated; see
[Discovery Concentration](#discovery-concentration).

#### What none of this addresses

Cluster **redundancy**. One real motif split across two clusters inflates every
count in this section, and it biases in the flattering direction. Bound it
separately with a tomtom self-comparison of
`motifcompendium_{head}_cluster_averages.meme`, or by re-running
`cluster_motifs.py --across-threshold 0.85`.

Also note the abundance ratio is computed over all motifs the compendium
assigned to a cluster, including any from experiments dropped by
`--min-reads`, since `cluster_metadata.tsv` carries only aggregates — it is an
abundance proxy, not an exact count over the retained subset.

#### Interpreting the sampling schemes on real data

Measured on the real count-head compendium (343 clusters, 198 experiments):

```text
k     diverse  redundant  uniform
10      113.2       96.7    109.5
25      190.1      159.2    180.9
50      254.4      220.8    246.7
```

`uniform` sits close to `diverse` because a random draw from 198 experiments
spanning 19 tissue groups is already tissue-diverse. The informative contrast
is `redundant` against the others: a study confined to one tissue recovers
~15–19% fewer motifs at matched experiment count. Experiment count, not tissue
diversity, is the primary driver — state the diversity effect at that size and
do not overclaim it.

The unmatched (non-JASPAR) class is the exception, and by a wide margin: 37
clusters, with diverse 18.0 vs uniform 13.3 vs redundant 10.9 at k=25 (+65%
diverse over redundant). The motifs most requiring tissue diversity to
discover are the ones absent from JASPAR. Before relying on that, inspect
those clusters' logos — `cluster_motifs.py` groups core promoter elements,
repeats and unannotated motifs together in this class, and a tissue-structured
repeat family would produce the same signal artifactually.

This panel measures where motifs are **discovered**, not where they are
**used**. Tissue-specificity claims belong to the hit-density panel below;
weak discovery-level tissue structure does not bound usage-level specificity,
since a motif can be discovered in two arbitrary experiments and still be used
in only one lineage.

#### Collapsing the lexicon by motif identity

`--collapse-by {cluster,jaspar_name,jaspar_family}` changes what counts as one
lexicon unit. Cluster level is the default and is the **upper bound** on
lexicon size, since ~10–18% of clusters are near-duplicates of another (see
[Compendium Redundancy](#compendium-redundancy)). Collapsing by JASPAR identity
removes that by construction — 31 clusters best-matching SP9 become one unit —
with no threshold to defend.

It errs the other way, so read the two as a bracket rather than picking one:
JASPAR annotation is a nearest-neighbour lookup, so genuinely distinct variants
can share a label and be merged when they shouldn't. Identity level is the
**lower bound**.

A unit's experiment set is the union over its member clusters, never the sum or
the max: a motif discovered in different experiments under different cluster
ids was still discovered in all of them, so prevalence can only grow.

Unnamed clusters stay as their own units by default. On the real count head 37%
of clusters carry no JASPAR name, and that unmatched class is where the
strongest tissue concentration sits, so dropping it would discard the most
interesting part of the lexicon. `--drop-unnamed` excludes them if a purely
annotation-based lexicon is wanted.

```bash
python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2
python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2 --collapse-by jaspar_name
python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2 --collapse-by jaspar_family
```

Requires `cluster_metadata.tsv` (the pattern-to-cluster mapping carries no
JASPAR names).

`--annotation-tsv` takes a curated `cluster_final<TAB>class` table for
stratified curves. Without it the script falls back to a JASPAR-match proxy
(matched vs. unmatched), which is only a proxy: JASPAR2026 has essentially no
coverage of core promoter elements, which is why `cluster_motifs.py`'s reports
annotate Inr/TATA and repeats by hand.

### Compendium Redundancy

Measures how much of the lexicon is the same motif counted twice. Every count
derived from the compendium — lexicon size, rarefaction curves, the number of
tissue-restricted clusters — is inflated when `cluster_motifs.py
--across-threshold` fails to merge variants of one motif, and unlike the other
caveats in this directory redundancy biases in the **flattering** direction.

```bash
python src/analysis/motif_redundancy.py --head count
python src/analysis/motif_redundancy.py --head count --report-threshold 1e-6
python src/analysis/motif_redundancy.py --head count --min-overlap-frac 0.8     --cluster-metadata motifcompendium/bpnet/motifcompendium_count_cluster_metadata.tsv
```

Outputs:

```text
figures/motif_atlas/motif_redundancy_{head}_pairs.tsv       # passing pairs, loosest threshold
figures/motif_atlas/motif_redundancy_{head}_summary.tsv     # excess clusters per threshold
figures/motif_atlas/motif_redundancy_{head}_components.tsv  # cluster -> merged component
figures/motif_atlas/motif_redundancy_{head}.{png,pdf}
```

Uses TOMTOM from `memelite` (the "tomtom-lite" reimplementation), which is a
Python API with **no command-line entry point** — hence a script rather than a
shell command. Self-compares `motifcompendium_{head}_cluster_averages.meme`
with the diagonal masked, symmetrizing each pair on the larger of the two
p-values since TOMTOM is asymmetric (the query sets the background scale).

Redundancy is reported as **excess clusters** — how many clusters would
disappear if each near-duplicate group collapsed to one motif. Two properties
of the real data make the computation delicate, and the first run got both
wrong:

**Use the h5, not the MEME export.** MotifCompendium exports fixed-width CWM
windows — all 944 count-head clusters are exactly 50bp — while the informative
core is typically 6–15bp. Untrimmed, TOMTOM largely aligns low-information
flanks, which resemble background and so resemble each other, and
`--min-overlap-frac` goes inert (35bp of a 50-vs-50 comparison is satisfied at
nearly any offset). The untrimmed run reported 77.6% excess at `p ≤ 1e-6`, which
is flank similarity, not redundancy.

Information content on the MEME PFMs does **not** fix this, as the real data
showed: trimming at `0.3 × max(IC)` went from 50bp to a median of 49bp, barely
shrinking anything. Cluster-average PFMs are soft, so the core's IC is modest,
while PRO-cap peaks are GC-rich enough that flanking columns carry real
composition and clear the threshold. Contribution magnitude is the only signal
that marks where a motif is — which is what Fi-NeMo's own `trim_motif` uses:

```bash
python src/analysis/motif_redundancy.py --head count --modisco-h5 auto
```

`--modisco-h5` reads `motifcompendium_{head}_cluster_averages.h5`, takes the
trim span from each cluster's `contrib_scores`, and applies it to the
`sequence` PFM for the TOMTOM comparison (TOMTOM needs probability-like
columns, so the span comes from contributions while the comparison runs on
probabilities). `--min-trim-len 6` mirrors Kelly Cochran's ProCapNet floor. The
MEME route still works and warns when its trimming barely shrank anything;
`--no-trim` reproduces the original behaviour.

**Single linkage chains.** Connected components merge A~B~C even when A and C
are unrelated. On the untrimmed run one component held 142 clusters at
`p ≤ 1e-12` and 919 of 944 at `p ≤ 1e-2` — the sweep cannot fix that. Three
criteria are now reported side by side:

| criterion | behaviour |
|---|---|
| `mutual` | only mutual best hits merge; cannot chain — a lower bound |
| `complete` | complete linkage: a group merges only if *every* pair passes — the usable middle estimate |
| `single` | connected components; chaining-prone upper bound, kept for contrast |

A large `single` − `complete` gap means the threshold is too loose for this
data, not that redundancy is high; the run warns when it exceeds 20% of the
lexicon. `--linkage` picks which criterion's components get written out
(default `complete`).

No single p-value threshold is defensible either — significance scales with
motif length and information content, and family members are genuinely similar
without being duplicates — so the output stays a sweep. Flat across orders of
magnitude means redundancy is well determined; steadily climbing means lexicon
size is threshold-dependent and should be quoted as a range.

**Measure over the clusters a claim rests on, not all 944.** `--subset` takes a
name list or any TSV with a `motif`/`compendium_motif_name` column, so the
prevalence-filtered set (343) or the tissue-restricted set (59) can be tested
directly. That is both the number actually at risk of inflation and far less
prone to chaining, since chaining scales with how many motifs are in play. The
59 restricted clusters already show 6 excess by JASPAR name alone (RELA ×3,
SP2 ×2, ATF1 ×2, POU2F3 ×2, POU2F1::SOX2 ×2), which is the figure to check
against.

Why this needs measuring rather than eyeballing: on the real count-head
compendium 306 clusters carry only **112 distinct JASPAR names** (SP9 claimed
by 31 clusters, TBP by 15, NFYA by 14), which looks like ~57% redundancy. But
JASPAR annotation is a nearest-neighbour lookup, so a bare GC-box and a GC-box
with an ETS half-site can both best-match SP9 while being genuinely distinct.

`--cluster-metadata` therefore reports **JASPAR agreement within merged groups,
per criterion, against its chance baseline** — the only external check
available on whether a merge is real.

Read the enrichment, not the raw percentage. Names are heavily skewed (306
named clusters, 112 names, SP9 alone claiming 31), but the measured chance that
two randomly merged clusters share a name is only **2.6%** (4.1% at family
level), so a ~50% observation is a **~20× enrichment** and is evidence the
merges are real, not the "low agreement" it superficially resembles. Family
agreement additionally absorbs JASPAR's own redundancy: SP1/SP2/SP9 and
ETV4/ETV6/ETV7 are near-identical PWMs, so two clusters can be the same motif
while carrying different best-hit labels.

#### Measured results

Trim threshold matters, and the calibration target below fixes it. Across
`--trim-threshold` 0.3 / 0.5 / 0.7 with `--drop-untrimmable`, at `p ≤ 1e-6`:

```text
trim  median   n      mutual excess   complete excess   name agree (mutual)
0.3    25bp   897   159 (17.7%)      422 (47.0%)       43%
0.5    16bp   941   147 (15.6%)      325 (34.5%)       51%
0.7     7bp   945   123 (13.0%)      224 (23.7%)       52%
```

`mutual` is stable at **13–18%** across the whole range while `complete` halves,
and at 0.7 the chaining warning stops firing entirely (largest complete-linkage
group falls 11 → 6).

The criteria are separated decisively by how agreement behaves as the p-value
threshold loosens (at `--trim-threshold 0.5`):

```text
p       mutual excess   mutual name   complete name   single name
1e-08   101 (10.7%)     51% (26x)     62%             36%
1e-06   147 (15.6%)     51% (26x)     44%             20%
1e-04   171 (18.2%)     47% (24x)     23%              0%
1e-02   172 (18.3%)     46% (23x)     15%              0%
```

`mutual` holds ~46–56% name agreement (57–64% at family level) over seven
orders of magnitude while its excess grows five-fold and then saturates:

```text
p        mutual excess   name agree   family agree
1e-11     36 (3.8%)      56%          61%
1e-09     70 (7.4%)      50%          60%
1e-06    147 (15.6%)     51%          59%
1e-04    171 (18.2%)     47%          58%
1e-02    172 (18.3%)     46%          57%
```

Agreement staying flat while the pair count grows five-fold is the key result:
if loosening the threshold were adding spurious merges, agreement would decay
toward the 2% chance level. It goes 56% → 46%, and the pairs added across that
whole range have ~44% agreement on their own — still 22× chance. So there is no
principled place to stop short, and `mutual` saturates at 172.

`complete` degrades from 62% to 15% agreement and `single` collapses to 0% over
the same range, so both are upper bounds only.

**But `mutual` itself over-counts, because of containment.** Many clusters are
tandem composites — the same core repeated in one 50bp window — and a tandem
mutual-best-matches its own single-core counterpart. That is a hierarchy, not
"the same motif counted twice", and `--min-overlap-frac` cannot catch it since
it measures coverage of the *shorter* motif, which containment satisfies by
construction. Cluster 34 is the clearest case: prevalence 31 across 13 tissue
groups, 11,704 seqlets, no JASPAR name — a tandem SP/KLF, i.e. a double
GC-box, which is unnamed precisely because a 50bp window holding two GC-boxes
does not best-match a single short SP motif.

Splitting the 147 mutual pairs by JASPAR-name status shows the signature
directly:

```text
pair class             n    med min/max len   >=25bp member   weaker member seqlets
>=1 unnamed           78        19 / 27           49/78              27.5
both named, agree     35        10 / 22           14/35              84
both named, disagree  34      13.5 / 15            5/34              80
```

The unnamed-involving pairs are the majority, the widest, and have by far the
weakest second member — and they were **never externally validated**, since the
51% agreement figure could only be computed on the 69 evaluable pairs. The
disagreeing pairs are the narrow, near-symmetric, well-supported ones: the
profile of genuine near-duplicates with ambiguous labels.

Excluding containment:

```text
filter                                 pairs   of 941   of 343
no filter                                147    15.6%    11.7%
exclude any unnamed member                69     7.3%     5.5%
exclude any member >=25bp                 79     8.4%     6.3%
both named AND narrow AND symmetric       42     4.5%     3.3%
```

**Quote 4–8%, not 15–18%.** The higher figures count tandem-vs-core containment
as duplication. For the prevalence≥2 lexicon of 343 the containment-free
estimate is ~3–6%, which is small enough that the lexicon size needs no
material correction — a change from what this README previously recorded.

Downstream, at the containment-free 3–6% the prevalence-filtered lexicon of 343
becomes ~325–335 and the 59 tissue-restricted clusters ~56–57 — a correction
small enough not to affect any claim. The concentration *ratios* in
[Discovery Concentration](#discovery-concentration) are unaffected: duplicates
share an experiment set, so they are equally restricted under both observed and
null. Only the counts move.

The residual uncertainty is the ~41% of mutual pairs that disagree even at
family level — either real cross-family over-merges or JASPAR mislabelling.
That is what the visual review below resolves.

`--trim-threshold 0.5` best matches Fi-NeMo's own per-motif median (16bp vs
14bp); 0.7 matches its hit-weighted median (7bp vs 6bp) but over-trims relative
to the per-motif distribution.

Earlier sweep, before trimming was calibrated (945 clusters at median 25bp):

```text
p_threshold  excess_mutual  excess_complete  excess_single  largest_single
1e-12             0.0%            0.0%           0.0%              1
1e-10            12.0%           20.9%          34.6%            116
1e-08            15.0%           32.4%          59.6%            265
1e-06            17.5%           47.8%          85.5%            737
1e-04            17.6%           65.0%          97.3%            917
1e-02            17.7%           79.3%          99.9%            945
```

`mutual` plateaus at ~17.5% from `1e-6` onward with a largest group of 2, while
`complete` and `single` keep climbing — the signature of the looser criteria
absorbing family members rather than finding duplicates. Note there is no
threshold at which all three agree, so the lexicon size should be quoted with
this range attached rather than as a single corrected number.

#### Experiment universe: 224 / 219 / 198

Three counts appear and all three are correct for different steps:

- **224** — every ENCODE PRO-cap experiment.
- **219** — after dropping the 4 uncapped experiments and the one anomalous
  TSS-positioning experiment (ENCSR973QQI). This is the set the MotifCompendium
  clustering was run over, and what the manuscript methods quote.
- **198** — the subset of those with >10M reads, which every script in this
  directory defaults to via `--min-reads`, to hold motif-discovery power
  roughly fixed.

Confirmable from the data: of the 945 count-head clusters, 76 have no
experiment in the 198-set at all, so they were discovered only in sub-10M
experiments — which could not happen if the compendium had been built from 198.
Lexicon sizes reported here are therefore on the 198-experiment subset, not the
219 the compendium spans. State that restriction explicitly anywhere both
numbers appear, or run with `--min-reads 0` to match.

#### Calibrating the trim threshold against Fi-NeMo

Contribution trimming at `--trim-threshold 0.3` leaves a median of 25bp out of
a 50bp window on the cluster averages. That is about twice as wide as Fi-NeMo
achieves at the same threshold. Measured over a real profile-head run
(ENCSR342WAR, 2.24M hits, 65 motifs, via `start`/`end` vs
`start_untrimmed`/`end_untrimmed` in `hits_unique.tsv`):

```text
Fi-NeMo trimmed widths, 50bp windows, --cwm-trim-threshold 0.3
  median across motifs        14 bp
  median weighted by hits      6 bp   (the high-volume motifs are the narrow ones)
  motifs at the 6bp floor      8 / 65
  motifs never trimmed         7 / 65  (these received 1-140 hits out of 2.2M)
```

Cluster averages are wider at the same threshold because averaging variably
offset instances smears contribution into the flanks. So `0.3` is not
transferable from per-experiment CWMs to cluster averages: raise
`--trim-threshold` until the reported median width approaches ~14bp. The run
prints width quartiles and how many clusters failed to shrink, and warns when
the median exceeds twice Fi-NeMo's.

`--drop-untrimmable` removes clusters whose contributions are diffuse enough
that trimming does not shrink them at all. These have no locatable core, so any
comparison against them is meaningless, and they act as chaining hubs — the
equivalent motifs in a real Fi-NeMo run received 1–140 hits out of 2.2M, so
little is lost by excluding them.

Finally, `--subset` on the 59 tissue-restricted clusters is the cleanest
calibration available, since those show 6 excess by JASPAR name alone.

`--min-overlap-frac` (default 0.7) requires the best alignment to cover that
fraction of the shorter motif, suppressing significant-but-spurious
short-inside-long matches.

#### Reviewing the merges by eye

Statistics can only go so far here; the merges are checkable directly. Every
run writes the **mutual-best-hit pairs** — each cluster's reciprocated closest
match, so each row is a self-contained claim that two clusters are the same
motif, with no chaining involved:

```text
figures/motif_atlas/motif_redundancy_{head}_mutual_pairs.tsv
figures/motif_atlas/motif_redundancy_{head}_mutual_pairs.html
```

The TSV carries both clusters' JASPAR name and score, seqlet counts, trimmed
widths, the p-value, and `name_agree`/`family_agree` flags. The HTML shows the
two forward logos side by side (resolved from
`motifcompendium_{head}_cluster_logo_paths.tsv`), **sorted to put disagreeing
pairs first** — those are where over-merging would be visible, so they are what
a reviewer should spend time on. Pairs are otherwise ordered by seqlet count,
so the ones that most affect the lexicon size come first.

Logos are **embedded as base64 data URIs**, so the HTML is self-contained and
survives being copied off the cluster — which is how these reports get read.
The report is written to `figures/` while the logos live under
`motifcompendium/`, so a linked copy renders only in place. `--link-logos`
references them instead when a smaller file is wanted, and `--top-pairs N`
limits how many rows are shown (priority order preserved). The run prints the
resulting file size.

**Check the `logos:` line the run prints.** It reports how many paths resolved,
how many SVG files were actually found, their total size, and a concrete
example path. Three failure modes previously looked identical in the report —
no path table, paths that resolved to nothing, and paths whose files were
missing — and all three simply rendered as dashes in every row. If the paths
point somewhere unexpected, `--logo-paths` overrides the table and
`--logo-root` overrides the directory its entries are relative to. A report
with no usable logos is still written, now carrying an explicit banner rather
than silently omitting the images.

Embedding uses `<img src="data:...">` rather than inline `<svg>` markup because
matplotlib SVGs carry internal ids referenced through `<defs>`, and inlining
several hundred into one document risks id collisions that silently break
rendering; an `<img>` keeps each logo in its own rendering context. A logo file
that cannot be read is marked in place rather than failing the run.

A disagreeing pair is not automatically a bad merge: JASPAR contains
near-identical motifs, so `family_agree` distinguishes SP1-vs-SP9 (benign) from
SP1-vs-GATA1 (not).

The review set is much smaller than the pair count suggests. On the real count
head at `p ≤ 1e-6`: 147 mutual pairs, but only 69 have both clusters
JASPAR-named, of which 35 agree and **34 disagree** — and family-level
agreement rescues 6 of those, leaving **28 pairs** that disagree even at family
level. Those 28 are the entire question, and they sort to the top of the HTML.

#### Footgun: degenerate p-values

TOMTOM estimates its column background from the target set, and that estimate
can collapse — returning p-values of exactly 0.0 or 1.0 for everything,
including **p = 1.0 for two identical motifs**. It is not simply a matter of
having too few motifs, and it is not monotone in size: measured on the test
fixture after trimming, 41 motifs of 14bp behave correctly (duplicate
p = 7e-9) while 21 of 14bp, 81 of 10bp and 81 of 20bp all collapse entirely.

Nothing in the output reveals this, so the script checks for it and warns when
more than 90% of off-diagonal p-values are exactly 0 or 1, or fewer than 10
distinct values appear. If that fires, every redundancy figure in the run is
meaningless — change `--trim-threshold`, or drop `--subset` so more motifs
contribute to the background. This is also why synthetic one-hot fixtures
cannot test the p-value path; the tests use Dirichlet-drawn PWMs and a
verified-safe motif count and width.

### Annotation Scaffold

JASPAR misses whole categories that matter here — core promoter elements
(Inr, TATA, DPE), repeats, and composite arrangements — so 37 of the 343
reproducible count-head clusters carry no name. Reviewing those by eye showed
the class is largely **tandem composites**: a double CCAAT box (cluster 150), a
double GGAAT (cluster 329), whose cores *are* in JASPAR but not as repeats, so
the nearest-neighbour lookup fails. Those need a human label, which the
manuscript methods already commit to providing.

```bash
python src/analysis/make_annotation_scaffold.py --head count
python src/analysis/make_annotation_scaffold.py --head count --min-prevalence 2
python src/analysis/make_annotation_scaffold.py --head count --all
```

Outputs:

```text
figures/motif_atlas/motif_annotation_{head}_scaffold.tsv    # blank `class` column to fill
figures/motif_atlas/motif_annotation_{head}_scaffold.html   # the same rows with logos embedded
```

Fill the TSV's `class` column while looking at the HTML; both are keyed on
`cluster_final` and sorted by seqlet count so the consequential clusters come
first. The completed TSV is exactly what `plot_motif_rarefaction.py
--annotation-tsv` and `motif_group_concentration.py --annotation-tsv` consume,
so stratified curves and per-class concentration follow with no further work.
Suggested vocabulary (kept short, since stratified curves are only readable
with a handful of classes): `core_promoter`, `tandem_composite`,
`hetero_composite`, `repeat`, `tf_unannotated`, `unclear`.

#### Decision: the non-JASPAR class is not analyzed further (Sep 2026)

The non-JASPAR class was where the strongest apparent tissue restriction sat
(6.4× single-group enrichment, `p = 2e-9`), but on review that rests on 14
clusters, 12 of them at prevalence ≤4, and 7 from `stem_ipsc` — a group with
only 4 experiments. Direct inspection of the logos showed the class is
dominated by **tandem composites** (cluster 150 is a double CCAAT box, 329 a
double GGAAT), whose cores are in JASPAR but not as repeats. 22 of the 37 sit
at prevalence 2.

So the restriction describes recurring promoter architecture discovered
idiosyncratically, not novel lineage-specific TF motifs, and the class is
**excluded from analysis** rather than annotated. Consequences:

- Quote concentration for the TF-matched class only: swap-null concentration
  0.827 at tissue level (45 single-group vs 7.87 expected, `p < 0.001`) and
  0.919 at biosample level. Those are already reported separately, so nothing
  needs recomputing.
- The +65% diverse-over-redundant figure for the unmatched class is withdrawn;
  it rested entirely on these 37 clusters.
- They stay *in* the lexicon count (343, not 306). Excluding them would need
  its own justification and would shift every recovery figure; making no claim
  about them costs nothing.
- They stay in the released compendium, with a methods sentence noting ~11% of
  clusters had no JASPAR match, were predominantly composite or tandem, and
  were retained without further analysis.

**Five exceptions worth a look if the class is ever revisited**: clusters 34
(prevalence 31 across 13 tissue groups, 11,704 seqlets), 57, 68, 95 and 149
(prevalence 6–18). Cluster 34 in particular is broad, deep and unrestricted,
which is the profile of a genuine core promoter element rather than a
tandem-discovery artifact — JASPAR carries no Inr/TATA/DPE entries at all.

The scaffold remains available for the profile head, whose unmatched class will
be larger (63–93 motifs per experiment vs 23–30 for count).

### Discovery Concentration

Tests whether the experiments a cluster was discovered in come from fewer
biosample groups than chance allows. This is the discriminating test that
abundance thresholds cannot provide: a lineage-restricted motif in 8 liver
experiments has `n_groups = 1`, while a cluster spread over 8 arbitrary
experiments sits near the random expectation, and the two can have identical
seqlet counts.

```bash
python src/analysis/motif_group_concentration.py --head count
python src/analysis/motif_group_concentration.py --head count --group-level biosample
python src/analysis/motif_group_concentration.py --head count --min-cluster-experiments 3
python src/analysis/motif_group_concentration.py --head count --jaspar-score-threshold 0.85
```

Outputs:

```text
figures/motif_atlas/motif_concentration_{head}_{level}.tsv          # per-cluster observed/expected/ratio
figures/motif_atlas/motif_concentration_{head}_{level}_summary.tsv  # per-split aggregate + enrichment test
figures/motif_atlas/motif_concentration_{head}_{level}.{png,pdf}    # n_groups vs prevalence against expectation
```

Two exact statistics, for a cluster of prevalence *p* over *N* experiments in
groups of size *n_g*:

```text
E[n_groups | p]   = Σ_g [1 - C(N - n_g, p) / C(N, p)]
P[n_groups = 1|p] = Σ_g C(n_g, p) / C(N, p)
```

`concentration = n_groups / E[n_groups]` is ~1 for a random spread and ≪1 when
concentrated. At low prevalence it has almost no dynamic range, so the
single-group count is the sharper test; it is compared against its **exact
Poisson-binomial** distribution (by DP, not a Poisson or normal approximation —
the per-cluster probabilities are small and very unequal, which is where those
approximations fail). Read `pooled_concentration` and `single_group_p` rather
than `median_concentration`, which is granular enough to jump between adjacent
values.

**Run both `--group-level` values.** They answer different questions, and a
conclusion holding at both is not a grouping artifact:

- `tissue` — the keyword grouping in `_biosample_groups.py`. Asks whether a
  motif is *lineage-restricted*. Coarse: on the real atlas three groups hold
  45% of experiments, and `blood_immune` alone spans erythroid, T/NK, B,
  myeloid and lymphoid-tissue biosamples, so it cannot see restriction
  *within* those groups. Coarse grouping also lowers `E[n_groups]`, which
  inflates the ratio and biases toward "not concentrated".
- `biosample` — the raw ENCODE biosample string (112 groups over the 198
  retained experiments). Asks whether discovery is *replicate-driven*
  (HCT116 ×16, brain metastases ×10, PBMC ×8, K562 ×7). Assumption-free and
  higher resolution, but most biosamples appear once, so `P[n_groups = 1]` is
  near zero for `p > 1` and the single-group test loses power.

Measured on the real count-head compendium, both classes are strongly
tissue-concentrated, and the conclusion holds at both group levels:

```text
tissue level (19 groups)
motif_class   n    pooled_conc  n_single  expected  enrichment  p
TF-matched    306  0.832        45        8.50      5.3x        2.0e-22
unmatched      37  0.767        14        2.19      6.4x        2.0e-09

biosample level (112 groups)
TF-matched    306  0.928         8        1.24      6.5x        3.6e-05
unmatched      37  0.872         1        0.33      3.1x        0.28 (underpowered)
```

The sharpest result comes from comparing the two levels. Biosample groups nest
inside tissue groups, so every biosample-confined cluster is necessarily
tissue-confined: of the 59 clusters confined to one tissue group, only 9 are
confined to a single biosample, so **50 span two or more distinct biosamples
within one tissue**. That is lineage restriction rather than replicate
redundancy from the heavily repeated biosamples (HCT116 ×16, brain metastases
×10, PBMC ×8, K562 ×7).

At biosample level the single-group test is underpowered by construction —
most biosamples appear once, so `P[n_groups = 1]` is tiny for `p > 1` — which
is why `pooled_concentration` is the statistic to read there.

`--min-cluster-experiments 3` checks whether a result rests on clusters sitting
at the reproducibility floor. It does not — the effect strengthens sharply once
prevalence-2 clusters are dropped (239 clusters remain of 343):

```text
tissue level, prevalence >= 3
motif_class   n    pooled_conc  median_conc  n_single  expected  enrichment  p
TF-matched    224  0.829        0.864        16        0.57      28x         2.2e-19
unmatched      15  0.729        0.624         5        0.06      82x         4.7e-10
```

`pooled_concentration` barely moves (0.832 → 0.829 and 0.767 → 0.729) while the
single-group enrichment rises an order of magnitude, because the expectation
collapses much faster than the observed count. At this prevalence
`median_concentration` finally becomes informative too, and it shows the
unmatched class is the more concentrated of the two (0.624 vs 0.864).

Two things to watch when raising the floor. `expected_single_group` can fall
below 1, at which point `single_group_enrichment` is uninterpretable — 2
observed against 0.02 expected reads as "90×" and 0 against 0.004 reads as
"0×", when both mean the test had almost nothing to detect. The summary carries
an `enrichment_reliable` flag and the run warns about affected rows; read
`single_group_p` (still exact) and `pooled_concentration` there instead. And at
biosample level with a raised floor, the single-group test is unpowered by
construction, so only `pooled_concentration` is usable.

#### The read-depth confound, and the null that fixes it

The uniform null treats all retained experiments as exchangeable. **On the real
atlas they are not**: library size differs across biosample groups
(Kruskal–Wallis `p = 1.2e-4`, group medians from 19.5M for
`metastatic_carcinoma` to 41.6M for `reproductive`), and deeper experiments
contribute more discovered motifs for reasons unrelated to lineage. A motif
restricted to a deep group would look tissue-concentrated under a uniform null.
`--min-reads` only sets a floor; it does not equalize depth.

`--swap-permutations` (default 1000) is the fix, and it is on by default. It
randomizes the cluster × experiment presence matrix with the curveball trade
(Strona et al. 2014), preserving **both** margins exactly: each cluster's
prevalence and each experiment's total discovered-motif count. Holding the
latter fixed absorbs read depth, peak count and model quality together, without
having to model any of them — whatever made an experiment productive, it stays
equally productive in the null. The only thing randomized is *which*
experiments a cluster's motifs came from.

Output (`..._swapnull.tsv`) reports observed vs null single-group counts and
mean `n_groups`, with empirical one-sided p-values in add-one form
`(#{null ≥ obs} + 1)/(n + 1)`, so a p-value is never reported as exactly zero.
Read `swap_concentration` (observed mean `n_groups` / null mean): below 1 means
concentrated beyond what per-experiment discovery propensity can explain.

On a matrix with no group structure the swap null lands on the same value as
the closed-form uniform expectation (verified in the tests), so the two nulls
diverge only when the column margins genuinely carry information.

Measured on the real count-head compendium, they barely diverge — the depth
confound is detectable in the depth distribution but has almost no effect on
the estimate:

```text
                       swap_conc  pooled_conc (uniform)  obs_single  null_mean  swap_p
tissue     TF-matched   0.827     0.832                  45          7.87       <0.001
           unmatched    0.762     0.767                  14          2.08       <0.001
biosample  TF-matched   0.919     0.928                   8          0.95       <0.001
           unmatched    0.865     0.872                   1          0.27       <0.001 (mean stat)
```

Concentration estimates move by under 0.01, and the swap null's expected
single-group counts come out slightly *lower* than the analytic ones
(7.87 vs 8.50; 2.08 vs 2.19), so conditioning on per-experiment productivity
makes the enrichment marginally stronger, not weaker (45/7.87 = 5.7×). The
finding is robust to the confound.

Quote `swap_concentration` and `swap_mean_p`, since the uniform null's
exchangeability assumption is known to be violated. Two reporting notes:
permutation p-values floor at `1/(n+1)` — every real cell hits 0.000999 at
n=1000, so report `p < 0.001` and cite the analytic p separately if a
smaller number is wanted. And for the unmatched class at biosample level the
single-group test is underpowered (1 observed, 0.27 expected, p = 0.23); the
mean-`n_groups` statistic is significant there and is the one to use.

**Redundancy caveat.** Duplicate clusters inflate the *counts* (45, 59, 343)
without much biasing the *ratio*: split a real motif into three clusters and
all three carry the same experiment set, so each is equally restricted under
both observed and null. What it does inflate is confidence, since those three
are not independent observations. Treat the counts as clusters rather than
distinct motifs, and see the tomtom cross-check noted above.

The run also prints where restricted clusters land alongside each group's share
of experiments, and `sole_group` in the per-cluster TSV names the lineage, so a
pile-up in the deepest groups is still visible directly.

`--jaspar-score-threshold` matters because the default `motif_class` proxy is
JASPAR *name presence* with no score floor, which admits matches as weak as
~0.82. Raising the floor moves those clusters into the unmatched class and
changes both the class sizes and any per-class result.

Note this measures where motifs are **discovered**. Usage-level tissue
specificity is the hit-density panel below, and weak discovery-level structure
does not bound it — a motif can be discovered in two arbitrary experiments and
still be used in only one lineage.

### Motif Hit Density

Motif x experiment hit-density matrix and tissue-specificity scores, rendered
as a clustered heatmap with biosample-group and read-depth column strips.
Reads each experiment's `hits_linked.tsv` (from
[`link_hits_to_compendium.py`](../bpnet/hitcall/README.md)), whose
`compendium_motif_name` column is the only cross-experiment-comparable motif
identity available — hits are called per experiment against that experiment's
own MoDISco motifs, so the raw `motif_name` means a different motif in every
experiment.

```bash
python src/analysis/motif_hit_density.py
python src/analysis/motif_hit_density.py --head count --min-trim-len 6
python src/analysis/motif_hit_density.py --mask-undiscovered
python src/analysis/motif_hit_density.py --top-n 60 --sort-by specificity
```

Outputs:

```text
figures/motif_atlas/motif_hit_density_{head}.tsv          # cluster x experiment hits/peak, unfiltered
figures/motif_atlas/motif_hit_density_{head}_status.tsv   # per-cell discovered/undiscovered mask
figures/motif_atlas/motif_hit_density_{head}_columns.tsv  # per-experiment peak counts, depth, biosample group
figures/motif_atlas/motif_specificity_{head}.tsv          # per-cluster specificity, breadth, top group
figures/motif_atlas/motif_hit_density_{head}.{png,pdf}
```

`--min-trim-len` must match whatever `hitcall/launch.py` was run with, since it
resolves the same trim-coords-suffixed hits directory. Peak counts come from
`peaks.narrowPeak` (cached at the `{experiment}_{head}/` level, outside any
trim suffix); if it is missing, the count falls back to `max(peak_id) + 1`,
which underestimates whenever trailing peaks got no hits, so the
`peak_count_source` column in `_columns.tsv` records which was used.

Two confounds are reported rather than hidden:

- **Discovery power.** A cluster can only receive hits in an experiment whose
  own MoDISco run discovered a motif assigned to it, so a zero is ambiguous:
  unused, or never discovered at that depth. The `_status.tsv` mask separates
  the two from the compendium's own `experiments` lists, the run prints what
  fraction of cells are structurally zero, and `--mask-undiscovered` leaves
  those cells blank instead of drawing them as true zeros.
- **Read depth.** Hits are normalized per peak, and a depth strip is drawn
  next to the biosample-group strip so depth-driven column structure is
  visible rather than being read as tissue structure.

Specificity is scored over biosample *groups*, not experiments: per-group mean
hits/peak is normalized to a distribution `q` over the groups where the cluster
was detected, and specificity is `1 - H(q)/log(G)` (0 = ubiquitous, 1 = one
group only). Averaging within group first is what keeps heavily replicated
biosamples (HCT116 n=16, Metastatic Breast Carcinoma in the Brain n=10, PBMC
n=8) from dominating the score.

`--sort-by balanced` (the default) splits the drawn rows between the most-used
clusters and the most tissue-restricted ones. Ranking purely by total hits
fills every row with ubiquitous motifs and crowds out the lineage motifs the
panel exists to show; ranking purely by specificity fills it with
low-abundance noise. The tissue-restricted half is gated on
`mean_hits_per_peak_detected` (density among experiments where the motif was
actually called) rather than summed density, which shrinks in direct
proportion to how few tissues a motif is restricted to and so would exclude
exactly those motifs.

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
