# Figure 2 / motif lexicon handoff

Status snapshot for picking this back up. Fig 1 is finished; Fig 2 and all
supplementary/ED figures are still placeholders in `procap_atlas.pdf`.

**This file is the runbook.** The "Full run order" section below is the
single place with every command needed to regenerate every Fig 2 panel and
its supporting ED/supplement figures, in dependency order.

## Full run order (the runbook)

Commands in the order you'd actually run them. Reference this section
directly rather than re-deriving paths/flags from the discussion below.

Steps 1-5 are also mirrored in `generate_figure2.sh`, a single sbatch
script -- `sbatch src/analysis/generate_figure2.sh` -- for running the
whole thing unattended instead of babysitting an interactive session
through timeouts/dropped connections. Everything mutually independent
(QC summary, the three diagnostic metaplots, the rarefaction/
concentration/exemplars commands, the JASPAR-name pair, redundancy,
cross-cell-type prediction) runs concurrently in the background, each
logged to its own file under `logs/generate_figure2_<job id>/`; only step
4 (needs step 2/3's outputs) and the cross-cell-type figure assembly
(needs its own prediction step) wait on anything. Keep the two in sync if
either changes; the script is not a replacement for this section, just a
batch wrapper around the same commands. Step 0 (atlas-wide housekeeping)
is intentionally not included in the script -- it's a one-off
disk-cleanup pass, not part of "generate the figure," and stays a manual
step.

`assemble_figure2.sh` runs just step 4 as its own sbatch job --
`sbatch src/analysis/assemble_figure2.sh` -- for when steps 1-3/5 are
already done and only the final assembly needs (re)running.

### 0. Atlas-wide housekeeping (stale files from dual min-trim-len runs)

```bash
python src/bpnet/hitcall/cleanup_hitcalls.py                              # dry-run report first
python src/bpnet/hitcall/cleanup_hitcalls.py --execute --include-abandoned-trim-dirs
```
Removes base-level `hits*.tsv`/`report/` left over from runs without
`--min-trim-len` when a `trimcoords-*/` sibling exists. Never touches
`regions.npz`/`peaks.narrowPeak` (trim-independent, still shared/needed).

### 1. Atlas-wide QC summary (per-experiment reports already exist; this aggregates them)

```bash
python src/bpnet/hitcall/consolidate_motif_reports.py --min-trim-len 6
```
Individual per-experiment reports need no regeneration — they're already at
`hitcalls/bpnet/{experiment}_{head}/trimcoords-*/report/motif_report.tsv`
from the pipeline run in step 1 of the blocking chain above.

### 2. Figure 2 main, count head (panels a-c)

```bash
python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2 --sweep

python src/analysis/motif_group_concentration.py --head count --group-level tissue --save-null-draws
python src/analysis/motif_group_concentration.py --head count --group-level biosample --save-null-draws

python src/analysis/select_motif_exemplars.py --head count --max-groups 2 --per-group 3 --with-metaplots \
    --modisco-h5 motifcompendium/bpnet/motifcompendium_count_cluster_averages.h5 \
    --logo-paths motifcompendium/bpnet/motifcompendium_count_cluster_logo_paths.tsv \
    --logo-root motifcompendium/bpnet/
```

### 3. Profile-head core-promoter band — NOT select_motif_exemplars.py

Only presenting the three curated core-promoter motifs in
`configs/core_promoter_names.tsv` (cluster 4 = CA-Inr, 21 = TATA,
24 = TA-Inr), not a generic lineage-restricted/ubiquitous selection —
profile head's compendium is known-contaminated with poly-nucleotide/
duplicate/composite motifs (see Decision section below), so it isn't run
through the same selection algorithm as count head. Note `cluster_final`
ids are not stable across `cluster_motifs.py` reruns (see its own module
comment) — re-verify these ids by eye against
`motifcompendium_profile_cluster_report.html` if the compendium is ever
reclustered.

```bash
# compendium-wide metaplots for the three named clusters. cluster_final ids
# are namespaced separately per pos_patterns/neg_patterns group in the
# compendium h5 -- "cluster 4" only exists on one side, not as a pos/neg
# pair sharing one id -- and 4/21/24 are all confirmed pos_patterns-side
# (neg_patterns/4 etc. don't exist in the h5 at all). Don't loop over both;
# metaplot_motif.py raises SystemExit on the missing side.
for cid in 4 21 24; do
    python src/bpnet/hitcall/metaplot_motif.py --source compendium-seqlets \
        --head profile --compendium-motif-name "pos_patterns.${cid}" -v
done

# still need select_motif_exemplars.py --head profile once, to produce the
# TSV plot_figure2.py's --profile-exemplars reads (not for its own report,
# so the selection quality doesn't matter -- only that clusters 4/21/24
# actually show up in the ubiquitous file's rows). TATA-driven promoters
# are a genuine minority (~10-20% of promoters use a canonical TATA box),
# so cluster 21 doesn't clear the default --broad-groups floor (15) the
# way near-universal CA-Inr does -- --broad-groups 1 --top-ubiquitous 999
# keeps everything instead of re-deriving which motifs "count" as broad.
python src/analysis/select_motif_exemplars.py --head profile --include-unmatched \
    --broad-groups 1 --top-ubiquitous 999
```

### 4. Assemble Figure 2 (count panels + profile core-promoter band, fused with metaplots)

```bash
python src/analysis/plot_figure2.py --head count --modisco-h5 auto --with-metaplots \
    --n-restricted 14 \
    --profile-exemplars figures/motif_atlas/motif_exemplars_profile_ubiquitous.tsv  \
    --profile-h5 motifcompendium/bpnet/motifcompendium_profile_cluster_averages.h5 \
    --profile-names configs/core_promoter_names.tsv --n-profile 3
```

### 5. Extended Data / Supplement

```bash
# JASPAR-name-level robustness pair
python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2 \
    --collapse-by jaspar_name --sweep --out-dir figures/motif_atlas/collapsed
python src/analysis/motif_group_concentration.py --head count --group-level tissue \
    --collapse-by jaspar_name --out-dir figures/motif_atlas/collapsed

# compendium redundancy, count head only -- do not extend to profile head
python src/analysis/motif_redundancy.py --head count --modisco-h5 auto \
    --trim-threshold 0.5 --drop-untrimmable --report-threshold 1e-6

# cross-cell-type prediction
python src/analysis/cross_celltype_prediction.py \
    --observed figures/count_correlation_all198/observed_counts.tsv \
    --predicted figures/count_correlation_all198/predicted_counts.tsv \
    --out-dir figures/cross_celltype_all198
python src/analysis/plot_cross_celltype_figure.py --in-dir figures/cross_celltype_all198
```

## Background: what each panel is and why (commands are in the runbook above)

### Figure 2 main (count head)

Three panels, all already scripted and run on real data via
`plot_figure2.py`:

- **a — motif exemplars**: `select_motif_exemplars.py`. Textbook sites —
  MEF2A (heart+muscle), POU2F3 (lymphoid_b+bulk), GATA2
  (myeloid_erythroid), etc. `--max-groups 2` is the setting to use.
- **b — rarefaction**: `plot_motif_rarefaction.py`. 5-experiment study
  recovers only 20.7% of the lexicon; ≥55% missed at k=5 under every
  abundance threshold, ≥47% even collapsed to JASPAR names.
- **c — discovery concentration**: `motif_group_concentration.py`. 36
  single-group TF-matched clusters vs. 6.15 expected (5.86x enrichment,
  p=5.5e-19), swap-null concentration 0.811.

**Plus a small profile-head core-promoter band** (TATA/Inr/DPE, via
`plot_figure2.py --profile-exemplars`), *not* a parallel rarefaction/
concentration analysis for profile head — see decision below.

## Decision: no quantitative profile-head lexicon numbers

Explicitly decided **not** to report:
- The count-vs-profile concentration contrast as a headline number (was
  5.86x vs 3.47x enrichment) — real, but resting on a profile-head
  compendium with known contamination (below), so not worth the review
  risk for a Brief Communication.
- Profile-head redundancy/lexicon size (5529 raw clusters vs. count's 945;
  mutual-linkage redundancy only 7.2%, comparable to count's 3-6%, so
  redundancy doesn't even explain the raw scale gap).

**Why**: visual inspection of `motifcompendium_profile_cluster_report.html`
shows the profile compendium is full of poly-nucleotide runs, duplicate
motifs, and composite motifs that standard tools (tomtom self-comparison,
JASPAR matching) can't cleanly separate from real motifs. Adam explicitly
declined to hand-annotate the profile compendium the way count head's was
(`cluster_motifs.py`'s "Core promoter motifs, repeats, composite motifs...
annotated manually" — that pass covered count head only) and doesn't care
about exact profile-head motif count accuracy. So: use profile head only
for a handful of hand-picked, visually-verified core-promoter exemplars,
never for a cluster-count or redundancy claim.

## Extended Data / Supplement

- **JASPAR-name-level robustness pair** — Lexicon bracket 343→155,
  concentration re-run at name level (enrichment *rises* 6.15x→7.73x).
  Pre-empts "how many of the 343 are real?"
- **Compendium redundancy (count head only)** — 3-6% containment-free
  near-duplicates. Do not extend to profile head (see decision above).
- **Cross-cell-type prediction** — numbers confirmed final on all 198
  experiments (matched/same-tissue/different-tissue medians reproduce the
  README's documented 21-group benchmark to 4 decimals, confirming the
  extraction genuinely used `--held-out-folds`, not the leaky default).
  Assembly script: `plot_cross_celltype_figure.py` (composites the 5
  panels a/b1/b2/c/d from files `cross_celltype_prediction.py` already
  wrote; never touches the standalone per-panel PDFs, so those stay
  available for manual rearranging).
