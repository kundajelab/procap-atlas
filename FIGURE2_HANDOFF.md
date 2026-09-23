# Figure 2 / motif lexicon handoff (2026-09-22)

Status snapshot for picking this back up. Fig 1 is finished; Fig 2 and all
supplementary/ED figures are still placeholders in `procap_atlas.pdf`.

**This file is the runbook.** The "Full run order" section below is the
single place with every command needed to regenerate every Fig 2 panel and
its supporting ED/supplement figures, in dependency order.

## Status right now (2026-09-23, read this first)

Every prerequisite is done: profile-head and count-head post-hoc hitcall
pipelines, `launch_link.py --head count`, and panel 2d has been cut from
Fig 2 entirely (redundant with panels b/c — see history at the bottom if
you want the reasoning). **Nothing is blocked.** Run the "Full run order"
section below top to bottom.

Five real bugs were found and fixed today in the `--with-metaplots` path
(steps 3-4), #3-5 below being the ones actually causing "metaplots look
flat/random" and "double-peaked":

1. Panel c's layout was broken (rows colliding, illegible captions,
   metaplot stacked below instead of beside its logo) — fixed.
2. The profile row showed the wrong motifs (`--profile-names` was only
   used for captions, not for restricting which clusters got selected) —
   fixed.
3. **`seqlet_positions()` in `metaplot_motif.py` was resolving every
   seqlet-based metaplot (`--source seqlets`/`compendium-seqlets`) to the
   wrong genome position by a fixed ~557bp offset.** modisco-lite's own
   `motifs -w/--window` crops the raw attribution array to 1000bp
   (`modisco.sh`'s hardcoded flag) *before* seqlet discovery, so seqlet
   `start`/`end` are local to that 1000bp crop, not the full 2114bp raw
   window this module assumed. Averaging PRO-cap signal at a position
   shifted the same fixed amount off the true motif center for every
   instance doesn't look like an error — it looks exactly like the "flat/
   random, no positional dependence" signal reported, since there's no
   real alignment at all. Fixed by reading the exact window size back from
   the modisco h5's own `window_size` attribute (`modiscolite.io.save_hdf5`
   writes it) rather than assuming the seqlet and raw windows are the
   same. **This is the actual fix for the flat-metaplot report — 1 and 2
   were real but unrelated bugs found along the way.**

A fourth bug, found chasing the same flat/random-looking CA-Inr metaplot
after fixing #3: `metaplot_motif.py`'s `auto_orient()` was flipping
sense/antisense per contributing experiment based on which one had more
signal, applied independently in `compendium-seqlets` pooling. That's
circular (deciding orientation from the thing you're measuring) and
produced a fake symmetric double-peak instead of a real single peak.
Removed entirely — every source (`hits`/`seqlets`/`compendium-seqlets`) now
trusts each seqlet's/hit's own labeled strand with zero post-hoc
correction, same convention `diagnose_hit_signal_metaplot.py` uses (that
script's own reorientation is a single one-time decision from one large
trusted reference group, not a per-experiment loop, so it isn't affected).
**Confirmed fixed on a real render** — CA-Inr now shows a clean single
peak.

Also, `configs/core_promoter_names.tsv`'s cluster ids were stale (2 of 3
wrong) because `cluster_motifs.py`'s `cluster_final` ids aren't stable
across reruns — corrected to `4 = CA-Inr, 8 = TATA-Inr, 24 = TA-Inr` (was
`2, 4, 7`) by direct visual check against the cluster report HTML.

A fifth bug, found immediately after confirming #4: removing `auto_orient`
fixed CA-Inr, but a fresh full-size standalone render of the same cluster
still showed a symmetric double peak (-13/+13, sense vs. antisense). Root
cause: MotifCompendium's clustering explicitly checks a pattern against
both orientations of a cluster and keeps whichever aligns better
(`similarity_core.compute_similarity_and_align`), so one `cluster_final`
id can legitimately contain some contributing experiments' patterns in one
orientation and others' in the mirror-image orientation --
`cluster_motifs.py`'s `export_pattern_to_cluster_mapping()` never records
which members got flipped, so `compendium-seqlets` pooling mixed two
internally-consistent but globally-mirrored subpopulations. Since
`collect_windows()` flips the whole window for "-"-labeled entries, that
doesn't blur a peak, it splits one real peak into a sense-channel bump at
one offset and an antisense-channel bump at the mirror offset -- exactly
the symptom seen. Fixed by `pattern_is_flipped_relative_to_cluster()`:
aligns each contributing experiment's own CWM against the cluster's
reference CWM (`motifcompendium_{head}_cluster_averages.h5`) and flips
that experiment's seqlet strand labels if the reverse complement aligns
better. Not circular like `auto_orient` -- compares motif shape to the
cluster's own reference shape, never the observed PRO-cap signal being
pooled. **Not yet verified on a real render** -- needs Sherlock's actual
per-experiment modisco h5s and `motifcompendium_{head}_cluster_averages.h5`,
neither available locally. Re-run the CA-Inr standalone test first:
```bash
python src/bpnet/hitcall/metaplot_motif.py --source compendium-seqlets \
    --head profile --compendium-motif-name pos_patterns.4 -v
```

Steps 3-4 below should be re-run once that's confirmed.

## Full run order (the runbook)

Commands in the order you'd actually run them. Reference this section
directly rather than re-deriving paths/flags from the discussion below.

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
`configs/core_promoter_names.tsv` (cluster 4 = CA-Inr, 8 = TATA-Inr,
24 = TA-Inr), not a generic lineage-restricted/ubiquitous selection —
profile head's compendium is known-contaminated with poly-nucleotide/
duplicate/composite motifs (see Decision section below), so it isn't run
through the same selection algorithm as count head. Note `cluster_final`
ids are not stable across `cluster_motifs.py` reruns (see its own module
comment) — re-verify these ids by eye against
`motifcompendium_profile_cluster_report.html` if the compendium is ever
reclustered.

```bash
# compendium-wide metaplots for the three named clusters, both strands
for cid in 4 8 24; do
    for posneg in pos neg; do
        python src/bpnet/hitcall/metaplot_motif.py --source compendium-seqlets \
            --head profile --compendium-motif-name "${posneg}_patterns.${cid}" -v
    done
done

# still need select_motif_exemplars.py --head profile once, to produce the
# TSV plot_figure2.py's --profile-exemplars reads (not for its own report)
python src/analysis/select_motif_exemplars.py --head profile --include-unmatched
```

### 4. Assemble Figure 2 (count panels + profile core-promoter band, fused with metaplots)

```bash
python src/analysis/plot_figure2.py --head count --modisco-h5 auto --with-metaplots \
    --n-restricted 14 \
    --profile-exemplars figures/motif_atlas/motif_exemplars_profile_restricted.tsv \
    --profile-h5 motifcompendium/bpnet/motifcompendium_profile_cluster_averages.h5 \
    --profile-names configs/core_promoter_names.tsv --n-profile 3
```
See "Status right now" at the top for the three bugs fixed in this path
today (layout, profile-row selection, and the seqlet coordinate offset
that actually caused the flat/random metaplots). Not yet re-verified —
this is the command to rerun once you've pulled.

### 5. Extended Data / Supplement (no blockers, runnable now)

```bash
# JASPAR-name-level robustness pair
python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2 \
    --collapse-by jaspar_name --sweep --out-dir figures/motif_atlas/collapsed
python src/analysis/motif_group_concentration.py --head count --group-level tissue \
    --collapse-by jaspar_name --out-dir figures/motif_atlas/collapsed

# compendium redundancy, count head only -- do not extend to profile head
python src/analysis/motif_redundancy.py --head count --modisco-h5 auto \
    --trim-threshold 0.5 --drop-untrimmable --report-threshold 1e-6

# cross-cell-type prediction -- already run once this session; rerun after
# the b2 declutter/visibility fix to get a fresh render
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

- **a — rarefaction**: `plot_motif_rarefaction.py`. 5-experiment study
  recovers only 20.7% of the lexicon; ≥55% missed at k=5 under every
  abundance threshold, ≥47% even collapsed to JASPAR names.
- **b — discovery concentration**: `motif_group_concentration.py`. 36
  single-group TF-matched clusters vs. 6.15 expected (5.86x enrichment,
  p=5.5e-19), swap-null concentration 0.811.
- **c — motif exemplars**: `select_motif_exemplars.py`. Textbook sites —
  MEF2A (heart+muscle), POU2F3 (lymphoid_b+bulk), GATA2
  (myeloid_erythroid), etc. `--max-groups 2` is the setting to use.

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

- **JASPAR-name-level robustness pair** — done. Lexicon bracket 343→155,
  concentration re-run at name level (enrichment *rises* 6.15x→7.73x).
  Pre-empts "how many of the 343 are real?"
- **Compendium redundancy (count head only)** — done, 3-6% containment-free
  near-duplicates. Do not extend to profile head (see decision above).
- **Cross-cell-type prediction** — numbers confirmed final on all 198
  experiments (matched/same-tissue/different-tissue medians reproduce the
  README's documented 21-group benchmark to 4 decimals, confirming the
  extraction genuinely used `--held-out-folds`, not the leaky default).
  Assembly script written: `src/analysis/plot_cross_celltype_figure.py`
  (composites the 5 panels a/b1/b2/c/d from files
  `cross_celltype_prediction.py` already wrote; never touches the
  standalone per-panel PDFs, so those stay available for manual
  rearranging). Just decluttered panel b2 (removed redundant arrow
  annotations, disambiguated the `n=` legend counts) and widened spacing
  everywhere after the first render showed text overlap — **needs a fresh
  visual check** after rerunning:
  ```bash
  python src/analysis/plot_cross_celltype_figure.py --in-dir figures/cross_celltype_all198
  ```
- **Motif hit density** — panel 2d is cut (see blocking-chain note above).
  `motif_hit_density.py`'s motif×experiment clustered heatmap (no
  `--panel2d`) is still an open candidate for an Extended Data figure, not
  yet in this runbook — flag if wanted.

## Manuscript text to-dos

- Results paragraph for Fig 2: state the three main-panel numbers above
  (rarefaction, concentration, exemplars) plus the qualitative two-lexicon
  contrast (profile head = small, largely tissue-invariant core-promoter
  grammar; count head = large, tissue-restricted TF lexicon) — **without**
  a profile-head lexicon-size number.
- One caveat sentence on why no profile-head cluster count/redundancy is
  quoted (contamination by low-complexity/composite discovery, not
  hand-curated).
- Methods: add a paragraph on the cross-cell-type prediction analysis
  (held-out-fold extraction, tier definitions, differential-ceiling logic)
  once that ED figure is finalized — currently undocumented in Methods.
- Once JASPAR-name-level and redundancy panels are placed, cross-check
  Methods' clustering description (>0.95 within, >0.9 across, Leiden CPM,
  manual annotation for count head) still matches what's cited in each ED
  panel's caption.
