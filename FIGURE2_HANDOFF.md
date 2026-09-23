# Figure 2 / motif lexicon handoff (2026-09-22)

Status snapshot for picking this back up. Fig 1 is finished; Fig 2 and all
supplementary/ED figures are still placeholders in `procap_atlas.pdf`.

## Blocking dependency chain, in order

1. **Post-hoc hitcall pipeline, atlas-wide** (`launch_post_hoc_pipeline.py
   --min-trim-len 6 --force`) — regenerates hit calls with the CA-Inr
   compendium-cluster scoping fix. Currently running.
2. **`launch_link.py --head count`** — relabels count-head hits with their
   MotifCompendium cluster identity. Needs step 1 done first (reads
   `hits_filtered.tsv`/`hits_unique.tsv`).
3. **`motif_hit_density.py`** — now in scope for Fig 2 (previously marked
   optional/blocked in the README; free once step 2 finishes). Never run on
   real data yet, only synthetic fixtures.
4. **Metaplots** — also gated on step 1 finishing.

Nothing else below depends on this chain; everything else can proceed now.

## Figure 2 main (count head)

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
- **Motif hit density (panel 2d)** — see blocking chain above for when real
  data lands. Proposed design once it does:
  - Global distribution of motif-usage specificity, compared against a
    tissue-label permutation null. `motif_hit_density.py` already computes
    a per-cluster `specificity` (`1 - H(q)/log(G)`, entropy-based) via
    `specificity_table()`, but has no permutation-null comparison yet —
    needs adding (`cross_celltype_prediction.py`'s degree-preserving/swap
    null pattern, or `motif_group_concentration.py`'s swap-null, are the
    two existing null implementations to reuse rather than inventing a
    third).
  - A compact motif x tissue-group heatmap (not motif x experiment — the
    existing `plot_heatmap()` clusters by experiment, 198 columns) for a
    small set of representative, high-confidence, hand-picked motifs
    (MEF2A, GATA2, POU2F3 suggested). Unavailable motif-experiment/tissue
    combinations must render as missing (NaN/masked), not zero — the
    existing "undiscovered" handling in `motif_hit_density.py`'s docstring
    already distinguishes this, needs checking it survives group-level
    aggregation.
  - Both pieces need building; only the per-experiment groundwork exists.

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
- Methods/Results: add motif hit density once run on real data.
- Once JASPAR-name-level and redundancy panels are placed, cross-check
  Methods' clustering description (>0.95 within, >0.9 across, Leiden CPM,
  manual annotation for count head) still matches what's cited in each ED
  panel's caption.
