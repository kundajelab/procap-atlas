# Figure 2 / motif lexicon handoff (2026-09-22)

Status snapshot for picking this back up. Fig 1 is finished; Fig 2 and all
supplementary/ED figures are still placeholders in `procap_atlas.pdf`.

**This file is the runbook.** The "Full run order" section below is the
single place with every command needed to regenerate every Fig 2 panel and
its supporting ED/supplement figures, in dependency order.

## Blocking dependency chain

1. ✅ **Post-hoc hitcall pipeline, profile head** (`launch_post_hoc_pipeline.py
   --min-trim-len 6 --force`) — done. Regenerated hit calls atlas-wide with
   the CA-Inr compendium-cluster scoping fix. **Note: this defaulted to
   `--head profile` only** (`launch_post_hoc_pipeline.py`'s `--head` default
   is `["profile"]` when omitted) — count head was never run through this
   pipeline, so `hitcalls/bpnet/{experiment}_count/regions.npz` etc. don't
   exist yet. Discovered when `--with-metaplots` silently rendered "no
   signal yet" placeholders for every count-head cluster (`plot_figure2.py`'s
   `_logo_metaplot_grid` swallows the `SystemExit` reason from
   `collect_metaplot()`, so this needed a direct `metaplot_motif.py
   --source compendium-seqlets --head count -v` call to surface the real
   error: `regions.npz missing -- run extract_regions_bpnet.py first`).
1b. ✅ Not actually blocking. `regions.npz` was missing for count head
    because rebuilding it needs `attributions/bpnet/{model_dir_name}_count.npz`
    (corrupted, needs regenerating — separate GPU work, not started), but
    count-head Fi-NeMo results (`hits.tsv` etc.) already exist and never
    needed `regions.npz` (genome coords are written directly into
    `hits.tsv`). The seqlet-based metaplot sources (`seqlets`,
    `compendium-seqlets`) don't need it either, on inspection —
    `metaplot_motif.py`'s `region_geometry()` now reads peak coordinates
    directly from `peaks.narrowPeak` via finemo's own `load_peaks()`, at
    half-width = the raw attribution window's half-width. That's the same
    genome-coordinate source `regions.npz` itself is built from; the only
    thing `regions.npz` adds is a cropped copy of the sequence/contribution
    arrays, which this module never read. `peaks.narrowPeak` is written
    before the attribution-dependent step and is protected from
    `cleanup_hitcalls.py`, so it survives even when `regions.npz` doesn't.
    No action needed here — retest cluster 126 and rerun `plot_figure2.py
    --with-metaplots` directly.
2. ✅ **`launch_link.py --head count`** — relabels count-head hits with their
   MotifCompendium cluster identity. Done.
3. ✅ **`motif_hit_density.py --panel2d`** — Fig 2 panel 2d. Built, then
   redesigned once Adam saw the first render and called it uncompelling —
   both halves, not a tweak:
   - **Left (null comparison)**: first version overlaid the full per-cluster
     specificity distributions (observed vs. pooled null draws), which
     diluted the comparison into two similarly-shaped curves since most
     clusters have low specificity either way. Now mirrors
     `plot_figure2.py`'s `panel_concentration()` exactly: a single scalar
     test statistic (atlas-wide mean specificity) against its own null
     distribution (`n_null` permutation draws), bold observed line, add-one
     empirical p-value — the same visual language the rest of Fig 2 uses.
   - **Right (hand-picked TFs)**: first version was a 3-row motif×tissue-
     group heatmap — an awkward chart for 3 rows, and redundant with panel c
     (which already shows what MEF2A/GATA2/POU2F3 look like and their
     signal shape). Replaced with a usage-vs-specificity scatter over every
     qualifying cluster, with the named TFs highlighted/labeled — shows
     *where* they sit in the global landscape instead of repeating panel c.
   - `select_named_clusters()` still resolves `--panel2d-motifs`' JASPAR
     names to compendium clusters. `group_discovery_mask()` (the heatmap's
     masking helper) was removed along with the heatmap — the scatter has
     no missing-cell ambiguity to mask.
   - Not yet run on real count-head data — see runbook step 6.

Nothing else below depends on this chain; everything else can run now.

**Separately (not blocking Fig 2), `attributions/bpnet/*_count.npz` needs
regenerating** — those files are corrupted atlas-wide for count head. Only
matters if `extract_regions_bpnet.py`/`launch_post_hoc_pipeline.py --head
count` needs to rebuild `regions.npz` from scratch for some other reason
later (e.g. a `--region-width` change); the metaplot fallback above avoids
needing that for now.

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
`configs/core_promoter_names.tsv` (cluster 2 = CA-Inr, 4 = TATA, 7 = TA-Inr),
not a generic lineage-restricted/ubiquitous selection — profile head's
compendium is known-contaminated with poly-nucleotide/duplicate/composite
motifs (see Decision section below), so it isn't run through the same
selection algorithm as count head.

```bash
# compendium-wide metaplots for the three named clusters, both strands
for cid in 2 4 7; do
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
Double-check `--n-profile 3` actually restricts the profile band to just the
2/4/7 rows rather than whatever `select_motif_exemplars.py` ranks highest by
seqlet count — not yet verified.

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

### 6. Panel 2d (motif hit density)

```bash
python src/analysis/motif_hit_density.py --head count --panel2d
```
Needs `--cluster-metadata` (or the default
`motifcompendium/bpnet/motifcompendium_count_cluster_metadata.tsv` to
exist) — `--panel2d` uses it to resolve `--panel2d-motifs`' JASPAR names to
compendium clusters for the scatter's highlighted points. Writes the
existing per-cluster specificity table/motif×experiment heatmap as before,
plus `motif_hit_density_count_panel2d.{png,pdf}` (atlas-wide mean
specificity vs. its tissue-permutation null, left; usage-vs-specificity
scatter with MEF2A/GATA2/POU2F3 highlighted, right) and
`..._panel2d_null.npy` (the raw null draws). Not yet run on real count-head
data — check the printed p-value and that the highlighted points land
where the "textbook sites" claim in panel c expects (high specificity,
correct top_group) before treating this as final.

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
- **Motif hit density (panel 2d)** — built, not yet run on real count-head
  data (runbook step 6). Redesigned once after the first render (see
  blocking-chain step 3 above for the full rationale):
  - Left: atlas-wide mean specificity vs. its tissue-label permutation null
    (`specificity_permutation_null()` permutes biosample-group labels
    across experiments, density matrix and each cluster's own discovery
    pattern held fixed; `panel2d_null_stats()` reports the mean against
    that null with an add-one empirical p-value) — a single scalar test
    statistic against its null, matching `plot_figure2.py`'s
    `panel_concentration()` visual convention, not a per-cluster
    distribution overlay.
  - Right: a usage (log mean hits/peak when detected) vs. specificity
    scatter over every qualifying cluster, with `--panel2d-motifs` (default
    MEF2A, GATA2, POU2F3) highlighted and labeled — resolved from JASPAR
    name to compendium cluster by `select_named_clusters()`. Not a
    motif×tissue-group heatmap (that version was redundant with panel c and
    an awkward fit for 3 rows) — no missing-cell masking needed since it's
    a scatter, not a matrix.
  - Both in `plot_panel2d()`, wired into `main()` behind `--panel2d`. Run:
    ```bash
    python src/analysis/motif_hit_density.py --head count --panel2d
    ```

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
