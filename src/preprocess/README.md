# Preprocessing

Transforms downloaded ENCODE files into per-experiment inputs for modeling,
benchmarking, metaplots, analyses, and the UCSC track hub.

## Status

These scripts are active production preprocessing steps. They are normally run
after all files in `src/download/` have completed.

## Prerequisites

- Python environment from `environment.yml`
- Downloaded genome, BigWigs, and peak files under `data/`
- `bgzip`
- UCSC Kent tools for BigWig merging: `bigWigMerge`, `bedGraphToBigWig`

## Commands

```bash
# Manifest and metadata parsing
python src/preprocess/generate_config.py

# Merge replicate BigWigs by experiment and strand
python src/preprocess/merge_bigwigs.py
python src/preprocess/merge_bigwigs.py -j 8

# Merge bidirectional and unidirectional peak calls
python src/preprocess/process_peaks.py

# Remove peaks whose model input window contains non-ACGT sequence
python src/preprocess/filter_peaks_run.py
python src/preprocess/filter_peaks_run.py -j 8

# Build an atlas-wide union peak set
python src/preprocess/make_union_peaks.py
python src/preprocess/make_union_peaks.py --min-reads 10000000 --slop 100
python src/preprocess/make_union_peaks.py --dry-run

# Generate GC-matched negatives
python src/preprocess/gc_match_run.py
python src/preprocess/gc_match_run.py -j 8

# Count reads across processed strand BigWigs
python src/preprocess/count_reads.py
python src/preprocess/count_reads.py --tsv configs/n_reads.txt
```

## Outputs

```text
configs/
+-- experiment_config.yaml
+-- n_reads.txt

data/processed/
+-- bigwigs/{experiment}_{biosample}_{strand}.bigWig
+-- peaks/{experiment}_{biosample}.bed.gz
+-- peaks/{experiment}_{biosample}_filtered.bed.gz
+-- peaks/union_peaks.bed.gz
+-- negatives/{experiment}_{biosample}_gc_negatives.bed.gz
```

## Notes

- `generate_config.py` excludes archived ENCODE file IDs listed in
  `data_manifests/archive_blacklist.txt`.
- If bidirectional peaks are missing, `generate_config.py` falls back to
  divergent peaks and emits a warning.
- `process_peaks.py` combines bidirectional and unidirectional peak files when
  both are available, and copies the available peak type when only one exists.
- Minus-strand BigWig values are stored as negative values; read counting uses
  absolute values for totals.
