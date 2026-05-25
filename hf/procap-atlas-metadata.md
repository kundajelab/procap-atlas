---
license: mit
tags:
- genomics
- transcriptomics
- pro-cap
- grch38
- hg38
- transcription-start-sites
- tss
- transcription
- initiation
- promoter
- enhancer
- encode
- metadata
---

# PRO-cap Atlas Metadata

This dataset contains metadata and configuration files for the ENCODE PRO-cap atlas on the human GRCh38/hg38 genome. The files describe 224 ENCODE PRO-cap experiments across 126 biosamples, including experiment metadata, processed file paths, chromosome fold splits, read-count summaries, peak-count summaries, model warning flags, and curated ENCODE file manifests used by the companion [`procap-atlas`](https://github.com/kundajelab/procap-atlas) preprocessing and modeling workflows.

This repository is intended to be used together with:

- BPNet models: https://huggingface.co/adamyhe/procap-atlas
- Track assets and UCSC hub files: https://huggingface.co/datasets/adamyhe/procap-atlas-tracks
- Code repository: https://github.com/kundajelab/procap-atlas

## Dataset Details

- **Curated by:** Adam Y. He and collaborators
- **Source project:** ENCODE PRO-cap atlas
- **Assay:** PRO-cap
- **Organism:** Homo sapiens
- **Genome assembly:** GRCh38/hg38
- **Number of experiments:** 224
- **Number of biosamples:** 126
- **License:** MIT
- **Dataset repo:** https://huggingface.co/datasets/adamyhe/procap-atlas-metadata
- **Code repository:** https://github.com/kundajelab/procap-atlas

## Uses

### Direct Use

Use this dataset to:

- map ENCODE experiment accessions to biosamples, library construction metadata, and processed atlas file paths
- identify plus/minus strand signal BigWigs, processed peak files, filtered peak files, and GC-matched negative-region files expected by the atlas pipeline
- reproduce the chromosome-fold definitions used for BPNet and Cherimoya model training and benchmarking
- filter experiments by read depth, perturbation status, uncapped-library status, and manual warning flags
- generate downstream target tables, model launch jobs, browser hubs, and atlas-level analyses

### Out-of-Scope Use

This dataset does not contain the raw or processed genomics signal tracks themselves. Large BigWig, bigBed, peak, attribution, and browser-track assets are hosted separately in [`adamyhe/procap-atlas-tracks`](https://huggingface.co/datasets/adamyhe/procap-atlas-tracks). The metadata should not be treated as clinical or diagnostic information.

## Dataset Structure

The dataset mirrors metadata files from the companion GitHub repository.

### `configs/`

- `experiment_config.yaml`: primary experiment configuration. For each ENCODE experiment accession, this file records biosample metadata, lab metadata, raw ENCODE file IDs for plus/minus BigWigs and peak files, and processed atlas paths:
  - `processed.pl_bigwig`
  - `processed.mn_bigwig`
  - `processed.peaks`
  - `processed.filtered_peaks`
  - `processed.gc_negatives`
- `chrom_splits.yaml`: seven chromosome folds used for cross-validation. Fold `i` is held out for testing, fold `(i + 1) % 7` is used for validation, and the remaining folds are used for training.
- `n_reads.txt`: per-experiment plus-strand, minus-strand, and total read-count estimates from processed BigWigs.
- `n_peaks.txt`: per-processed-peak-file peak counts.
- `model_warning_flags.tsv` and `model_warning_flags.json`: read-depth, perturbation, uncapped-library, and manual warning flags for all 224 experiments.
- `promoterai_procap_bigwigs.tsv`: PromoterAI/MetaFormer target table derived from processed plus/minus BigWig paths.

### `data_manifests/`

- `experiment_report_2026_2_4_3h_11m.tsv`: ENCODE experiment report used as the metadata source.
- `pl_bigwigs.txt` and `mn_bigwigs.txt`: curated plus- and minus-strand BigWig URL manifests.
- `bidirectional_peaks.txt`, `divergent_peaks.txt`, and `unidirectional_peaks.txt`: curated peak file URL manifests.
- `archive_blacklist.txt`: archived ENCODE file IDs excluded during config generation.
- `expt_with_dups.txt`: experiments with duplicated files or metadata issues tracked during curation.

## Quality Flags

The warning flag files summarize model/data suitability signals:

- `overall_flag`: combined green/yellow/red status
- `read_flag`: read-depth status using default thresholds
- `is_perturbation`: whether metadata matched perturbation/treatment keywords
- `is_uncapped`: whether metadata indicated uncapped-library construction
- `reasons`: structured JSON explanations for each flag

In the current metadata snapshot:

| Summary | Count |
| --- | ---: |
| Experiments | 224 |
| Biosamples | 126 |
| Overall green | 163 |
| Overall yellow | 35 |
| Overall red | 26 |
| Read-depth green | 165 |
| Read-depth yellow | 38 |
| Read-depth red | 21 |
| Perturbation-flagged experiments | 18 |
| Uncapped-library experiments | 4 |

These flags are intended to help users decide which experiments are appropriate for model training, benchmarking, visualization, or downstream analysis. They are not a substitute for inspecting the original ENCODE records and experiment-specific quality metrics.

## Dataset Creation

### Curation Rationale

The metadata dataset provides a stable, compact index for the PRO-cap atlas. It separates reproducibility-critical metadata and manifests from large genomic data assets, making it easier to inspect experiment coverage, reproduce preprocessing, and connect model artifacts to their source ENCODE experiments.

### Source Data

The original data producers are ENCODE consortium laboratories that generated and released PRO-cap experiments. This dataset curates metadata and file manifests from ENCODE experiment reports and file URLs, then augments them with processed-path conventions and atlas-specific quality summaries generated by the `procap-atlas` pipeline. The majority of these datasets are described in the preprint: [Shah et al., 2025](https://www.biorxiv.org/content/10.1101/2025.09.24.676871).

### Processing

The configuration is generated by `src/preprocess/generate_config.py` in the companion repository. The script cross-references the ENCODE experiment report with curated URL manifests, excludes archived file IDs listed in `archive_blacklist.txt`, falls back to divergent peaks when bidirectional peak calls are missing, and records unidirectional peak files separately. Read counts, peak counts, model warning flags, and PromoterAI target tables are generated by downstream scripts in `src/preprocess/`, `src/analysis/`, and `src/metaformer/`.

## Bias, Risks, and Limitations

- ENCODE biosample coverage is uneven across cell types, tissues, developmental contexts, treatments, and disease states.
- Some experiments are perturbation or treatment datasets; users should consult `model_warning_flags.*` and source ENCODE metadata before pooling experiments.
- Uncapped-library experiments are included in the metadata but are flagged because they differ from the capped PRO-cap experiments used for most model and browser-track workflows. These models do not cleanly learn sequence-to-initiation maps, and I would exclude them from atlas-wide analyses or multitask model training.
- File paths in `experiment_config.yaml` are atlas pipeline paths. They identify expected locations in the companion code repository and track dataset, not necessarily local files on a user's machine.
- This metadata does not include raw sequencing reads or full ENCODE quality-control reports.

## How to Use

Download or clone the dataset files from Hugging Face, or use them directly with the companion repository:

```bash
git clone https://github.com/kundajelab/procap-atlas.git
cd procap-atlas

python src/preprocess/generate_config.py
python src/preprocess/count_reads.py
python src/analysis/generate_warning_flags.py
```

Example Python access:

```python
import yaml
import pandas as pd

with open("configs/experiment_config.yaml") as f:
    config = yaml.safe_load(f)

experiments = config["experiments"]
reads = pd.read_csv("configs/n_reads.txt", sep="\t")
flags = pd.read_csv("configs/model_warning_flags.tsv", sep="\t")
```

## Citation

If you use this metadata, please cite the PRO-cap atlas repository and the underlying ENCODE experiments ([Shah et al., 2025](https://www.biorxiv.org/content/10.1101/2025.09.24.676871)). For model-related use, also cite the companion BPNet model repository and relevant software dependencies.

## Contact

For questions, bug reports, or reuse notes, please use the GitHub repository issues: https://github.com/kundajelab/procap-atlas/issues
