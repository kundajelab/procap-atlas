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
- ucsc-track-hub
- bigwig
- bigbed
---

# PRO-cap Atlas Tracks

This dataset hosts browser-ready track assets for the ENCODE PRO-cap atlas on the human GRCh38/hg38 genome, as well as the processed training/evaluation data for the PRO-cap model atlas. It contains observed strand-specific PRO-cap signal BigWigs, processed TSS peak files, bigBed peak tracks, BPNet profile/count contribution score BigWigs, and UCSC track hub configuration files.

The track hub can be loaded in the UCSC Genome Browser:

```text
https://huggingface.co/datasets/adamyhe/procap-atlas-tracks/resolve/main/ucsc/hub.txt
```

This repository is intended to be used together with:

- BPNet models: https://huggingface.co/adamyhe/procap-atlas
- Atlas metadata: https://huggingface.co/datasets/adamyhe/procap-atlas-metadata
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
- **Dataset repo:** https://huggingface.co/datasets/adamyhe/procap-atlas-tracks
- **Code repository:** https://github.com/kundajelab/procap-atlas
- **Metadata repo:** https://huggingface.co/datasets/adamyhe/procap-atlas-metadata

## Uses

### Direct Use

Use this dataset to:

- visualize ENCODE PRO-cap signal, peaks, and BPNet contribution scores in UCSC Genome Browser
- download processed strand-specific PRO-cap BigWigs for atlas experiments
- download processed peak BED.gz files and converted bigBed peak tracks
- inspect BPNet profile and count contribution score tracks alongside observed signal
- support downstream regulatory genomics analyses that need processed browser-scale track files

### Out-of-Scope Use

This dataset does not contain raw sequencing reads, raw ENCODE analysis files, trained model checkpoints, or full experiment metadata tables. Model checkpoints are hosted in [`adamyhe/procap-atlas`](https://huggingface.co/adamyhe/procap-atlas), and metadata/configuration files are hosted in [`adamyhe/procap-atlas-metadata`](https://huggingface.co/datasets/adamyhe/procap-atlas-metadata). These tracks are research resources and should not be used for clinical or diagnostic decision-making.

## Dataset Structure

```text
observed/
+-- {exp_id}_pl.bigWig
+-- {exp_id}_mn.bigWig

peaks/
+-- bed/{exp_id}_{biosample}.bed.gz
+-- bigbed/{exp_id}_{biosample}_peaks.bb

attributions/
+-- bpnet/{exp_id}_profile.bigWig
+-- bpnet/{exp_id}_count.bigWig

ucsc/
+-- hub.txt
+-- genomes.txt
+-- hg38/trackDb.txt
```

### Observed Signal

`observed/{exp_id}_pl.bigWig` and `observed/{exp_id}_mn.bigWig` contain processed PRO-cap signal for each ENCODE experiment. Plus-strand values are positive. Minus-strand values are stored as negative values so they appear below the axis in genome browsers.

### Peaks

`peaks/bed/` contains processed peak BED.gz files. `peaks/bigbed/` contains UCSC-compatible bigBed conversions of those peaks. Peak files merge bidirectional and unidirectional TSS peak calls when both are available.

The merged 8-column peak format is:

| Column | Field        | Description                                                  |
| ------ | ------------ | ------------------------------------------------------------ |
| 1      | `chrom`      | Chromosome                                                   |
| 2      | `chromStart` | Start position                                               |
| 3      | `chromEnd`   | End position                                                 |
| 4      | `strand`     | `+`, `-`, or `Both`                                          |
| 5      | `confidence` | Bidirectional confidence label; `.` for unidirectional peaks |
| 6      | `peakType`   | `Unidirectional` or `Bidirectional`                          |
| 7      | `summitsPos` | Plus-strand summit position(s), or `.`                       |
| 8      | `summitsMn`  | Minus-strand summit position(s), or `.`                      |

### BPNet Contribution Scores

`attributions/bpnet/{exp_id}_profile.bigWig` and `attributions/bpnet/{exp_id}_count.bigWig` contain BPNet profile and count contribution score tracks. In the UCSC hub these are displayed as standalone dynseq logo tracks with `logo on`.

### UCSC Track Hub

The `ucsc/` directory contains the track hub configuration. Experiments are grouped into supertracks by biosample. For capped experiments, observed signal, peaks, BPNet profile contribution scores, and BPNet count contribution scores are visible by default. Uncapped-library experiments are hidden by default.

## Dataset Creation

### Source Data

The source experiments are ENCODE PRO-cap datasets. Observed signal tracks and peak calls are downloaded from ENCODE using curated manifests in the companion [`procap-atlas`](https://github.com/kundajelab/procap-atlas) repository.

### Processing

The companion repository preprocesses tracks as follows:

- replicate BigWigs are merged per experiment while keeping plus and minus strands separate
- bidirectional and unidirectional peak files are merged into processed peak BED.gz files
- peak BED.gz files are converted to bigBed for UCSC browser compatibility
- BPNet contribution score NPZ files are converted to observed-nucleotide BigWigs and averaged over overlapping windows
- `src/hub/generate_hub.py` writes `ucsc/hub.txt`, `ucsc/genomes.txt`, and `ucsc/hg38/trackDb.txt`

Track assets are uploaded with `src/hub/upload_tracks_hf.py`, which stages a temporary repo-shaped directory and uses Hugging Face `upload_large_folder`.

## Bias, Risks, and Limitations

- ENCODE biosample coverage is uneven across cell types, tissues, disease states, perturbations, and library construction types.
- Some experiments are perturbation, treatment, or uncapped-library datasets. Consult the companion metadata repo before pooling experiments or interpreting tracks as baseline cell-state data.
- The tracks are aligned to GRCh38/hg38 and should not be used directly on other genome assemblies.
- BPNet contribution score tracks reflect trained model behavior and should be interpreted alongside model performance metrics and experiment quality flags.
- Large BigWig and bigBed files require tools that support HTTP byte-range requests for efficient remote access.

## How to Use

Load the hub in UCSC Genome Browser through **My Data -> Track Hubs -> My Hubs**:

```text
https://huggingface.co/datasets/adamyhe/procap-atlas-tracks/resolve/main/ucsc/hub.txt
```

Download individual files directly from Hugging Face resolve URLs, for example:

```text
https://huggingface.co/datasets/adamyhe/procap-atlas-tracks/resolve/main/observed/ENCSR083AMN_pl.bigWig
https://huggingface.co/datasets/adamyhe/procap-atlas-tracks/resolve/main/peaks/bigbed/ENCSR083AMN_B_cell_peaks.bb
https://huggingface.co/datasets/adamyhe/procap-atlas-tracks/resolve/main/attributions/bpnet/ENCSR083AMN_profile.bigWig
```

Programmatic users should pair these tracks with `configs/experiment_config.yaml` and `configs/model_warning_flags.tsv` from [`adamyhe/procap-atlas-metadata`](https://huggingface.co/datasets/adamyhe/procap-atlas-metadata).

## Citation

If you use these tracks, please cite the PRO-cap atlas repository and the underlying ENCODE experiments ([Shah et al., 2025](https://www.biorxiv.org/content/10.1101/2025.09.24.676871)). For model-derived contribution score tracks, also cite the companion BPNet model repository and relevant attribution tooling.

## Contact

For questions, bug reports, or reuse notes, please use the GitHub repository issues: https://github.com/kundajelab/procap-atlas/issues
