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
- motifs
- tfmodisco
- tf-modisco
- motif-discovery
- cwm
- contribution-weight-matrix
- motifcompendium
---

# PRO-cap Atlas Motifs

This dataset contains TF-MoDISco motif discovery results and MotifCompendium atlas-wide motif clustering outputs for BPNet models trained on the ENCODE PRO-cap atlas (GRCh38/hg38). It includes per-experiment MoDISco `.h5` files and HTML report archives for 224 experiments, plus the deduplicated atlas-wide motif compendium that collapses per-experiment motifs into shared clusters across biosamples.

This repository is intended to be used together with:

- BPNet models: https://huggingface.co/adamyhe/procap-atlas-bpnet
- Track assets (signal, peaks, contribution scores): https://huggingface.co/datasets/adamyhe/procap-atlas-tracks
- Atlas metadata: https://huggingface.co/datasets/adamyhe/procap-atlas-metadata
- Code repository: https://github.com/kundajelab/procap-atlas

## Dataset Details

- **Curated by:** Adam Y. He, Claire Tian, Anshul Kundaje
- **Source project:** ENCODE PRO-cap atlas
- **Assay:** PRO-cap
- **Organism:** Homo sapiens
- **Genome assembly:** GRCh38/hg38
- **Number of experiments:** 224
- **Number of biosamples:** 126
- **Attribution heads:** profile, count
- **License:** MIT
- **Dataset repo:** https://huggingface.co/datasets/adamyhe/procap-atlas-motifs
- **Code repository:** https://github.com/kundajelab/procap-atlas

## Uses

### Direct Use

Use this dataset to:

- explore motifs discovered by TF-MoDISco from BPNet attribution scores across ENCODE PRO-cap experiments
- load per-experiment contribution weight matrices (CWMs) and seqlet counts from `.modisco.h5` files
- browse per-experiment HTML motif reports (`.modisco.tar` archives)
- access the atlas-wide MotifCompendium: deduplicated motif clusters with average CWMs, MEME-format motifs, cluster metadata, JASPAR annotations, and SVG logos
- match per-experiment motifs to shared atlas-wide cluster identities via pattern-to-cluster mappings
- use cluster-average CWMs (`cluster_averages.h5`) as a motif set for atlas-wide hit calling with Fi-NeMo

### Out-of-Scope Use

This dataset does not contain trained model checkpoints, raw sequencing data, or processed signal tracks. Models are hosted in [`adamyhe/procap-atlas`](https://huggingface.co/adamyhe/procap-atlas), and track assets are hosted in [`adamyhe/procap-atlas-tracks`](https://huggingface.co/datasets/adamyhe/procap-atlas-tracks). CWMs are attribution-derived motif representations, not position frequency matrices (PFMs); they reflect model-learned sequence contributions and should not be treated as binding affinity measurements. These are research artifacts and should not be used for clinical or diagnostic decision-making.

## Dataset Structure

```text
modisco/
├── {experiment}_{head}.modisco.h5         # TF-MoDISco results (CWMs, seqlets, metaclusters)
└── {experiment}_{head}.modisco.tar        # HTML motif report archive

motifcompendium/
├── motifcompendium_{head}_cluster_averages.h5        # Cluster-average CWMs (modisco-lite h5 format)
├── motifcompendium_{head}_cluster_averages.meme       # MEME-format cluster-average motifs
├── motifcompendium_{head}_pattern_to_cluster.tsv      # Per-experiment motif → atlas cluster mapping
├── motifcompendium_{head}_cluster_metadata.tsv        # Per-cluster stats: n_motifs, total_seqlets, experiments, JASPAR label
├── motifcompendium_{head}_cluster_report.html         # Logo-heavy cluster summary table
├── motifcompendium_{head}_cluster_summary.html        # Lightweight all-clusters table with SVG links
├── motifcompendium_{head}_cluster_logo_paths.tsv      # Cluster → logo SVG path mapping
└── motifcompendium_{head}_cluster_logos/
    ├── fwd/*.svg                                      # Forward-orientation cluster logos
    └── rev/*.svg                                      # Reverse-complement cluster logos
```

`{experiment}` is an ENCODE experiment accession (e.g., `ENCSR882DWM`). `{head}` is `profile` or `count`, corresponding to the BPNet prediction head whose DeepLIFT attributions were used for motif discovery.

### Per-Experiment MoDISco Results

Each `.modisco.h5` file contains the full TF-MoDISco output for one experiment and attribution head, including:

- Discovered motif patterns organized into positive and negative metaclusters
- Contribution weight matrices (CWMs) and hypothetical contribution scores for each pattern
- Seqlet coordinates and per-seqlet attribution snippets
- Pattern similarity and clustering metadata

The `.modisco.tar` archives contain the corresponding HTML motif reports with interactive logos, seqlet counts, and TOMTOM matches. Extract with `tar xf {experiment}_{head}.modisco.tar`.

### Atlas-Wide Motif Compendium

The motif compendium, built with [MotifCompendium](https://github.com/kundajelab/MotifCompendium), collapses per-experiment MoDISco motifs across all atlas experiments into one deduplicated set of atlas-wide clusters. Each cluster represents a motif family observed across one or more biosamples.

Key files:

- **`cluster_averages.h5`**: Cluster-average CWMs in modisco-lite h5 format. Can be used directly as a motif set for atlas-wide hit calling (e.g., with Fi-NeMo via `call_hits_bpnet.py --modisco-h5`).
- **`cluster_averages.meme`**: The same clusters in MEME format for compatibility with MEME Suite tools.
- **`pattern_to_cluster.tsv`**: Maps each `(experiment, local_motif_name)` pair to its atlas-wide `compendium_motif_name`. Used by `link_hits_to_compendium.py` to relabel per-experiment hits with shared cluster identities.
- **`cluster_metadata.tsv`**: Per-cluster summary statistics including number of contributing motifs, total seqlets, number of contributing experiments, and best JASPAR match.
- **`cluster_logos/`**: SVG logos for each cluster in forward and reverse-complement orientations.

## Dataset Creation

### Source Data

MoDISco results are generated from DeepLIFT attribution scores computed on trained BPNet models from the companion model repository. Each experiment's attribution scores are computed over its processed peak regions and GC-matched negative regions across all seven chromosome folds.

### Processing

1. **Attributions**: BPNet DeepLIFT attributions are computed per experiment and prediction head using `src/bpnet/attribute/attribute_bpnet.py`.
2. **MoDISco**: `tfmodisco-lite` discovers motifs from attribution scores using `src/bpnet/modisco/launch.py`. HTML reports are generated with `src/bpnet/modisco/launch_report.py` and tarred for upload.
3. **MotifCompendium**: `src/bpnet/motifcompendium/cluster_motifs.py` loads all per-experiment MoDISco h5 files, builds a MotifCompendium, applies within-experiment and across-experiment similarity clustering, and exports the deduplicated cluster set.

The compendium excludes uncapped-library experiments and experiments below a minimum read-depth threshold. See `src/bpnet/README.md` (Motif Clustering section) for clustering parameters and variant compendium workflows.

## Bias, Risks, and Limitations

- CWMs are model-derived attribution summaries, not direct measurements of TF binding affinity. They reflect what the model learned and are subject to model limitations and training data biases.
- ENCODE biosample coverage is uneven across cell types, tissues, and conditions. Motif recovery depends on sequencing depth, peak set size, and negative training set composition (see Supplementary Note 1 in the companion paper for an example with GATA motifs).
- Compendium cluster identities (`cluster_final` IDs) are specific to the clustering run and its parameters. Different clustering thresholds, experiment subsets, or MotifCompendium versions produce different cluster IDs. The `pattern_to_cluster.tsv` mapping is the authoritative link between per-experiment motifs and atlas clusters.
- The `.modisco.tar` report archives can be large. Extract individual archives as needed rather than unpacking all at once.

## How to Use

### Load a MoDISco h5 file

```python
import h5py

with h5py.File("modisco/ENCSR882DWM_count.modisco.h5", "r") as f:
    # List discovered patterns
    for metacluster in f["pos_patterns"]:
        pattern = f["pos_patterns"][metacluster]
        cwm = pattern["contrib_scores"][:]
        n_seqlets = pattern["seqlets"]["n_seqlets"][()]
        print(f"{metacluster}: {n_seqlets} seqlets, shape {cwm.shape}")
```

### Load the atlas-wide compendium

```python
import pandas as pd

metadata = pd.read_csv(
    "motifcompendium/motifcompendium_count_cluster_metadata.tsv", sep="\t"
)
mapping = pd.read_csv(
    "motifcompendium/motifcompendium_count_pattern_to_cluster.tsv", sep="\t"
)

print(f"{len(metadata)} atlas-wide clusters")
print(metadata[["cluster_final", "n_motifs", "total_seqlets", "jaspar_label"]].head(10))
```

### Use cluster averages for hit calling

```bash
git clone https://github.com/kundajelab/procap-atlas.git
cd procap-atlas

python src/bpnet/hitcall/call_hits_bpnet.py \
    -e ENCSR882DWM \
    --modisco-h5 motifcompendium/bpnet/motifcompendium_profile_cluster_averages.h5
```

See `src/bpnet/README.md` in the companion repository for full hit calling, compendium relabeling, and motif analysis workflows.

## Citation

If you use these motif results, please cite the PRO-cap atlas repository and the underlying ENCODE experiments ([Shah et al., 2025](https://www.biorxiv.org/content/10.1101/2025.09.24.676871)). Please also cite the software dependencies used for motif discovery and clustering:

- [tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite)
- [tangermeme](https://github.com/jmschrei/tangermeme) (Fi-NeMo hit calling)
- [MotifCompendium](https://github.com/kundajelab/MotifCompendium) (atlas-wide clustering)

## Contact

For questions, bug reports, or reuse notes, please use the GitHub repository issues: https://github.com/kundajelab/procap-atlas/issues
