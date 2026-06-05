# MetaFormer / PromoterAI Helpers

Experimental helpers for converting PRO-cap atlas tracks into PromoterAI /
MetaFormer inputs and launching cluster preprocessing or training jobs.

## Status

This directory is still in development. The scripts are useful templates for
PromoterAI experiments, but they should not be treated as a stable production
pipeline or deployment path.

## Prerequisites

- Completed preprocessing with `configs/experiment_config.yaml`
- Model warning flags from `python src/analysis/generate_warning_flags.py`
- Processed plus/minus BigWigs
- PromoterAI / MetaFormer environment outside this repository
- Sherlock-style cluster resources for the provided SLURM shell templates

## Target TSV

Convert the PRO-cap experiment config into a PromoterAI target TSV:

```bash
python src/metaformer/procap_config_to_promoterai.py
python src/metaformer/procap_config_to_promoterai.py --absolute-paths
python src/metaformer/procap_config_to_promoterai.py --require-files
python src/metaformer/procap_config_to_promoterai.py --warning-flags configs/model_warning_flags.tsv
python src/metaformer/procap_config_to_promoterai.py --include-uncapped --include-perturbed
```

Output:

```text
configs/promoterai_procap_bigwigs.tsv
```

The TSV includes `fwd`, `rev`, `xform`, assay, target, experiment, biosample,
and metadata fields. By default it excludes manually red-flagged, uncapped, and
perturbed datasets according to `configs/model_warning_flags.tsv`.

## Cluster Templates

```bash
sbatch src/metaformer/preprocess.sh
sbatch src/metaformer/train.sh
sbatch src/metaformer/train_bridges2.sh
```

Expected outputs are environment-dependent, but current templates are organized
around:

```text
data/promoterai/
models/metaformer/all_tracks/
```

## Notes

- Treat the SLURM scripts as Sherlock-specific templates. Review paths,
  partitions, conda/module setup, and GPU assumptions before use on any other
  cluster. They activate `${PROCAP_ATLAS_ENV:-procap-atlas}` by default.
- The target TSV helper is the most reusable part of this directory.
- Public-facing atlas documentation should describe MetaFormer support as
  in-development until training, benchmarking, and deployment workflows settle.
