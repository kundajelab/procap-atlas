# ProCapNet Legacy Benchmarking

Legacy benchmark helper for evaluating historical ProCapNet models on the
PRO-cap atlas data.

## Status

This directory is for legacy model benchmarking only. It is not the active
training or deployment path for the atlas.

## Prerequisites

- Completed preprocessing with `configs/experiment_config.yaml`
- `configs/chrom_splits.yaml`
- Processed strand BigWigs and peaks
- Historical ProCapNet model files under `models/procapnet/{experiment}/`
- `personal_bpnet` and related ProCapNet dependencies available in a separate
  optional/local environment

## Commands

```bash
python src/procapnet/benchmark.py -e ENCSR261KBX
python src/procapnet/benchmark.py -e ENCSR261KBX -v
python src/procapnet/benchmark.py -e ENCSR261KBX --save-output
```

The script expects fold model files at:

```text
models/procapnet/{experiment}/fold_{fold}/{experiment}.procapnet_model.fold{fold}.state_dict.torch
```

## Outputs

```text
performance_metrics/procapnet/{experiment}.json
predictions/procapnet/{experiment}.npz
```

Prediction output is written only when `--save-output` is used.

## Notes

- The commented commands at the top of `benchmark.py` document one historical
  model download flow for `ENCSR261KBX`.
- Use BPNet for current atlas deployment and Cherimoya for the completed but
  not-yet-deployed next model family.
