# TSS Metaplots

TSS-centered PRO-cap metaplots and heatmaps from processed signal BigWigs and
GENCODE annotations.

## Status

This is an active analysis utility for visualizing sense and antisense signal
around gene or transcript TSSs.

## Prerequisites

- `configs/experiment_config.yaml`
- `configs/n_reads.txt`
- Processed strand BigWigs from preprocessing
- `data/gencode.v49.annotation.gff3.gz` from `src/download/download_annotations.sh`

## Commands

```bash
python src/metaplot/metaplot_tss.py
python src/metaplot/metaplot_tss.py --window 1000 --bin-size 5
python src/metaplot/metaplot_tss.py --experiment ENCSR882DWM
python src/metaplot/metaplot_tss.py --feature transcript
python src/metaplot/metaplot_tss.py --plot-type heatmap --max-tss 5000
python src/metaplot/metaplot_tss.py --plot-type both --min-reads 10000000
```

## Outputs

```text
figures/metaplots/
```

The script saves butterfly metaplots and/or per-TSS heatmaps depending on
`--plot-type`.

## Notes

- Signal is normalized to RPM with `configs/n_reads.txt`.
- Windows are oriented by gene or transcript strand so downstream points to the
  right.
- Sense signal is shown above the axis and antisense signal below the axis in
  the butterfly plots.
- Only canonical chromosomes `chr1` through `chr22`, `chrX`, and `chrY` are
  included.
