# UCSC Track Hub

UCSC track hub for the ENCODE PRO-cap atlas, covering 224 experiments across 126 cell types and tissues (GRCh38/hg38).

Hub URL: `https://mitra.stanford.edu/kundaje/oak/ayhe/procap-atlas/hub/hub.txt`

To load: in UCSC Genome Browser, go to **My Data → Track Hubs → My Hubs** and paste the URL above.

## Track organization

Experiments are grouped into supertracks by biosample (cell type or tissue). Each experiment has:

- **Signal** (`{exp_id}_signal`) — multiWig container showing plus-strand (red) and minus-strand (blue) PRO-cap signal. Plus-strand values are positive; minus-strand values are stored negative in the BigWig and appear below the axis automatically.
- **Peaks** (`{exp_id}_peaks`) — merged bidirectional and unidirectional TSS peak calls in bigBed format.
- **Attributions** (`{exp_id}_attr_profile`, `{exp_id}_attr_count`) — BPNet profile/count attribution BigWigs displayed as standalone UCSC dynseq tracks using `logo on`.

## Generating the hub files

Hub files are generated from `configs/experiment_config.yaml`. BigWig files are referenced by URL and do not need to be copied into this directory.

```bash
python src/hub/generate_hub.py --email you@example.com
python src/hub/generate_hub.py --email you@example.com --output-dir /other/dir
python src/hub/generate_hub.py --email you@example.com --base-url https://example.com/procap-atlas
python src/hub/generate_hub.py --email you@example.com --track-base-url https://huggingface.co/datasets/adamyhe/procap-atlas-tracks/resolve/main
python src/hub/generate_hub.py --email you@example.com --no-attributions
```

Output: `hub/hub.txt`, `hub/genomes.txt`, `hub/hg38/trackDb.txt`

Before regenerating the hub with attribution tracks, convert BPNet attribution NPZ files to BigWigs:

```bash
python src/bpnet/attribute/launch_bigwig_conversion.py --dry-run
python src/bpnet/attribute/launch_bigwig_conversion.py
```

Converted attribution BigWigs are referenced at `attributions/bpnet/bigwigs/{exp_id}_{head}.bigWig`. UCSC displays these as base-resolution dynseq logos through the `logo on` BigWig setting.

To host BigWig track assets on Hugging Face instead of serving them from Mitra:

```bash
python src/bpnet/benchmark/launch_bigwig_conversion.py --dry-run
python src/bpnet/benchmark/launch_bigwig_conversion.py
python src/hub/upload_tracks_hf.py --dry-run
python src/hub/upload_tracks_hf.py --repo-id adamyhe/procap-atlas-tracks
python src/hub/generate_hub.py --email you@example.com --track-base-url https://huggingface.co/datasets/adamyhe/procap-atlas-tracks/resolve/main
```

HF dataset layout:

- `observed/{exp_id}_{strand}.bigWig`
- `predicted/bpnet/{exp_id}_{strand}.bigWig`
- `attributions/bpnet/{exp_id}_{head}.bigWig`

## Converting peaks to bigBed

UCSC track hubs require bigBed format rather than BED.gz. Convert all experiments after running `process_peaks.py`:

```bash
python src/hub/convert_peaks_bigbed.py            # default: .bb alongside .bed.gz in data/processed/peaks/
python src/hub/convert_peaks_bigbed.py -j 4       # 4 parallel workers
python src/hub/convert_peaks_bigbed.py --output-dir hub/hg38/bigbed  # custom output directory
python src/hub/convert_peaks_bigbed.py --dry-run  # preview without converting
```

Requires `bedToBigBed` from [UCSC Kent tools](https://hgdownload.soe.ucsc.edu/admin/exe/). By default, `.bb` files are written alongside the source `.bed.gz` files in `data/processed/peaks/`. Use `--output-dir` to write them elsewhere (e.g. `hub/hg38/bigbed/`); update `--base-url` in `generate_hub.py` accordingly.

Skips experiments where the output `.bb` file already exists or the input `.bed.gz` is missing.

## Peak format

The 8-column BED format used for peak files (defined in `hub/hg38/bigbed/procap_peak.as`):

| Column | Field | Description |
|--------|-------|-------------|
| 1 | chrom | Chromosome |
| 2 | chromStart | Start position |
| 3 | chromEnd | End position |
| 4 | strand | `+`, `-`, or `Both` (bidirectional peaks) |
| 5 | confidence | Bidirectional confidence label (`.` for unidirectional) |
| 6 | peakType | `Unidirectional` or `Bidirectional` |
| 7 | summitsPos | Plus-strand summit position(s) (`.` if none) |
| 8 | summitsMn | Minus-strand summit position(s) (`.` if none) |

## Validating the hub

```bash
hubCheck hub/hub.txt
```

Requires `hubCheck` from UCSC Kent tools.

## File layout

```
hub/
├── hub.txt                    # Hub descriptor
├── genomes.txt                # Genome assembly reference
├── procap_peak.as             # AutoSQL schema for bigBed tracks
└── hg38/
    └── trackDb.txt            # Track definitions (generated)

data/processed/peaks/
├── {exp}_{biosample}.bed.gz           # Source peak files
└── {exp}_{biosample}_peaks.bb         # Converted bigBed (default output location)
```

BigWig files are served directly from `data/processed/bigwigs/` and are not stored here.
Attribution BigWigs are served directly from `attributions/bpnet/bigwigs/`.
