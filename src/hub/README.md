# UCSC Track Hub

UCSC track hub for the ENCODE PRO-cap atlas, covering 224 experiments across 126 cell types and tissues (GRCh38/hg38).

Hub URL: `https://mitra.stanford.edu/kundaje/oak/ayhe/procap-atlas/hub/hub.txt`

To load: in UCSC Genome Browser, go to **My Data -> Track Hubs -> My Hubs** and paste the URL above.

## Track organization

Experiments are grouped into supertracks by biosample (cell type or tissue). Each supertrack contains the observed PRO-cap signal, peak calls, and optional BPNet contribution score tracks for every experiment from that biosample.

- **Signal** (`{exp_id}_signal`) - multiWig container for observed
  strand-specific PRO-cap signal. The plus strand is red and positive; the minus
  strand is blue and stored as negative values so it appears below the axis.
- **Peaks** (`{exp_id}_peaks`) - merged bidirectional and unidirectional TSS peak
  calls in bigBed format.
- **Contribution scores** (`{exp_id}_attr_profile`, `{exp_id}_attr_count`) -
  BPNet profile/count contribution scores displayed as standalone UCSC dynseq
  logo tracks with `logo on`.

By default, uncapped-library experiments are hidden. For capped experiments, observed signal plus both BPNet profile and count contribution score tracks are shown, peaks use dense visibility, and contribution score tracks appear below the observed signal and peaks.

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

The attribution BigWig launcher is a SLURM helper with Sherlock defaults. Use `--dry-run` first and adjust resources, partitions, modules, and paths before running it on another cluster.

To host track assets on Hugging Face instead of serving them from Mitra:

```bash
python src/bpnet/predict/launch.py --dry-run
python src/bpnet/predict/launch.py
python src/hub/upload_tracks_hf.py --dry-run
python src/hub/upload_tracks_hf.py --repo-id adamyhe/procap-atlas-tracks
python src/hub/upload_tracks_hf.py --repo-id adamyhe/procap-atlas-tracks -j 16
sbatch src/hub/upload_tracks_hf.slurm
N_WORKERS=16 INCLUDE="observed peaks" sbatch src/hub/upload_tracks_hf.slurm
python src/hub/generate_hub.py --email you@example.com --track-base-url https://huggingface.co/datasets/adamyhe/procap-atlas-tracks/resolve/main
```

HF dataset layout:

- `observed/{exp_id}_{strand}.bigWig`
- `peaks/bed/{exp_id}_{biosample}.bed.gz`
- `peaks/bigbed/{exp_id}_{biosample}_peaks.bb`
- `attributions/bpnet/{exp_id}_{head}.bigWig`

Uploads are staged into a temporary repo-shaped directory and sent with
Hugging Face `upload_large_folder`, which is resumable and designed for large
folder uploads.

The SLURM wrapper uses Sherlock defaults and accepts environment-variable
overrides: `REPO_ID`, `REVISION`, `CONFIG`, `N_WORKERS`, `INCLUDE`, `HEADS`,
`VALIDATE_URL=1`, and `DRY_RUN=1`.

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
`hubCheck` is not installed by the root `environment.yml` because it is not
available from the configured conda channels.

## File layout

```
hub/
+-- hub.txt                    # Hub descriptor
+-- genomes.txt                # Genome assembly reference
+-- procap_peak.as             # AutoSQL schema for bigBed tracks
+-- hg38/
    +-- trackDb.txt            # Track definitions (generated)

data/processed/peaks/
+-- {exp}_{biosample}.bed.gz           # Source peak files
+-- {exp}_{biosample}_peaks.bb         # Converted bigBed (default output location)
```

Observed signal BigWigs are served from `data/processed/bigwigs/` by default. Peak bigBeds are served from `data/processed/peaks/` by default. Attribution BigWigs are served from `attributions/bpnet/bigwigs/` by default. When `--track-base-url` is set, observed BigWigs, peak bigBeds, and attribution BigWigs are referenced from the external track asset base URL.
