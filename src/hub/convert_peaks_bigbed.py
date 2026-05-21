#!/usr/bin/env python3
"""Convert processed PRO-cap peak BED.gz files to bigBed format.

Reads configs/experiment_config.yaml to find peak file paths, then converts
each data/processed/peaks/{exp}_{biosample}.bed.gz to a .bb file in the same
directory by default (alongside the source .bed.gz), or to a custom directory
via --output-dir.

Three peak file formats are handled automatically based on which peak types
were available for each experiment:

  Merged (both bidirectional + unidirectional) — 8 columns:
    chrom  start  end  strand  confidence  peakType  summitsPos  summitsMn

  Bidirectional-only (direct copy from PINTS) — 6 columns:
    chrom  start  end  confidence  summitsPos  summitsMn

  Unidirectional-only (direct copy from PINTS) — column count detected from file.

Requires bedToBigBed from UCSC Kent tools.

Usage:
    python src/hub/convert_peaks_bigbed.py
    python src/hub/convert_peaks_bigbed.py -j 4
    python src/hub/convert_peaks_bigbed.py --output-dir hub/hg38/bigbed
    python src/hub/convert_peaks_bigbed.py --dry-run
"""

import argparse
import gzip
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_config.yaml"
DEFAULT_CHROM_SIZES = REPO_ROOT / "data" / "hg38.chrom.sizes"

# 8-column schema: merged bidirectional + unidirectional peaks
AUTOSQL_MERGED = """\
table procapPeak
"PRO-cap TSS peak (merged bidirectional + unidirectional)"
(
    string chrom;       "Reference sequence chromosome or scaffold"
    uint   chromStart;  "Start position in chromosome"
    uint   chromEnd;    "End position in chromosome"
    string strand;      "Which strand: + / - / Both"
    string confidence;  "Bidirectional confidence label (. for unidirectional)"
    string peakType;    "Unidirectional or Bidirectional"
    string summitsPos;  "Plus-strand summit position(s) (. if none)"
    string summitsMn;   "Minus-strand summit position(s) (. if none)"
)
"""

# 6-column schema: bidirectional/divergent-only peaks (direct copy from PINTS)
AUTOSQL_BIDIR = """\
table procapBidirPeak
"PRO-cap bidirectional TSS peak (PINTS)"
(
    string chrom;       "Reference sequence chromosome or scaffold"
    uint   chromStart;  "Start position in chromosome"
    uint   chromEnd;    "End position in chromosome"
    string confidence;  "Confidence score"
    string summitsPos;  "Plus-strand summit position(s)"
    string summitsMn;   "Minus-strand summit position(s)"
)
"""


def check_dependencies():
    if shutil.which("bedToBigBed") is None:
        print(
            "Error: bedToBigBed not found on PATH.\n"
            "Install from https://hgdownload.soe.ucsc.edu/admin/exe/",
            file=sys.stderr,
        )
        sys.exit(1)


def sanitize_name(s):
    return re.sub(r"[^\w-]", "_", s).strip("_")


def peek_ncols(peaks_path):
    """Return column count from the first non-empty, non-comment line of a bed.gz."""
    with gzip.open(peaks_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return len(line.split("\t"))
    return None


def get_schema_and_type(ncols, peak_source):
    """Return (autosql_text, bed_type_str) for the given column count and peak source.

    peak_source: "merged", "bidirectional", or "unidirectional"
    """
    if ncols == 8 and peak_source == "merged":
        return AUTOSQL_MERGED, "bed3+5"
    if ncols == 6 and peak_source == "bidirectional":
        return AUTOSQL_BIDIR, "bed3+3"
    # Generic fallback: auto-generate schema with unnamed extra fields
    extra = ncols - 3
    fields = "\n".join(
        f'    string field{i + 4};    "Column {i + 4}"' for i in range(extra)
    )
    schema = (
        f'table procapPeakGeneric\n"PRO-cap TSS peak ({peak_source})"\n(\n'
        f'    string chrom;       "Chromosome"\n'
        f'    uint   chromStart;  "Start position"\n'
        f'    uint   chromEnd;    "End position"\n'
        f"{fields}\n)\n"
    )
    return schema, f"bed3+{extra}"


def convert_one(exp_id, peaks_path, output_path, chrom_sizes, autosql_dir, peak_source):
    """Convert one bed.gz to bigBed. Returns (exp_id, success, message)."""
    peaks_path = Path(peaks_path)
    output_path = Path(output_path)

    if output_path.exists():
        return exp_id, True, "skipped (exists)"

    if not peaks_path.exists():
        return exp_id, False, f"missing input: {peaks_path}"

    tmp_path = None
    try:
        ncols = peek_ncols(peaks_path)
        if ncols is None:
            return exp_id, False, "empty or unreadable file"

        autosql_text, bed_type = get_schema_and_type(ncols, peak_source)
        autosql_path = Path(autosql_dir) / f"procap_peak_{peak_source}_{ncols}col.as"
        autosql_path.write_text(autosql_text)

        with tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w") as tmp:
            tmp_path = Path(tmp.name)
            with gzip.open(peaks_path, "rt") as f:
                tmp.write(f.read())

        result = subprocess.run(
            [
                "bedToBigBed",
                f"-type={bed_type}",
                f"-as={autosql_path}",
                str(tmp_path),
                str(chrom_sizes),
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return exp_id, False, result.stderr.strip()
        return exp_id, True, f"converted ({ncols}-col {peak_source})"
    except Exception as e:
        return exp_id, False, str(e)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PRO-cap peak BED.gz files to bigBed"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, metavar="PATH")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="directory for .bb files (default: same directory as each source .bed.gz)",
    )
    parser.add_argument(
        "--chrom-sizes", type=Path, default=DEFAULT_CHROM_SIZES, metavar="PATH"
    )
    parser.add_argument("-j", "--jobs", type=int, default=1, metavar="N")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        check_dependencies()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    experiments = config["experiments"]

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # AutoSQL files are written to hub/ alongside the hub files
    autosql_dir = REPO_ROOT / "hub"
    autosql_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for exp_id, exp in experiments.items():
        processed = exp.get("processed", {})
        peaks_path = processed.get("peaks")
        if not peaks_path:
            continue
        peaks_path = REPO_ROOT / peaks_path
        biosample_clean = sanitize_name(exp.get("biosample", "unknown"))
        bb_filename = f"{exp_id}_{biosample_clean}_peaks.bb"
        output_dir = args.output_dir if args.output_dir is not None else peaks_path.parent
        output_path = output_dir / bb_filename

        has_bi = bool(exp.get("peaks"))
        has_uni = bool(exp.get("unidirectional_peaks"))
        if has_bi and has_uni:
            peak_source = "merged"
        elif has_bi:
            peak_source = "bidirectional"
        else:
            peak_source = "unidirectional"

        tasks.append((exp_id, peaks_path, output_path, peak_source))

    if args.dry_run:
        for exp_id, peaks_path, output_path, peak_source in tasks:
            print(f"  [{peak_source}] {peaks_path.name} -> {output_path}")
        print(f"\n{len(tasks)} experiments total")
        return

    n_done = n_skip = n_fail = 0

    def handle_result(exp_id, ok, msg):
        nonlocal n_done, n_skip, n_fail
        if ok:
            if "skipped" in msg:
                n_skip += 1
            else:
                n_done += 1
                print(f"[OK] {exp_id}")
        else:
            n_fail += 1
            print(f"[FAIL] {exp_id}: {msg}", file=sys.stderr)

    if args.jobs == 1:
        for exp_id, peaks_path, output_path, peak_source in tasks:
            handle_result(
                *convert_one(exp_id, peaks_path, output_path, args.chrom_sizes, autosql_dir, peak_source)
            )
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    convert_one, exp_id, peaks_path, output_path, args.chrom_sizes, autosql_dir, peak_source
                ): exp_id
                for exp_id, peaks_path, output_path, peak_source in tasks
            }
            for future in as_completed(futures):
                handle_result(*future.result())

    print(f"\nDone: {n_done} converted, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
