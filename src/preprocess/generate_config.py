#!/usr/bin/env python3
"""Parse ENCODE PRO-cap data manifests and generate a YAML experiment config."""

import re
import warnings
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = REPO_ROOT / "data_manifests"
CONFIG_DIR = REPO_ROOT / "configs"

EXPERIMENT_REPORT = MANIFEST_DIR / "experiment_report_2026_2_4_3h_11m.tsv"
PL_BIGWIGS_MANIFEST = MANIFEST_DIR / "pl_bigwigs.txt"
MN_BIGWIGS_MANIFEST = MANIFEST_DIR / "mn_bigwigs.txt"
PEAKS_MANIFEST = MANIFEST_DIR / "bidirectional_peaks.txt"
DIVERGENT_PEAKS_MANIFEST = MANIFEST_DIR / "divergent_peaks.txt"
UNIDIRECTIONAL_PEAKS_MANIFEST = MANIFEST_DIR / "unidirectional_peaks.txt"
ARCHIVE_BLACKLIST = MANIFEST_DIR / "archive_blacklist.txt"


def parse_blacklist(path: Path) -> set[str]:
    """Parse a blacklist file into a set of ENCFF IDs."""
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(line)
    return ids


def parse_manifest(path: Path) -> dict[str, str]:
    """Parse a manifest URL file into {ENCFF_ID: filename}."""
    lookup = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:  # skip header/query URL
                continue
            line = line.strip()
            if not line:
                continue
            # Extract ENCFF ID and filename from URL like:
            # https://www.encodeproject.org/files/ENCFF568VYK/@@download/ENCFF568VYK.bigWig
            m = re.search(r"/files/(ENCFF\w+)/@@download/(.+)$", line)
            if m:
                lookup[m.group(1)] = m.group(2)
    return lookup


def parse_experiment_report(path: Path) -> list[dict]:
    """Parse the experiment report TSV, returning a list of experiment dicts."""
    experiments = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 2:  # skip timestamp line (0) and header line (1)
                continue
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            # Extract ENCFF IDs from the Files column (col 15, 0-indexed)
            files_str = cols[15] if len(cols) > 15 else ""
            encff_ids = re.findall(r"ENCFF\w+", files_str)

            experiments.append(
                {
                    "accession": cols[1],
                    "biosample_summary": cols[8] if len(cols) > 8 else "",
                    "biosample": cols[9] if len(cols) > 9 else "",
                    "description": cols[11] if len(cols) > 11 else "",
                    "lab": cols[12] if len(cols) > 12 else "",
                    "library_construction": cols[39] if len(cols) > 39 else "",
                    "encff_ids": encff_ids,
                }
            )
    return experiments


def main():
    pl_lookup = parse_manifest(PL_BIGWIGS_MANIFEST)
    mn_lookup = parse_manifest(MN_BIGWIGS_MANIFEST)
    peaks_lookup = parse_manifest(PEAKS_MANIFEST)
    divergent_peaks_lookup = parse_manifest(DIVERGENT_PEAKS_MANIFEST)
    unidirectional_peaks_lookup = parse_manifest(UNIDIRECTIONAL_PEAKS_MANIFEST)
    blacklist = parse_blacklist(ARCHIVE_BLACKLIST)

    if blacklist:
        print(
            f"Loaded {len(blacklist)} blacklisted ENCFF IDs from archive_blacklist.txt"
        )

    experiments = parse_experiment_report(EXPERIMENT_REPORT)

    config = {}
    for exp in experiments:
        entry = {
            "biosample": exp["biosample"],
            "biosample_summary": exp["biosample_summary"],
            "description": exp["description"],
            "lab": exp["lab"],
            "library_construction": exp["library_construction"],
            "peaks": [],
            "unidirectional_peaks": [],
            "pl_bigwigs": [],
            "mn_bigwigs": [],
        }
        divergent_peaks = []
        for encff_id in exp["encff_ids"]:
            if encff_id in blacklist:
                continue
            if encff_id in peaks_lookup:
                entry["peaks"].append(peaks_lookup[encff_id])
            if encff_id in divergent_peaks_lookup:
                divergent_peaks.append(divergent_peaks_lookup[encff_id])
            if encff_id in unidirectional_peaks_lookup:
                entry["unidirectional_peaks"].append(
                    unidirectional_peaks_lookup[encff_id]
                )
            if encff_id in pl_lookup:
                entry["pl_bigwigs"].append(pl_lookup[encff_id])
            if encff_id in mn_lookup:
                entry["mn_bigwigs"].append(mn_lookup[encff_id])

        if not entry["peaks"] and divergent_peaks:
            warnings.warn(
                f"{exp['accession']}: no bidirectional peaks found, "
                f"falling back to divergent peaks: {divergent_peaks}"
            )
            entry["peaks"] = divergent_peaks
            entry["peak_type"] = "divergent"

        # Add processed output paths
        biosample_clean = re.sub(r"[^\w-]", "_", exp["biosample"]).strip("_")
        peak_type = entry.get("peak_type", "bidirectional")
        acc = exp["accession"]
        entry["processed"] = {
            "pl_bigwig": f"data/processed/bigwigs/{acc}_{biosample_clean}_pl.bigWig",
            "mn_bigwig": f"data/processed/bigwigs/{acc}_{biosample_clean}_mn.bigWig",
            "peaks": f"data/processed/peaks/{acc}_{biosample_clean}.bed.gz",
            "gc_negatives": f"data/processed/negatives/{acc}_{biosample_clean}_gc_negatives.bed.gz",
        }

        config[exp["accession"]] = entry

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG_DIR / "experiment_config.yaml"

    with open(output_path, "w") as f:
        f.write("# Auto-generated from ENCODE PRO-cap atlas data manifests\n")
        yaml.dump(
            {"experiments": config},
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    print(f"Wrote {len(config)} experiments to {output_path}")


if __name__ == "__main__":
    main()
