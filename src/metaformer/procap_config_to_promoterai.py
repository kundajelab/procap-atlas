#!/usr/bin/env python3
"""Write a PromoterAI BigWig target TSV from the PRO-cap experiment config.

PromoterAI preprocessing expects a TSV with at least these columns:

    fwd  rev  xform

This script uses the processed plus/minus strand BigWig paths in
configs/experiment_config.yaml and uses configs/model_warning_flags.tsv for
default QC exclusions.

Usage:
    python src/metaformer/procap_config_to_promoterai.py
    python src/metaformer/procap_config_to_promoterai.py --absolute-paths
    python src/metaformer/procap_config_to_promoterai.py --require-files
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_config.yaml"
DEFAULT_WARNING_FLAGS = REPO_ROOT / "configs" / "model_warning_flags.tsv"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "promoterai_procap_bigwigs.tsv"
DEFAULT_XFORM = "lambda x: np.arcsinh(np.abs(np.nan_to_num(x)))"
OUTPUT_FIELDS = (
    "fwd",
    "rev",
    "xform",
    "assay",
    "target",
    "experiment",
    "biosample",
    "biosample_summary",
    "library_construction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PRO-cap experiment config to a PromoterAI BigWig TSV."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"input experiment config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output TSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--warning-flags",
        type=Path,
        default=DEFAULT_WARNING_FLAGS,
        help=f"input model warning flag TSV (default: {DEFAULT_WARNING_FLAGS})",
    )
    parser.add_argument(
        "--xform",
        default=DEFAULT_XFORM,
        help=f"PromoterAI target transform expression (default: {DEFAULT_XFORM!r})",
    )
    parser.add_argument(
        "--include-uncapped",
        action="store_true",
        help="include datasets marked uncapped in the warning flag table",
    )
    parser.add_argument(
        "--include-perturbed",
        action="store_true",
        help="include datasets marked as perturbations in the warning flag table",
    )
    parser.add_argument(
        "--include-blacklisted",
        action="store_true",
        help="include manually red-flagged datasets from the warning flag table",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="write absolute BigWig paths instead of repo-relative config paths",
    )
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="skip rows whose processed BigWig files do not exist",
    )
    return parser.parse_args()


def load_experiments(config_path: Path) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    experiments = config.get("experiments") if isinstance(config, dict) else None
    if not isinstance(experiments, dict):
        raise ValueError(f"{config_path} does not contain an experiments mapping")
    return experiments


def load_warning_flags(warning_flags_path: Path) -> dict[str, dict]:
    with open(warning_flags_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"experiment", "is_perturbation", "is_uncapped", "flags"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{warning_flags_path} is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )

        warning_flags = {}
        for row in reader:
            exp_id = row["experiment"]
            if exp_id in warning_flags:
                raise ValueError(
                    f"{warning_flags_path} contains duplicate experiment: {exp_id}"
                )
            warning_flags[exp_id] = {
                "is_perturbation": row["is_perturbation"].strip().lower() == "true",
                "is_uncapped": row["is_uncapped"].strip().lower() == "true",
                "flags": {
                    flag
                    for flag in row.get("flags", "").split(";")
                    if flag
                },
            }

    return warning_flags


def normalize_target(exp_id: str, biosample: str) -> str:
    label = f"{exp_id}-{biosample}"
    return re.sub(r"\s+", "_", label.strip())


def output_path_for_bigwig(path: str, absolute: bool) -> str:
    if not absolute:
        return path
    return str((REPO_ROOT / path).resolve())


def build_rows(
    experiments: dict,
    warning_flags: dict[str, dict],
    xform: str,
    include_uncapped: bool,
    include_perturbed: bool,
    include_blacklisted: bool,
    absolute_paths: bool,
    require_files: bool,
) -> tuple[list[dict], dict[str, int]]:
    rows = []
    stats = {
        "total": len(experiments),
        "excluded_blacklisted": 0,
        "excluded_uncapped": 0,
        "excluded_perturbed": 0,
        "skipped_missing_paths": 0,
        "written": 0,
    }

    for exp_id, exp in experiments.items():
        warning = warning_flags.get(exp_id)
        if warning is None:
            raise ValueError(
                f"{exp_id} is missing from the warning flag table; "
                "regenerate configs/model_warning_flags.tsv"
            )

        is_blacklisted = "red_manual" in warning["flags"]
        is_uncapped = warning["is_uncapped"]
        is_perturbed = warning["is_perturbation"]

        if is_blacklisted and not include_blacklisted:
            stats["excluded_blacklisted"] += 1
            continue

        if is_uncapped and not include_uncapped:
            stats["excluded_uncapped"] += 1
            continue
        if is_perturbed and not include_perturbed:
            stats["excluded_perturbed"] += 1
            continue

        processed = exp.get("processed", {})
        fwd = processed.get("pl_bigwig", "")
        rev = processed.get("mn_bigwig", "")
        if not fwd or not rev:
            stats["skipped_missing_paths"] += 1
            continue

        if require_files and (
            not (REPO_ROOT / fwd).exists() or not (REPO_ROOT / rev).exists()
        ):
            stats["skipped_missing_paths"] += 1
            continue

        biosample = str(exp.get("biosample", ""))
        rows.append(
            {
                "fwd": output_path_for_bigwig(fwd, absolute_paths),
                "rev": output_path_for_bigwig(rev, absolute_paths),
                "xform": xform,
                "assay": "PRO-cap",
                "target": normalize_target(exp_id, biosample),
                "experiment": exp_id,
                "biosample": biosample,
                "biosample_summary": str(exp.get("biosample_summary", "")),
                "library_construction": str(exp.get("library_construction", "")),
            }
        )

    stats["written"] = len(rows)
    return rows, stats


def write_tsv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    try:
        experiments = load_experiments(args.config)
        warning_flags = load_warning_flags(args.warning_flags)
        rows, stats = build_rows(
            experiments=experiments,
            warning_flags=warning_flags,
            xform=args.xform,
            include_uncapped=args.include_uncapped,
            include_perturbed=args.include_perturbed,
            include_blacklisted=args.include_blacklisted,
            absolute_paths=args.absolute_paths,
            require_files=args.require_files,
        )
        write_tsv(rows, args.output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote PromoterAI BigWig TSV: {args.output}")
    print(f"Warning flags: {args.warning_flags}")
    print(f"Total experiments: {stats['total']}")
    print(f"Excluded blacklisted: {stats['excluded_blacklisted']}")
    print(f"Excluded uncapped: {stats['excluded_uncapped']}")
    print(f"Excluded perturbed: {stats['excluded_perturbed']}")
    print(f"Skipped missing paths: {stats['skipped_missing_paths']}")
    print(f"Written rows: {stats['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
