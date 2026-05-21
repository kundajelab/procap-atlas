#!/usr/bin/env python3
"""Generate warning flags for PRO-cap atlas experiments.

The output is written as both a TSV for quick inspection and JSON for tools that
need structured flag reasons.

Usage:
    python src/analysis/generate_warning_flags.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metaformer.procap_config_to_promoterai import (  # noqa: E402
    DEFAULT_EXCLUDE_KEYWORDS,
    METADATA_FIELDS,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_config.yaml"
DEFAULT_READS = REPO_ROOT / "configs" / "n_reads.txt"
DEFAULT_TSV = REPO_ROOT / "configs" / "model_warning_flags.tsv"
DEFAULT_JSON = REPO_ROOT / "configs" / "model_warning_flags.json"

DEFAULT_YELLOW_READS = 20_000_000
DEFAULT_RED_READS = 10_000_000
DEFAULT_MANUAL_RED_EXPERIMENTS = {
    "ENCSR973QQI": "poor TSS-positioning",
}

TSV_FIELDS = [
    "experiment",
    "biosample",
    "overall_flag",
    "read_flag",
    "is_perturbation",
    "is_uncapped",
    "flags",
    "total_reads",
    "pl_reads",
    "mn_reads",
    "perturbation_keywords",
    "perturbation_fields",
    "uncapped_fields",
    "reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate green/yellow/red/perturb warning flags for experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"input experiment config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--reads",
        type=Path,
        default=DEFAULT_READS,
        help=f"input read-count TSV (default: {DEFAULT_READS})",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=DEFAULT_TSV,
        help=f"human-readable output TSV (default: {DEFAULT_TSV})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON,
        help=f"machine-readable output JSON (default: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--yellow-read-threshold",
        type=int,
        default=DEFAULT_YELLOW_READS,
        help="yellow-flag experiments below this total read count (default: 20M)",
    )
    parser.add_argument(
        "--red-read-threshold",
        type=int,
        default=DEFAULT_RED_READS,
        help="red-flag experiments below this total read count (default: 10M)",
    )
    parser.add_argument(
        "--perturb-keyword",
        action="append",
        default=[],
        help=(
            "additional case-insensitive metadata keyword for perturbation flags; "
            "may be repeated"
        ),
    )
    parser.add_argument(
        "--manual-red-experiment",
        action="append",
        default=[],
        metavar="EXP_ID:REASON",
        help=(
            "additional manually red-flagged experiment and reason; may be repeated "
            "(example: ENCSR000ABC:failed QC)"
        ),
    )
    return parser.parse_args()


def load_experiments(config_path: Path) -> dict[str, dict[str, Any]]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiments = config.get("experiments") if isinstance(config, dict) else None
    if not isinstance(experiments, dict):
        raise ValueError(f"{config_path} does not contain an experiments mapping")
    return experiments


def parse_int(value: str) -> int | None:
    if value == "":
        return None
    return int(float(value))


def load_read_counts(reads_path: Path) -> dict[str, dict[str, int | None]]:
    with open(reads_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"experiment", "pl_reads", "mn_reads", "total_reads"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{reads_path} is missing required columns: {', '.join(sorted(missing))}"
            )
        return {
            row["experiment"]: {
                "pl_reads": parse_int(row.get("pl_reads", "")),
                "mn_reads": parse_int(row.get("mn_reads", "")),
                "total_reads": parse_int(row.get("total_reads", "")),
            }
            for row in reader
        }


def parse_manual_red(values: list[str]) -> dict[str, str]:
    manual = dict(DEFAULT_MANUAL_RED_EXPERIMENTS)
    for value in values:
        if ":" not in value:
            raise ValueError(
                "--manual-red-experiment values must use EXP_ID:REASON format"
            )
        exp_id, reason = value.split(":", 1)
        exp_id = exp_id.strip()
        reason = reason.strip()
        if not exp_id or not reason:
            raise ValueError(
                "--manual-red-experiment values must include both EXP_ID and REASON"
            )
        manual[exp_id] = reason
    return manual


def find_perturbation_matches(
    exp: dict[str, Any], keywords: list[str]
) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for field in METADATA_FIELDS:
        value = str(exp.get(field, "")).lower()
        field_matches = [keyword for keyword in keywords if keyword in value]
        if field_matches:
            matches[field] = field_matches
    return matches


def find_uncapped_matches(exp: dict[str, Any]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for field in METADATA_FIELDS:
        value = str(exp.get(field, "")).lower()
        if "uncapped" in value:
            matches[field] = ["uncapped"]
    return matches


def read_flag(total_reads: int | None, red_threshold: int, yellow_threshold: int) -> str:
    if total_reads is None:
        return "missing"
    if total_reads < red_threshold:
        return "red"
    if total_reads < yellow_threshold:
        return "yellow"
    return "green"


def build_rows(
    experiments: dict[str, dict[str, Any]],
    read_counts: dict[str, dict[str, int | None]],
    keywords: list[str],
    manual_red_experiments: dict[str, str],
    red_threshold: int,
    yellow_threshold: int,
) -> list[dict[str, Any]]:
    rows = []

    for exp_id, exp in experiments.items():
        reads = read_counts.get(exp_id, {})
        total_reads = reads.get("total_reads")
        flag = read_flag(total_reads, red_threshold, yellow_threshold)
        reasons = []
        flags = []

        if flag == "red":
            flags.append("red_low_reads")
            reasons.append(
                {
                    "flag": "red",
                    "rule": "total_reads < red_read_threshold",
                    "value": total_reads,
                    "threshold": red_threshold,
                    "reason": f"total reads {total_reads:,} < {red_threshold:,}",
                }
            )
        elif flag == "yellow":
            flags.append("yellow_low_reads")
            reasons.append(
                {
                    "flag": "yellow",
                    "rule": "total_reads < yellow_read_threshold",
                    "value": total_reads,
                    "threshold": yellow_threshold,
                    "reason": f"total reads {total_reads:,} < {yellow_threshold:,}",
                }
            )
        elif flag == "missing":
            flags.append("missing_read_count")
            reasons.append(
                {
                    "flag": "missing",
                    "rule": "experiment absent from read-count TSV",
                    "value": None,
                    "reason": "no total_reads value available",
                }
            )

        perturb_matches = find_perturbation_matches(exp, keywords)
        perturb_keywords = sorted(
            {keyword for field_keywords in perturb_matches.values() for keyword in field_keywords}
        )
        is_perturbation = bool(perturb_matches)
        if is_perturbation:
            flags.append("perturbation")
            reasons.append(
                {
                    "flag": "perturb",
                    "rule": "metadata keyword match",
                    "fields": perturb_matches,
                    "reason": (
                        "metadata matched perturbation keywords: "
                        + ", ".join(perturb_keywords)
                    ),
                }
            )

        uncapped_matches = find_uncapped_matches(exp)
        is_uncapped = bool(uncapped_matches)
        if is_uncapped:
            flags.append("red_uncapped")
            reasons.append(
                {
                    "flag": "red",
                    "rule": "metadata contains uncapped",
                    "fields": uncapped_matches,
                    "reason": "uncapped experiment",
                }
            )

        if exp_id in manual_red_experiments:
            flags.append("red_manual")
            reasons.append(
                {
                    "flag": "red",
                    "rule": "manual red-list",
                    "reason": manual_red_experiments[exp_id],
                }
            )

        overall_flag = "red" if any(reason["flag"] == "red" for reason in reasons) else flag
        if overall_flag == "missing":
            overall_flag = "yellow"
        if not flags:
            flags.append("green")

        row = {
            "experiment": exp_id,
            "biosample": str(exp.get("biosample", "")),
            "overall_flag": overall_flag,
            "read_flag": flag,
            "is_perturbation": is_perturbation,
            "is_uncapped": is_uncapped,
            "flags": flags,
            "total_reads": total_reads,
            "pl_reads": reads.get("pl_reads"),
            "mn_reads": reads.get("mn_reads"),
            "perturbation_keywords": perturb_keywords,
            "perturbation_fields": perturb_matches,
            "uncapped_fields": uncapped_matches,
            "reasons": reasons,
        }
        rows.append(row)

    return rows


def tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            return json.dumps(value, sort_keys=True)
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def write_tsv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: tsv_value(row.get(field)) for field in TSV_FIELDS})


def write_json(
    rows: list[dict[str, Any]],
    output_path: Path,
    config_path: Path,
    reads_path: Path,
    keywords: list[str],
    red_threshold: int,
    yellow_threshold: int,
    manual_red_experiments: dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "config": str(config_path),
            "read_counts": str(reads_path),
            "red_read_threshold": red_threshold,
            "yellow_read_threshold": yellow_threshold,
            "perturbation_definition_source": (
                "src/metaformer/procap_config_to_promoterai.py "
                "DEFAULT_EXCLUDE_KEYWORDS and METADATA_FIELDS"
            ),
            "perturbation_metadata_fields": list(METADATA_FIELDS),
            "perturbation_keywords": keywords,
            "manual_red_experiments": manual_red_experiments,
        },
        "experiments": rows,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    args = parse_args()
    if args.red_read_threshold >= args.yellow_read_threshold:
        raise ValueError("--red-read-threshold must be less than --yellow-read-threshold")

    keywords = sorted(
        {keyword.lower() for keyword in (*DEFAULT_EXCLUDE_KEYWORDS, *args.perturb_keyword)}
    )
    manual_red_experiments = parse_manual_red(args.manual_red_experiment)

    experiments = load_experiments(args.config)
    read_counts = load_read_counts(args.reads)
    rows = build_rows(
        experiments=experiments,
        read_counts=read_counts,
        keywords=keywords,
        manual_red_experiments=manual_red_experiments,
        red_threshold=args.red_read_threshold,
        yellow_threshold=args.yellow_read_threshold,
    )

    write_tsv(rows, args.output_tsv)
    write_json(
        rows=rows,
        output_path=args.output_json,
        config_path=args.config,
        reads_path=args.reads,
        keywords=keywords,
        red_threshold=args.red_read_threshold,
        yellow_threshold=args.yellow_read_threshold,
        manual_red_experiments=manual_red_experiments,
    )

    counts = {}
    for row in rows:
        counts[row["overall_flag"]] = counts.get(row["overall_flag"], 0) + 1
    n_perturb = sum(1 for row in rows if row["is_perturbation"])
    print(f"Wrote {len(rows)} experiments to {args.output_tsv}")
    print(f"Wrote structured output to {args.output_json}")
    print(
        "Overall flags: "
        + ", ".join(f"{flag}={counts[flag]}" for flag in sorted(counts))
    )
    print(f"Perturbation flags: {n_perturb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
