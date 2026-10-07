"""Call motif instances in Cherimoya attributions with Fi-NeMo.

Cherimoya-specific counterpart to src/bpnet/hitcall/call_hits_bpnet.py.
Builds a peak coordinate file aligned with the saved OHE/attribution arrays,
converts them to Fi-NeMo's region format, and runs Fi-NeMo hit calling against
this experiment's own per-experiment MoDISco motif set
(modisco/cherimoya/{experiment}_{head}.modisco.h5). Run
link_hits_to_compendium.py afterward to relabel hits with the atlas-wide
MotifCompendium cluster ID for cross-experiment comparability.

Shared Fi-NeMo logic (peaks.narrowPeak construction, regions.npz extraction,
trim-suffix resolution) is imported from src/bpnet/hitcall/call_hits_bpnet.py
-- these functions are model-family-agnostic.

Usage:
    python src/cherimoya/hitcall/call_hits_cherimoya.py -e ENCSR882DWM
    python src/cherimoya/hitcall/call_hits_cherimoya.py -e ENCSR882DWM --head count
    python src/cherimoya/hitcall/call_hits_cherimoya.py -e ENCSR882DWM --model-dir models/cherimoya/ENCSR882DWM
    python src/cherimoya/hitcall/call_hits_cherimoya.py -e ENCSR882DWM --global-lambda 0.6
"""

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_BPNET_HITCALL_DIR = str(REPO_ROOT / "src" / "bpnet" / "hitcall")
if _BPNET_HITCALL_DIR not in sys.path:
    sys.path.insert(0, _BPNET_HITCALL_DIR)

import compressed_io
from call_hits_bpnet import (
    DEFAULT_CWM_TRIM_THRESHOLD,
    HITS_FILE_STAGES,
    IN_WINDOW,
    ensure_regions_npz,
    resolve_hits_path,
    run,
    trim_suffix,
)

CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"


def resolve_experiment_paths(experiment, head, min_trim_len=None, model_dir=None):
    """Resolve (exp_dir, hits_dir, trim_coords, suffix) for a cherimoya experiment."""
    model_dir_name = Path(model_dir).name if model_dir else experiment
    modisco_dir = REPO_ROOT / "modisco" / "cherimoya"
    trim_coords = (
        modisco_dir / f"{experiment}_{head}_trim_coords_min{min_trim_len}bp.tsv"
        if min_trim_len is not None
        else None
    )
    suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, trim_coords)
    exp_dir = REPO_ROOT / "hitcalls" / "cherimoya" / f"{model_dir_name}_{head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir
    return exp_dir, hits_dir, trim_coords, suffix


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        default=None,
        help="override model directory, default derived from config",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="profile",
        choices=["profile", "count"],
        help="attribution/motif head to call hits against (default: profile)",
    )
    parser.add_argument(
        "--modisco-h5",
        type=str,
        default=None,
        help=(
            "override the modisco-lite-format h5 of motif CWMs to call hits "
            "against; default is this experiment's own "
            "modisco/cherimoya/{experiment}_{head}.modisco.h5"
        ),
    )
    parser.add_argument(
        "--region-width",
        type=int,
        default=IN_WINDOW,
        help="width of the region fed to Fi-NeMo hit calling (default: 2114)",
    )
    parser.add_argument(
        "--global-lambda",
        type=float,
        default=0.7,
        help="Fi-NeMo L1 sparsity weight (default: 0.7)",
    )
    parser.add_argument(
        "--cwm-trim-threshold",
        type=float,
        default=DEFAULT_CWM_TRIM_THRESHOLD,
        help="default motif trimming threshold (default: 0.3)",
    )
    parser.add_argument(
        "--cwm-trim-thresholds",
        type=str,
        default=None,
        help="path to a Fi-NeMo -T/--cwm-trim-thresholds mapping file",
    )
    parser.add_argument(
        "--cwm-trim-coords",
        type=str,
        default=None,
        help="path to a Fi-NeMo -R/--cwm-trim-coords mapping file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Fi-NeMo region batch size (default: 2000)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="JIT-compile the Fi-NeMo optimizer",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = config["experiments"]
    if args.experiment not in experiments:
        print(f"Error: {args.experiment} not found in config", file=sys.stderr)
        sys.exit(1)
    exp = experiments[args.experiment]
    processed = exp.get("processed", {})
    peaks_path = str(REPO_ROOT / processed["filtered_peaks"])

    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = {int(k): v for k, v in yaml.safe_load(f)["folds"].items()}

    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else REPO_ROOT / "models" / "cherimoya" / args.experiment
    )
    model_dir_name = model_dir.name

    attr_dir = REPO_ROOT / "attributions" / "cherimoya"
    ohe_path = attr_dir / f"{args.experiment}_ohe.npz"
    attr_path = attr_dir / f"{model_dir_name}_{args.head}.npz"
    if args.modisco_h5:
        modisco_h5 = Path(args.modisco_h5)
    else:
        modisco_h5 = (
            REPO_ROOT
            / "modisco"
            / "cherimoya"
            / f"{args.experiment}_{args.head}.modisco.h5"
        )

    for path, label in [
        (peaks_path, "filtered_peaks"),
        (ohe_path, "OHE sequences"),
        (attr_path, "attributions"),
        (modisco_h5, "motif CWMs"),
        (args.cwm_trim_thresholds, "cwm-trim-thresholds mapping"),
        (args.cwm_trim_coords, "cwm-trim-coords mapping"),
    ]:
        if path is not None and not compressed_io.exists(path):
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    out_dir = REPO_ROOT / "hitcalls" / "cherimoya" / f"{model_dir_name}_{args.head}"
    out_dir.mkdir(parents=True, exist_ok=True)

    regions_npz = ensure_regions_npz(
        peaks_path, chrom_splits, ohe_path, attr_path, out_dir,
        args.region_width, args.verbose,
    )

    suffix = trim_suffix(
        args.cwm_trim_threshold, args.cwm_trim_thresholds, args.cwm_trim_coords
    )
    call_hits_dir = out_dir / suffix.lstrip("_") if suffix else out_dir
    call_hits_dir.mkdir(parents=True, exist_ok=True)

    call_hits_cmd = [
        "finemo",
        "call-hits",
        "-r",
        str(regions_npz),
        "-m",
        str(modisco_h5),
        "-o",
        str(call_hits_dir),
        "-t",
        str(args.cwm_trim_threshold),
        "-l",
        str(args.global_lambda),
        "-b",
        str(args.batch_size),
    ]
    with ExitStack() as stack:
        if args.cwm_trim_thresholds:
            trim_thresholds = stack.enter_context(
                compressed_io.ensure_plain(args.cwm_trim_thresholds)
            )
            call_hits_cmd += ["-T", str(trim_thresholds)]
        if args.cwm_trim_coords:
            trim_coords = stack.enter_context(
                compressed_io.ensure_plain(args.cwm_trim_coords)
            )
            call_hits_cmd += ["-R", str(trim_coords)]
        if args.compile:
            call_hits_cmd.append("-J")
        run(call_hits_cmd, args.verbose)

    print(f"\nFi-NeMo hits saved to {call_hits_dir}")


if __name__ == "__main__":
    main()
