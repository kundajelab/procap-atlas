#!/usr/bin/env python3
"""Convert saved BPNet prediction NPZ files to strand-specific BigWigs.

This consumes predictions saved by benchmark_bpnet.py --save-output. The saved
profile predictions are mapped back to genomic loci using the same fold/test
chromosome split and extraction mask used during benchmarking. Overlapping
prediction windows are averaged base-by-base before writing.

Usage:
    python src/bpnet/benchmark/predictions_to_bigwig.py -e ENCSR882DWM
    python src/bpnet/benchmark/predictions_to_bigwig.py -e ENCSR882DWM --pred-npz predictions/bpnet/ENCSR882DWM.npz
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bpnet.attribute.attribution_to_bigwig import (  # noqa: E402
    DEFAULT_BLACKLIST,
    DEFAULT_CHROM_SIZES,
    DEFAULT_FASTA,
    CHROM_SPLITS_PATH,
    CONFIG_PATH,
    iter_averaged_intervals,
    load_chrom_sizes,
    load_chrom_splits,
    load_loci,
    make_windows,
    write_bigwig,
)

DEFAULT_PRED_DIR = REPO_ROOT / "predictions" / "bpnet"
DEFAULT_OUTPUT_DIR = DEFAULT_PRED_DIR / "bigwigs"


def prediction_to_strand_scores(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return plus and negative-valued minus scores from a prediction array."""
    if prediction.ndim != 3:
        raise ValueError(
            "prediction arrays must be 3D with shape (N, 2, L) or (N, L, 2); "
            f"got {prediction.shape}"
        )

    strand_axes = [axis for axis in (1, 2) if prediction.shape[axis] == 2]
    if not strand_axes:
        raise ValueError(
            "could not identify strand axis; expected dimension 2 in shape "
            f"{prediction.shape}"
        )
    strand_axis = 1 if 1 in strand_axes else strand_axes[0]
    if strand_axis == 2:
        prediction = np.moveaxis(prediction, 2, 1)

    plus = np.asarray(prediction[:, 0, :], dtype=np.float64)
    minus = -np.abs(np.asarray(prediction[:, 1, :], dtype=np.float64))
    return plus, minus


def load_prediction_arrays(path: Path) -> dict[int, np.ndarray]:
    """Load predict_fold{i} arrays from an NPZ, keyed by fold number."""
    with np.load(path) as data:
        predictions = {}
        for key in data.files:
            m = re.fullmatch(r"predict_fold(\d+)", key)
            if m:
                predictions[int(m.group(1))] = data[key]
    if not predictions:
        raise ValueError(f"no predict_fold{{i}} arrays found in {path}")
    return dict(sorted(predictions.items()))


def fold_loci_mask(
    loci,
    test_chroms: list[str],
    fasta: Path,
    blacklist: Path,
    in_window: int,
):
    """Recreate the benchmark extraction mask for one fold."""
    try:
        from tangermeme.io import extract_loci
    except ImportError as e:
        raise ImportError(
            "tangermeme is required to reconstruct prediction loci"
        ) from e

    extracted = extract_loci(
        loci=loci,
        sequences=str(fasta),
        chroms=test_chroms,
        in_window=in_window,
        verbose=False,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        exclusion_lists=[str(blacklist)],
        return_mask=True,
    )
    mask = extracted[-1]
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)
    return mask.astype(bool)


def prediction_loci_and_scores(
    predictions: dict[int, np.ndarray],
    loci,
    chrom_splits: dict[int, list[str]],
    fasta: Path,
    blacklist: Path,
    in_window: int,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Return retained loci plus concatenated plus/minus prediction scores."""
    retained_loci = []
    plus_scores = []
    minus_scores = []

    for fold, prediction in predictions.items():
        if fold not in chrom_splits:
            raise ValueError(f"predict_fold{fold} has no matching chrom split")

        plus, minus = prediction_to_strand_scores(prediction)
        mask = fold_loci_mask(
            loci=loci,
            test_chroms=chrom_splits[fold],
            fasta=fasta,
            blacklist=blacklist,
            in_window=in_window,
        )
        fold_loci = loci.loc[mask].reset_index(drop=True)
        if len(fold_loci) != plus.shape[0]:
            raise ValueError(
                f"predict_fold{fold} row count does not match retained loci: "
                f"prediction rows={plus.shape[0]}, retained loci={len(fold_loci)}"
            )

        retained_loci.append(fold_loci)
        plus_scores.append(plus)
        minus_scores.append(minus)

    import pandas as pd

    return (
        pd.concat(retained_loci, ignore_index=True),
        np.concatenate(plus_scores, axis=0),
        np.concatenate(minus_scores, axis=0),
    )


def write_prediction_bigwigs(
    output_prefix: Path,
    loci,
    plus_scores: np.ndarray,
    minus_scores: np.ndarray,
    chrom_sizes: dict[str, int],
    out_window: int,
) -> tuple[Path, Path]:
    pl_path = output_prefix.with_name(output_prefix.name + "_pl.bigWig")
    mn_path = output_prefix.with_name(output_prefix.name + "_mn.bigWig")

    for scores, output in [(plus_scores, pl_path), (minus_scores, mn_path)]:
        windows = make_windows(loci, scores, chrom_sizes, out_window)
        intervals = iter_averaged_intervals(windows, scores, list(chrom_sizes.keys()))
        write_bigwig(output, chrom_sizes, intervals)

    return pl_path, mn_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("--pred-npz", type=Path, default=None)
    parser.add_argument("--peaks-bed", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--chrom-splits", type=Path, default=CHROM_SPLITS_PATH)
    parser.add_argument("--chrom-sizes", type=Path, default=DEFAULT_CHROM_SIZES)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
    parser.add_argument("--in-window", type=int, default=2114)
    parser.add_argument(
        "--out-window",
        type=int,
        default=None,
        help="prediction window size (default: infer from prediction array)",
    )
    args = parser.parse_args()

    try:
        if args.pred_npz is None:
            args.pred_npz = DEFAULT_PRED_DIR / f"{args.experiment}.npz"
        if args.output_prefix is None:
            args.output_prefix = DEFAULT_OUTPUT_DIR / args.experiment

        with open(args.config) as f:
            config = yaml.safe_load(f)
        experiments = config["experiments"]
        if args.experiment not in experiments:
            raise ValueError(f"{args.experiment} not found in {args.config}")
        if args.peaks_bed is None:
            args.peaks_bed = REPO_ROOT / experiments[args.experiment]["processed"]["peaks"]

        for path, label in [
            (args.pred_npz, "prediction NPZ"),
            (args.peaks_bed, "peaks BED"),
            (args.chrom_splits, "chrom splits"),
            (args.chrom_sizes, "chrom sizes"),
            (args.fasta, "FASTA"),
            (args.blacklist, "blacklist"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        predictions = load_prediction_arrays(args.pred_npz)
        first_plus, _ = prediction_to_strand_scores(next(iter(predictions.values())))
        out_window = args.out_window if args.out_window is not None else first_plus.shape[1]

        loci = load_loci(args.peaks_bed)
        chrom_splits = load_chrom_splits(args.chrom_splits)
        retained_loci, plus_scores, minus_scores = prediction_loci_and_scores(
            predictions=predictions,
            loci=loci,
            chrom_splits=chrom_splits,
            fasta=args.fasta,
            blacklist=args.blacklist,
            in_window=args.in_window,
        )
        if plus_scores.shape[1] != out_window:
            raise ValueError(
                f"--out-window is {out_window}, but prediction length is "
                f"{plus_scores.shape[1]}"
            )

        chrom_sizes = load_chrom_sizes(args.chrom_sizes)
        pl_path, mn_path = write_prediction_bigwigs(
            output_prefix=args.output_prefix,
            loci=retained_loci,
            plus_scores=plus_scores,
            minus_scores=minus_scores,
            chrom_sizes=chrom_sizes,
            out_window=out_window,
        )
        print(f"Wrote {pl_path}")
        print(f"Wrote {mn_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
