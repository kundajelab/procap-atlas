#!/usr/bin/env python3
"""Generate final predicted PRO-cap BigWigs from trained BPNet fold models.

Unlike benchmark_bpnet.py, this script predicts every filtered peak with every
fold checkpoint for an experiment. Profile logits and log-count predictions are
averaged across checkpoints before conversion to count-scale signal and writing
as strand-specific BigWigs for visualization.

Usage:
    python src/bpnet/predict/generate_predicted_tracks.py -e ENCSR882DWM
    python src/bpnet/predict/generate_predicted_tracks.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_gc0.1
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pybigtools
import torch
import yaml
from tangermeme.io import extract_loci
from tangermeme.predict import predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.profile import count_scaled_profile

CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
DEFAULT_FASTA = REPO_ROOT / "data" / "hg38.fa"
DEFAULT_BLACKLIST = REPO_ROOT / "data" / "hg38.blacklist.bed.gz"
DEFAULT_CHROM_SIZES = REPO_ROOT / "data" / "hg38.chrom.sizes"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "predictions" / "bpnet" / "bigwigs"
IGNORE_BASES = list("QWERYUIOPSDFHJKLZXVBNM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--experiment",
        required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="override model directory, default: models/bpnet/{experiment}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for predicted plus/minus BigWigs",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--chrom-splits", type=Path, default=CHROM_SPLITS_PATH)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
    parser.add_argument("--chrom-sizes", type=Path, default=DEFAULT_CHROM_SIZES)
    parser.add_argument("--in-window", type=int, default=2114)
    parser.add_argument("--out-window", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="prediction device; auto uses CUDA when available",
    )
    parser.add_argument(
        "--merge-equal-values",
        action="store_true",
        help="write compact spans for adjacent equal scores instead of one-base intervals",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def scale_profile_logits(profile_logits, log_counts) -> np.ndarray:
    """Convert profile logits and log-count predictions to count-scale profiles."""
    logits = np.asarray(profile_logits, dtype=np.float64)
    scaled = count_scaled_profile(logits, log_counts)
    return scaled.reshape(logits.shape).astype(np.float32, copy=False)


class FoldPredictionAccumulator:
    """Streams per-fold profile logits/log-counts to average across folds.

    Sums fold outputs incrementally rather than collecting them into lists, so
    peak-level predictions for a whole experiment stay in memory for only one
    fold at a time instead of all folds at once.
    """

    def __init__(self) -> None:
        self._logits_sum: np.ndarray | None = None
        self._log_counts_sum: np.ndarray | None = None
        self._n_folds = 0

    def add(self, profile_logits: np.ndarray, log_counts: np.ndarray) -> None:
        if self._logits_sum is None:
            self._logits_sum = np.zeros_like(profile_logits, dtype=np.float64)
            self._log_counts_sum = np.zeros_like(log_counts, dtype=np.float64)
        elif self._logits_sum.shape != profile_logits.shape:
            raise ValueError(
                f"profile prediction shape changed from {self._logits_sum.shape} "
                f"to {profile_logits.shape}"
            )
        elif self._log_counts_sum.shape != log_counts.shape:
            raise ValueError(
                f"log-count prediction shape changed from "
                f"{self._log_counts_sum.shape} to {log_counts.shape}"
            )
        self._logits_sum += profile_logits
        self._log_counts_sum += log_counts
        self._n_folds += 1

    def finalize(self) -> np.ndarray:
        """Average accumulated fold outputs, then convert to count-scale profiles."""
        if self._n_folds == 0:
            raise ValueError("at least one fold prediction is required")
        return scale_profile_logits(
            self._logits_sum / self._n_folds,
            self._log_counts_sum / self._n_folds,
        )


def prediction_to_strand_scores(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a prediction tensor into plus and minus BigWig score arrays."""
    pred = np.asarray(prediction)
    if pred.ndim != 3:
        raise ValueError(
            "prediction must be 3D with shape (N, 2, L) or (N, L, 2); "
            f"got {pred.shape}"
        )

    if pred.shape[1] == 2:
        plus = pred[:, 0, :]
        minus = pred[:, 1, :]
    elif pred.shape[2] == 2:
        plus = pred[:, :, 0]
        minus = pred[:, :, 1]
    else:
        raise ValueError(
            "could not identify strand channel axis; expected dimension 2 "
            f"in shape {pred.shape}"
        )

    return plus, -minus


def make_windows(
    loci: pd.DataFrame,
    scores: np.ndarray,
    chrom_sizes: dict[str, int],
    out_window: int,
) -> dict[str, list[tuple[int, int, int, int, int]]]:
    if len(loci) != scores.shape[0]:
        raise ValueError(
            f"number of loci ({len(loci)}) does not match prediction rows "
            f"({scores.shape[0]})"
        )

    half_left = out_window // 2
    windows = defaultdict(list)
    skipped = 0
    for idx, row in enumerate(loci.itertuples(index=False)):
        chrom_size = chrom_sizes.get(row.chrom)
        if chrom_size is None:
            skipped += 1
            continue

        center = (int(row.start) + int(row.end)) // 2
        start = center - half_left
        end = start + out_window
        clipped_start = max(start, 0)
        clipped_end = min(end, chrom_size)
        if clipped_start >= clipped_end:
            skipped += 1
            continue

        score_start = clipped_start - start
        score_end = score_start + (clipped_end - clipped_start)
        windows[row.chrom].append((clipped_start, clipped_end, score_start, score_end, idx))

    if skipped:
        print(f"WARNING: skipped {skipped} loci outside known chromosomes", file=sys.stderr)
    for chrom in windows:
        windows[chrom].sort(key=lambda x: (x[0], x[1]))
    return dict(windows)


def iter_averaged_intervals(
    windows_by_chrom: dict[str, list[tuple[int, int, int, int, int]]],
    scores: np.ndarray,
    chrom_order: list[str],
    merge_equal_values: bool,
):
    for chrom in chrom_order:
        windows = windows_by_chrom.get(chrom)
        if not windows:
            continue

        cluster = []
        cluster_start = None
        cluster_end = None
        for window in windows:
            start, end, score_start, score_end, idx = window
            if cluster and start >= cluster_end:
                yield from _emit_cluster(
                    chrom, cluster_start, cluster_end, cluster, scores, merge_equal_values
                )
                cluster = []
                cluster_start = None
                cluster_end = None

            if not cluster:
                cluster_start = start
                cluster_end = end
            else:
                cluster_end = max(cluster_end, end)
            cluster.append((start, end, score_start, score_end, idx))

        if cluster:
            yield from _emit_cluster(
                chrom, cluster_start, cluster_end, cluster, scores, merge_equal_values
            )


def _emit_cluster(
    chrom,
    cluster_start,
    cluster_end,
    cluster,
    scores: np.ndarray,
    merge_equal_values: bool,
):
    length = cluster_end - cluster_start
    sums = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.uint32)

    for start, end, score_start, score_end, idx in cluster:
        dst_start = start - cluster_start
        dst_end = end - cluster_start
        sums[dst_start:dst_end] += scores[idx, score_start:score_end]
        counts[dst_start:dst_end] += 1

    covered = counts > 0
    averaged = np.zeros(length, dtype=np.float64)
    averaged[covered] = sums[covered] / counts[covered]

    if not merge_equal_values:
        for i, value in enumerate(averaged):
            if covered[i] and value != 0:
                yield chrom, cluster_start + i, cluster_start + i + 1, float(value)
        return

    i = 0
    while i < length:
        if not covered[i] or averaged[i] == 0:
            i += 1
            continue
        value = float(averaged[i])
        j = i + 1
        while j < length and covered[j] and averaged[j] == value:
            j += 1
        yield chrom, cluster_start + i, cluster_start + j, value
        i = j


def write_prediction_bigwigs(
    prediction: np.ndarray,
    loci: pd.DataFrame,
    chrom_sizes: dict[str, int],
    plus_output: Path,
    minus_output: Path,
    out_window: int,
    merge_equal_values: bool = False,
) -> None:
    """Write averaged predicted profiles to plus/minus BigWigs."""
    plus_scores, minus_scores = prediction_to_strand_scores(prediction)
    if plus_scores.shape[1] != out_window or minus_scores.shape[1] != out_window:
        raise ValueError(
            f"--out-window is {out_window}, but prediction length is "
            f"{plus_scores.shape[1]}"
        )

    chrom_order = list(chrom_sizes.keys())

    plus_windows = make_windows(loci, plus_scores, chrom_sizes, out_window)
    plus_output.parent.mkdir(parents=True, exist_ok=True)
    bw = pybigtools.open(str(plus_output), "w")
    bw.write(
        chrom_sizes,
        iter_averaged_intervals(
            plus_windows, plus_scores, chrom_order, merge_equal_values
        ),
    )

    minus_windows = make_windows(loci, minus_scores, chrom_sizes, out_window)
    bw = pybigtools.open(str(minus_output), "w")
    bw.write(
        chrom_sizes,
        iter_averaged_intervals(
            minus_windows, minus_scores, chrom_order, merge_equal_values
        ),
    )


def main() -> None:
    args = parse_args()
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        experiments = config["experiments"]
        if args.experiment not in experiments:
            raise ValueError(f"{args.experiment} not found in config")

        exp = experiments[args.experiment]
        processed = exp.get("processed", {})
        if "filtered_peaks" not in processed:
            raise ValueError(f"{args.experiment} has no processed.filtered_peaks entry")

        biosample = exp.get("biosample", "")
        peaks_path = REPO_ROOT / processed["filtered_peaks"]
        model_dir = args.model_dir or REPO_ROOT / "models" / "bpnet" / args.experiment
        with open(args.chrom_splits) as f:
            chrom_splits = yaml.safe_load(f)
        n_folds = len(chrom_splits["folds"])
        models = [
            model_dir / f"{args.experiment}.fold{fold}.torch"
            for fold in range(n_folds)
        ]

        for path, label in [
            (peaks_path, "filtered peaks"),
            (args.fasta, "FASTA"),
            (args.blacklist, "blacklist"),
            (args.chrom_sizes, "chrom sizes"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        missing_models = [path for path in models if not path.exists()]
        if missing_models:
            formatted = "\n".join(str(path) for path in missing_models)
            raise FileNotFoundError(f"missing model checkpoints:\n{formatted}")

        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("--device cuda was requested, but CUDA is not available")
        else:
            device = args.device
        print(f"Experiment: {args.experiment} ({biosample})")
        print(f"Model dir: {model_dir}")
        print(f"Peaks: {peaks_path}")
        print(f"Device: {device}")

        extracted = extract_loci(
            loci=str(peaks_path),
            sequences=str(args.fasta),
            in_window=args.in_window,
            out_window=args.out_window,
            verbose=args.verbose,
            ignore=IGNORE_BASES,
            exclusion_lists=[str(args.blacklist)],
            return_mask=True,
        )
        X = extracted[0]
        mask = extracted[-1]
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask).astype(bool)
        all_loci = pd.read_csv(
            peaks_path,
            sep="\t",
            usecols=[0, 1, 2],
            header=None,
            index_col=False,
            names=["chrom", "start", "end"],
            dtype={"chrom": str, "start": int, "end": int},
        )
        if mask.shape[0] != len(all_loci):
            raise ValueError(
                f"extraction mask length {mask.shape[0]} does not match "
                f"{len(all_loci)} loci"
            )
        loci = all_loci.loc[mask].reset_index(drop=True)
        if len(loci) != X.shape[0]:
            raise ValueError(
                f"retained loci ({len(loci)}) do not match sequences ({X.shape[0]})"
            )
        print(f"Predicting {len(loci):,} retained peaks across {n_folds} fold models")

        accumulator = FoldPredictionAccumulator()
        for i, model_path in enumerate(models):
            print(f"Predicting fold model {i + 1}/{len(models)}: {model_path}")
            model = torch.load(
                model_path, weights_only=False, map_location=torch.device("cpu")
            )
            pred = predict(
                model=model,
                X=X,
                verbose=args.verbose,
                device=device,
                batch_size=args.batch_size,
            )
            profile_logits = (
                pred[0].detach().cpu().numpy()
                if hasattr(pred[0], "detach")
                else np.asarray(pred[0])
            )
            log_counts = (
                pred[1].detach().cpu().numpy()
                if hasattr(pred[1], "detach")
                else np.asarray(pred[1])
            )
            accumulator.add(profile_logits, log_counts)

            del model, pred, profile_logits, log_counts
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        averaged = accumulator.finalize()

        chrom_sizes = {}
        with open(args.chrom_sizes) as f:
            for line in f:
                if not line.strip():
                    continue
                chrom, size = line.rstrip("\n").split("\t")[:2]
                chrom_sizes[chrom] = int(size)
        plus_output = args.output_dir / f"{model_dir.name}_pl.bigWig"
        minus_output = args.output_dir / f"{model_dir.name}_mn.bigWig"
        write_prediction_bigwigs(
            prediction=averaged,
            loci=loci,
            chrom_sizes=chrom_sizes,
            plus_output=plus_output,
            minus_output=minus_output,
            out_window=args.out_window,
            merge_equal_values=args.merge_equal_values,
        )
        print(f"Wrote {plus_output}")
        print(f"Wrote {minus_output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
