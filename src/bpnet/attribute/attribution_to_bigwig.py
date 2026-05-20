#!/usr/bin/env python3
"""Convert observed BPNet attribution scores from NPZ to dynseq BigWig.

The attribution scripts save hypothetical attributions with one channel per
base. This converter multiplies those scores by the matching one-hot-encoded
sequence and sums across channels, leaving only the score for the observed
nucleotide at each position.

Overlapping attribution windows are averaged base-by-base before writing.
By default, each nonzero score is emitted as its own one-base interval. This
matches UCSC dynseq's base-resolution logo renderer more reliably than compact
bedGraph-style spans, which are fine for normal BigWig bars but can disappear
when zoomed all the way in.

Usage:
    python src/bpnet/attribute/attribution_to_bigwig.py -e ENCSR882DWM
    python src/bpnet/attribute/attribution_to_bigwig.py -e ENCSR882DWM --head count
    python src/bpnet/attribute/attribution_to_bigwig.py --attr-npz attrs.npz --ohe-npz ohe.npz --peaks-bed peaks.bed.gz --output attrs.bigWig
"""

import argparse
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
DEFAULT_FASTA = REPO_ROOT / "data" / "hg38.fa"
DEFAULT_BLACKLIST = REPO_ROOT / "data" / "hg38.blacklist.bed.gz"
DEFAULT_CHROM_SIZES = REPO_ROOT / "data" / "hg38.chrom.sizes"
DEFAULT_ATTR_DIR = REPO_ROOT / "attributions" / "bpnet"
DEFAULT_OUTPUT_DIR = DEFAULT_ATTR_DIR / "bigwigs"


def load_npz_array(path: Path, label: str) -> np.ndarray:
    """Load a single array from an NPZ file, preferring arr_0."""
    with np.load(path) as data:
        keys = list(data.keys())
        if "arr_0" in data:
            return data["arr_0"]
        if len(keys) == 1:
            return data[keys[0]]
        raise ValueError(
            f"{label} NPZ has multiple arrays and no arr_0 key: {', '.join(keys)}"
        )


def observed_attribution(attributions: np.ndarray, ohe: np.ndarray) -> np.ndarray:
    """Return observed-nucleotide attribution scores as an (N, L) array."""
    if attributions.shape != ohe.shape:
        raise ValueError(
            "attribution and OHE arrays must have matching shapes; "
            f"got {attributions.shape} and {ohe.shape}"
        )
    if attributions.ndim != 3:
        raise ValueError(
            "attribution and OHE arrays must be 3D with shape (N, 4, L) "
            f"or (N, L, 4); got {attributions.shape}"
        )

    channel_axes = [axis for axis in (1, 2) if attributions.shape[axis] == 4]
    if not channel_axes:
        raise ValueError(
            "could not identify nucleotide channel axis; expected dimension 4 "
            f"in shape {attributions.shape}"
        )
    channel_axis = 1 if 1 in channel_axes else channel_axes[0]
    return np.sum(attributions * ohe, axis=channel_axis)


def load_chrom_sizes(path: Path) -> dict[str, int]:
    chroms = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            chrom, size = line.rstrip("\n").split("\t")[:2]
            chroms[chrom] = int(size)
    return chroms


def load_loci(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str, "start": int, "end": int},
    )


def load_chrom_splits(path: Path) -> dict[int, list[str]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data["folds"].items()}


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.experiment is None and (
        args.attr_npz is None or args.ohe_npz is None or args.peaks_bed is None
    ):
        raise ValueError(
            "--experiment is required unless --attr-npz, --ohe-npz, and "
            "--peaks-bed are all provided"
        )

    if args.experiment is not None:
        if args.attr_npz is None:
            args.attr_npz = DEFAULT_ATTR_DIR / f"{args.experiment}_{args.head}.npz"
        if args.ohe_npz is None:
            args.ohe_npz = DEFAULT_ATTR_DIR / f"{args.experiment}_ohe.npz"
        if args.peaks_bed is None:
            with open(args.config) as f:
                config = yaml.safe_load(f)
            experiments = config["experiments"]
            if args.experiment not in experiments:
                raise ValueError(f"{args.experiment} not found in {args.config}")
            processed = experiments[args.experiment].get("processed", {})
            if "filtered_peaks" not in processed:
                raise ValueError(
                    f"{args.experiment} has no processed.filtered_peaks entry"
                )
            args.peaks_bed = REPO_ROOT / processed["filtered_peaks"]
        if args.output is None:
            args.output = DEFAULT_OUTPUT_DIR / f"{args.experiment}_{args.head}.bigWig"

    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / (args.attr_npz.stem + ".bigWig")

    return args


def align_loci_to_scores(
    loci: pd.DataFrame,
    scores: np.ndarray,
    fasta: Path,
    blacklist: Path,
    chrom_splits: Path,
    in_window: int,
) -> pd.DataFrame:
    """Apply the same extraction mask used by attribution generation if needed."""
    if len(loci) == scores.shape[0]:
        return loci.reset_index(drop=True)

    for path, label in [
        (fasta, "FASTA"),
        (blacklist, "blacklist"),
        (chrom_splits, "chrom splits"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found while reconstructing retained loci mask: {path}"
            )

    try:
        from tangermeme.io import extract_loci
    except ImportError as e:
        raise ImportError(
            "tangermeme is required to reconstruct the retained locus mask when "
            "the peak BED row count differs from the attribution row count"
        ) from e

    all_chroms = list(chain.from_iterable(load_chrom_splits(chrom_splits).values()))
    extracted = extract_loci(
        loci=loci,
        sequences=str(fasta),
        chroms=all_chroms,
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
    mask = mask.astype(bool)

    if mask.shape[0] != len(loci):
        raise ValueError(
            "retained locus mask length does not match peak BED rows; "
            f"got mask length {mask.shape[0]} for {len(loci)} loci"
        )

    retained = loci.loc[mask].reset_index(drop=True)
    if len(retained) != scores.shape[0]:
        raise ValueError(
            "peak BED still does not match attribution rows after applying the "
            "same extraction mask; "
            f"raw loci={len(loci)}, retained loci={len(retained)}, "
            f"attribution rows={scores.shape[0]}. If this is a cross-peak or "
            "retrained attribution file, pass the matching --peaks-bed explicitly."
        )

    print(
        f"Applied extraction mask: retained {len(retained)} of {len(loci)} loci "
        f"to match attribution rows"
    )
    return retained


def make_windows(
    loci: pd.DataFrame,
    scores: np.ndarray,
    chrom_sizes: dict[str, int],
    in_window: int,
) -> dict[str, list[tuple[int, int, int, int, int]]]:
    """Return clipped windows grouped by chromosome.

    Each tuple is (chrom_start, chrom_end, score_start, score_end). The score
    slice accounts for any clipping at chromosome boundaries.
    """
    if len(loci) != scores.shape[0]:
        raise ValueError(
            f"number of loci ({len(loci)}) does not match attribution rows "
            f"({scores.shape[0]})"
        )

    half_left = in_window // 2
    windows = defaultdict(list)
    skipped = 0

    for idx, row in enumerate(loci.itertuples(index=False)):
        chrom = row.chrom
        chrom_size = chrom_sizes.get(chrom)
        if chrom_size is None:
            skipped += 1
            continue

        center = (int(row.start) + int(row.end)) // 2
        start = center - half_left
        end = start + in_window

        clipped_start = max(start, 0)
        clipped_end = min(end, chrom_size)
        if clipped_start >= clipped_end:
            skipped += 1
            continue

        score_start = clipped_start - start
        score_end = score_start + (clipped_end - clipped_start)
        windows[chrom].append((clipped_start, clipped_end, score_start, score_end, idx))

    if skipped:
        print(f"WARNING: skipped {skipped} loci outside known chromosomes", file=sys.stderr)

    for chrom in windows:
        windows[chrom].sort(key=lambda x: (x[0], x[1]))
    return dict(windows)


def iter_averaged_intervals(
    windows_by_chrom: dict[str, list[tuple[int, int, int, int, int]]],
    scores: np.ndarray,
    chrom_order: list[str],
):
    """Yield sorted BigWig intervals with overlaps averaged."""
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
                yield from _emit_cluster(chrom, cluster_start, cluster_end, cluster, scores)
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
            yield from _emit_cluster(chrom, cluster_start, cluster_end, cluster, scores)


def iter_base_intervals(
    windows_by_chrom: dict[str, list[tuple[int, int, int, int, int]]],
    scores: np.ndarray,
    chrom_order: list[str],
):
    """Yield one-base BigWig intervals after overlap averaging."""
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
                yield from _emit_base_cluster(
                    chrom, cluster_start, cluster_end, cluster, scores
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
            yield from _emit_base_cluster(
                chrom, cluster_start, cluster_end, cluster, scores
            )


def _emit_cluster(chrom, cluster_start, cluster_end, cluster, scores):
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


def _emit_base_cluster(chrom, cluster_start, cluster_end, cluster, scores):
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

    for i, value in enumerate(averaged):
        if covered[i] and value != 0:
            yield chrom, cluster_start + i, cluster_start + i + 1, float(value)


def write_bigwig(output: Path, chrom_sizes: dict[str, int], intervals) -> None:
    import pybigtools

    output.parent.mkdir(parents=True, exist_ok=True)
    bw = pybigtools.open(str(output), "w")
    bw.write(chrom_sizes, intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiment", type=str, default=None)
    parser.add_argument(
        "--head",
        type=str,
        default="profile",
        choices=["profile", "count"],
        help="attribution head for default input/output names",
    )
    parser.add_argument("--attr-npz", type=Path, default=None)
    parser.add_argument("--ohe-npz", type=Path, default=None)
    parser.add_argument("--peaks-bed", type=Path, default=None)
    parser.add_argument("--chrom-sizes", type=Path, default=DEFAULT_CHROM_SIZES)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
    parser.add_argument("--chrom-splits", type=Path, default=CHROM_SPLITS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--in-window", type=int, default=2114)
    parser.add_argument(
        "--merge-equal-values",
        action="store_true",
        help=(
            "write compact bedGraph-style spans for adjacent equal scores; "
            "default writes one-base intervals for UCSC dynseq logos"
        ),
    )
    args = parser.parse_args()

    try:
        args = resolve_default_paths(args)
        for path, label in [
            (args.attr_npz, "attribution NPZ"),
            (args.ohe_npz, "OHE NPZ"),
            (args.peaks_bed, "peaks BED"),
            (args.chrom_sizes, "chrom sizes"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        attributions = load_npz_array(args.attr_npz, "attribution")
        ohe = load_npz_array(args.ohe_npz, "OHE")
        scores = observed_attribution(attributions, ohe)
        if scores.shape[1] != args.in_window:
            raise ValueError(
                f"--in-window is {args.in_window}, but attribution length is "
                f"{scores.shape[1]}"
            )

        chrom_sizes = load_chrom_sizes(args.chrom_sizes)
        loci = load_loci(args.peaks_bed)
        loci = align_loci_to_scores(
            loci=loci,
            scores=scores,
            fasta=args.fasta,
            blacklist=args.blacklist,
            chrom_splits=args.chrom_splits,
            in_window=args.in_window,
        )
        windows = make_windows(loci, scores, chrom_sizes, args.in_window)
        interval_fn = (
            iter_averaged_intervals
            if args.merge_equal_values
            else iter_base_intervals
        )
        intervals = interval_fn(windows, scores, list(chrom_sizes.keys()))
        write_bigwig(args.output, chrom_sizes, intervals)
        print(f"Wrote {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
