#!/usr/bin/env python3
"""
Extract observed and predicted counts at union peaks and visualize pairwise
experiment correlations.

For each experiment:
  - Observed counts: sum of plus and minus strand BigWig signal over a 1000 bp
    window centered on each union peak, normalized to RPM.
  - Predicted counts (optional, requires --model): load all fold models, predict
    at every union peak, average fold predictions, and sum over the output window.

Pairwise Pearson correlations of log1p-transformed counts are visualized as
a seaborn clustermap with experiments labeled by biosample.

Outputs (in --out-dir):
  observed_counts.tsv    — RPM count matrix (experiments × peaks)
  predicted_counts.tsv   — predicted count matrix (experiments × peaks)
  count_correlation.png  — clustermap(s) of pairwise experiment correlations

Usage:
    python src/analysis/count_correlation.py
    python src/analysis/count_correlation.py --model bpnet
    python src/analysis/count_correlation.py --experiment ENCSR882DWM --model bpnet
    python src/analysis/count_correlation.py --min-reads 10000000 --device cuda
"""

import argparse
from collections import Counter
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image
from tangermeme.io import extract_loci
from tangermeme.predict import predict
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
FASTA = str(REPO_ROOT / "data" / "hg38.fa")
BLACKLIST = str(REPO_ROOT / "data" / "hg38.blacklist.bed.gz")
DEFAULT_UNION_PEAKS = REPO_ROOT / "data" / "processed" / "peaks" / "union_peaks.bed.gz"
N_FOLDS = 7
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"


def extract_observed_counts(
    union_peaks: pd.DataFrame,
    pl_bw: str,
    mn_bw: str,
    total_reads: float,
) -> tuple[torch.Tensor, np.ndarray]:
    """Extract sequences and total observed counts (RPM) at each union peak.

    Uses a 1000 bp output window centered on each peak midpoint, matching the
    model output window. Both strands are summed (abs value for minus strand).

    Returns (X, obs_counts) where X has shape (n_peaks, 4, 2114) and
    obs_counts has shape (n_peaks,).
    """
    X, y = extract_loci(
        loci=union_peaks,
        sequences=FASTA,
        signals=[pl_bw, mn_bw],
        in_window=2114,
        out_window=1000,
        max_jitter=0,
        verbose=False,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        exclusion_lists=[BLACKLIST],
    )
    # y shape: (n_peaks, 2, 1000); sum over strand and position
    counts = torch.abs(y).sum(dim=(-1, -2)).numpy()
    rpm_scale = 1e6 / total_reads
    return X, counts * rpm_scale


def balanced_subset(
    experiments: dict, groups: dict[str, str], per_group: int,
    read_counts: dict[str, float], min_reads: float = 0.0,
    exclude_groups: tuple[str, ...] = ("other",),
) -> dict:
    """The `per_group` deepest experiments from each tissue group.

    A full all-pairs matrix over 198 experiments is far more GPU time than the
    three-tier contrast needs, and an unbalanced subset would make the
    same-tissue tier mostly blood_immune, which is 41 of 198 experiments.
    Taking the deepest per group also holds model quality roughly fixed, since
    accuracy tracks read depth.

    `min_reads` is applied here rather than left to the caller's later filter:
    selecting the three deepest of a group and *then* dropping the shallow ones
    can leave a group with one usable experiment and no within-group pair,
    which is the tier the analysis exists to measure. `exclude_groups` drops
    the `other` bucket, which is the unmatched-biosample fallback rather than a
    lineage.
    """
    if per_group <= 0:
        return experiments
    by_group: dict[str, list[str]] = {}
    for exp_id in experiments:
        group = groups.get(exp_id)
        if group is None or group in exclude_groups:
            continue
        if read_counts.get(exp_id, 0) < min_reads:
            continue
        by_group.setdefault(group, []).append(exp_id)
    keep = []
    for group, members in sorted(by_group.items()):
        members.sort(key=lambda e: read_counts.get(e, 0), reverse=True)
        keep.extend(members[:per_group])
    return {e: experiments[e] for e in keep}


def load_chrom_folds(path: Path = CHROM_SPLITS_PATH) -> dict[int, set[str]]:
    """fold index -> the chromosomes held out as that fold's test set."""
    with open(path) as f:
        folds = yaml.safe_load(f)["folds"]
    return {int(k): set(v) for k, v in folds.items()}


def fold_by_chrom(folds: dict[int, set[str]]) -> dict[str, int]:
    """chromosome -> the single fold that holds it out.

    Raises if a chromosome appears in two folds, which would mean no fold
    actually held it out and the "held-out" guarantee is void.
    """
    out: dict[str, int] = {}
    for fold, chroms in sorted(folds.items()):
        for chrom in chroms:
            if chrom in out:
                raise ValueError(
                    f"{chrom} appears in folds {out[chrom]} and {fold}; "
                    "held-out predictions would not be held out"
                )
            out[chrom] = fold
    return out


def split_peaks_by_fold(
    union_peaks: pd.DataFrame, assignment: dict[str, int]
) -> list[tuple[int, pd.DataFrame]]:
    """Peaks grouped by the fold that held their chromosome out, fold order.

    Fold order is fixed and ascending so that concatenating each fold's
    results yields vectors that line up across experiments. That is what makes
    held-out extraction work at all: `extract_loci` silently drops peaks
    (blacklist, invalid sequence) without reporting which, so per-peak identity
    is unrecoverable -- but the filtering depends only on the peaks, the FASTA
    and the blacklist, never on the experiment, so every experiment loses
    exactly the same rows and a fixed concatenation order keeps them aligned.

    Peaks on chromosomes no fold claims are dropped; the caller reports them.
    """
    known = union_peaks["chrom"].map(assignment)
    out = []
    for fold in sorted(set(assignment.values())):
        sub = union_peaks[known == fold]
        if len(sub):
            out.append((fold, sub))
    return out


def unassigned_chroms(union_peaks: pd.DataFrame, assignment: dict[str, int]) -> list[str]:
    """Chromosomes present in the peaks that no fold holds out."""
    return sorted(set(union_peaks["chrom"]) - set(assignment))


def extract_heldout_counts(
    union_peaks: pd.DataFrame,
    pl_bw: str,
    mn_bw: str,
    total_reads: float,
    model_dir: Path,
    exp_id: str,
    assignment: dict[str, int],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Observed and held-out predicted counts, concatenated in fold order.

    Each peak is predicted by the one fold model that never trained on its
    chromosome. `extract_predicted_counts` instead averages all seven folds at
    every peak, which inflates a model's apparent accuracy on its *own*
    experiment -- six of the seven folds trained on those peaks -- while giving
    no such advantage when predicting a different experiment. That is exactly
    the quantity a matched-versus-mismatched comparison measures, so the
    averaged version cannot be used for it.

    Costs less, not more: one prediction per peak rather than seven.
    """
    obs_parts, pred_parts = [], []
    for fold, peaks in split_peaks_by_fold(union_peaks, assignment):
        X, obs = extract_observed_counts(peaks, pl_bw, mn_bw, total_reads)
        obs_parts.append(obs)
        model_path = model_dir / f"{exp_id}.fold{fold}.torch"
        if not model_path.exists():
            return np.concatenate(obs_parts) if obs_parts else np.array([]), None
        model = torch.load(model_path, weights_only=False, map_location="cpu")
        pred = predict(
            model=model, X=X, batch_size=batch_size, device=device, verbose=False
        )
        scaled = (
            torch.nn.functional.softmax(pred[0].reshape(pred[0].shape[0], -1), dim=-1)
            * torch.exp(pred[1])
        ).reshape(*pred[0].shape)
        pred_parts.append(scaled.sum(dim=(-1, -2)).numpy())
    if not obs_parts:
        return np.array([]), None
    return np.concatenate(obs_parts), np.concatenate(pred_parts)


def extract_predicted_counts(
    X: torch.Tensor,
    model_dir: Path,
    exp_id: str,
    batch_size: int,
    device: str,
) -> np.ndarray | None:
    """Average predicted counts across all fold models at every union peak.

    Loads each fold model, predicts using the pre-extracted sequence tensor X,
    sums the scaled predicted profile over strand and position to get total
    predicted counts per peak, and averages across folds.

    X should come from extract_observed_counts to ensure the same blacklist
    and sequence-validity filtering is applied.

    Returns array of shape (n_peaks,), or None if no fold models are found.
    """
    fold_counts = []

    for fold in range(N_FOLDS):
        model_path = model_dir / f"{exp_id}.fold{fold}.torch"
        if not model_path.exists():
            continue
        model = torch.load(model_path, weights_only=False, map_location="cpu")
        pred = predict(model=model, X=X, batch_size=batch_size, device=device, verbose=False)
        # pred: (profile_logits, log_counts)
        # scaled = softmax(profile) * exp(log_counts), shape (n_peaks, 2, 1000)
        scaled = (
            torch.nn.functional.softmax(pred[0].reshape(pred[0].shape[0], -1), dim=-1)
            * torch.exp(pred[1])
        ).reshape(*pred[0].shape)
        fold_counts.append(scaled.sum(dim=(-1, -2)).numpy())

    if not fold_counts:
        return None
    return np.mean(fold_counts, axis=0)


def plot_clustermaps(
    obs_df: pd.DataFrame,
    pred_df: pd.DataFrame | None,
    biosample_map: dict[str, str],
    out_path: Path,
) -> None:
    """Plot seaborn clustermaps of pairwise experiment correlations.

    Experiments are clustered by Ward linkage on 1 - Pearson correlation of
    log1p-transformed count vectors. A color strip labels each experiment by
    biosample.
    """
    # Build biosample color mapping
    biosamples = [biosample_map.get(exp, "unknown") for exp in obs_df.index]
    unique_biosamples = sorted(set(biosamples))
    palette = sns.color_palette("tab20", len(unique_biosamples))
    biosample_colors = dict(zip(unique_biosamples, palette))
    row_colors = pd.Series(biosamples, index=obs_df.index).map(biosample_colors)

    def make_corr(df: pd.DataFrame) -> pd.DataFrame:
        log_counts = np.log1p(df.values)
        corr = np.corrcoef(log_counts)
        return pd.DataFrame(corr, index=df.index, columns=df.index)

    panels = [("Observed counts", make_corr(obs_df))]
    if pred_df is not None:
        panels.append(("Predicted counts", make_corr(pred_df)))

    n_panels = len(panels)
    # clustermap handles its own figure, so we plot separately and combine
    clustermap_paths = []
    for label, corr_df in panels:
        g = sns.clustermap(
            corr_df,
            method="ward",
            metric="euclidean",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            row_colors=row_colors,
            col_colors=row_colors,
            figsize=(max(8, len(corr_df) * 0.15 + 2), max(8, len(corr_df) * 0.15 + 2)),
            xticklabels=False,
            yticklabels=True if len(corr_df) <= 60 else False,
        )
        g.fig.suptitle(f"Pairwise Pearson correlation — {label}", y=1.01)

        # Add biosample legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=biosample_colors[b], label=b)
            for b in unique_biosamples
        ]
        g.ax_heatmap.legend(
            handles=handles,
            title="Biosample",
            bbox_to_anchor=(1.25, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
        )

        suffix = "observed" if label.startswith("Observed") else "predicted"
        p = out_path.with_name(out_path.stem + f"_{suffix}" + out_path.suffix)
        g.fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        clustermap_paths.append(p)
        print(f"Saved {p}", file=sys.stderr)

    # If both panels, also save a combined figure (side-by-side images)
    if n_panels == 2:
        imgs = [Image.open(p) for p in clustermap_paths]
        w = sum(im.width for im in imgs)
        h = max(im.height for im in imgs)
        combined = Image.new("RGB", (w, h), (255, 255, 255))
        x = 0
        for im in imgs:
            combined.paste(im, (x, 0))
            x += im.width
        combined.save(out_path)
        print(f"Combined figure saved to {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Count correlation analysis at union peaks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--union-peaks",
        type=Path,
        default=DEFAULT_UNION_PEAKS,
        metavar="PATH",
        help=f"union peaks BED file (default: {DEFAULT_UNION_PEAKS.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--model",
        choices=["bpnet", "cherimoya"],
        default=None,
        metavar="MODEL",
        help="model type for predictions (default: observed only)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="override default models/{model}/ directory",
    )
    parser.add_argument(
        "--min-reads",
        type=float,
        default=0,
        metavar="N",
        help="skip experiments with fewer than N total reads (default: 0)",
    )
    parser.add_argument(
        "--experiment",
        metavar="EXP_ID",
        help="run only this experiment (for testing)",
    )
    parser.add_argument(
        "--balanced-per-group",
        type=int,
        default=0,
        metavar="N",
        help="run only the N deepest experiments per tissue group. A full "
             "all-pairs matrix over 198 experiments is more GPU time than the "
             "three-tier contrast needs, and an unbalanced subset would make "
             "the same-tissue tier mostly blood_immune (41 of 198). 0 = all.",
    )
    parser.add_argument(
        "--held-out-folds",
        action="store_true",
        help="predict each peak with the one fold model that did not train on "
             "its chromosome, instead of averaging all 7 folds. Required for "
             "any matched-vs-mismatched comparison: fold averaging lets a "
             "model see its own experiment's peaks in training and so inflates "
             "the diagonal. Also 7x cheaper.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="batch size for model predictions (default: 64)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="device for model predictions (default: cpu)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "figures" / "count_correlation",
        metavar="DIR",
        help="output directory (default: figures/count_correlation/)",
    )
    args = parser.parse_args()

    if not args.union_peaks.exists():
        print(f"ERROR: union peaks not found: {args.union_peaks}", file=sys.stderr)
        print("Run src/preprocess/make_union_peaks.py first.", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    n_reads_map = {}
    if N_READS_PATH.exists():
        with open(N_READS_PATH) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    n_reads_map[parts[0]] = float(parts[4])

    union_peaks = pd.read_csv(
        args.union_peaks,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )
    print(f"Loaded {len(union_peaks):,} union peaks", file=sys.stderr)

    experiments = config["experiments"]
    if args.experiment:
        if args.experiment not in experiments:
            print(f"ERROR: {args.experiment!r} not in config", file=sys.stderr)
            sys.exit(1)
        experiments = {args.experiment: experiments[args.experiment]}

    if args.balanced_per_group > 0:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _biosample_groups import load_group_map

        tissue, _ = load_group_map(experiments, None, quiet=True)
        before = len(experiments)
        experiments = balanced_subset(
            experiments, tissue, args.balanced_per_group, n_reads_map,
            min_reads=args.min_reads,
        )
        if not experiments:
            print(
                "ERROR: balanced subset is empty; lower --min-reads or "
                "--balanced-per-group",
                file=sys.stderr,
            )
            sys.exit(1)
        sizes = Counter(tissue[e] for e in experiments)
        singles = sorted(g for g, n in sizes.items() if n < 2)
        print(
            f"Balanced subset: {len(experiments)} of {before} experiments, "
            f"up to {args.balanced_per_group} per group over {len(sizes)} groups",
            file=sys.stderr,
        )
        if singles:
            print(
                f"NOTE: {len(singles)} group(s) contribute one experiment and so "
                f"no same-tissue pair: {', '.join(singles)}",
                file=sys.stderr,
            )

    fold_assignment: dict[str, int] = {}
    if args.held_out_folds:
        fold_assignment = fold_by_chrom(load_chrom_folds())
        skipped = unassigned_chroms(union_peaks, fold_assignment)
        if skipped:
            print(
                f"NOTE: dropping peaks on {len(skipped)} chromosome(s) no fold "
                f"holds out: {', '.join(skipped)}",
                file=sys.stderr,
            )

    obs_rows: dict[str, np.ndarray] = {}
    pred_rows: dict[str, np.ndarray] = {}
    biosample_map: dict[str, str] = {}

    for exp_id, exp in tqdm(experiments.items(), unit="exp"):
        total_reads = n_reads_map.get(exp_id, 0)
        if total_reads < args.min_reads:
            continue

        biosample = exp.get("biosample", "unknown")
        biosample_safe = re.sub(r"[^\w-]", "_", biosample).strip("_")
        processed = exp.get("processed", {})
        pl_path = str(REPO_ROOT / processed["pl_bigwig"])
        mn_path = str(REPO_ROOT / processed["mn_bigwig"])

        missing = [p for p in (pl_path, mn_path) if not Path(p).exists()]
        if missing:
            print(
                f"WARNING: {exp_id}: missing {[Path(p).name for p in missing]}, skipping",
                file=sys.stderr,
            )
            continue

        # Observed counts (also extracts sequence tensor X for predictions)
        if total_reads == 0:
            print(f"WARNING: {exp_id}: total_reads=0, skipping", file=sys.stderr)
            continue
        base_dir = args.model_dir or (REPO_ROOT / "models" / (args.model or ""))
        model_dir = base_dir / exp_id

        if args.held_out_folds and args.model is not None:
            if not model_dir.exists():
                print(
                    f"WARNING: {exp_id}: model dir not found ({model_dir}), skipping",
                    file=sys.stderr,
                )
                continue
            obs, pred = extract_heldout_counts(
                union_peaks, pl_path, mn_path, total_reads, model_dir, exp_id,
                fold_assignment, args.batch_size, args.device,
            )
            if not len(obs):
                print(f"WARNING: {exp_id}: no peaks survived, skipping", file=sys.stderr)
                continue
            obs_rows[exp_id] = obs
            biosample_map[exp_id] = biosample_safe
            if pred is not None:
                pred_rows[exp_id] = pred
            else:
                print(
                    f"WARNING: {exp_id}: missing a fold model, predictions skipped",
                    file=sys.stderr,
                )
            continue

        X, obs = extract_observed_counts(union_peaks, pl_path, mn_path, total_reads)
        obs_rows[exp_id] = obs
        biosample_map[exp_id] = biosample_safe

        # Predicted counts — reuse X from observed extraction (same filtering)
        if args.model is not None:
            if not model_dir.exists():
                print(
                    f"WARNING: {exp_id}: model dir not found ({model_dir}), skipping predictions",
                    file=sys.stderr,
                )
            else:
                pred = extract_predicted_counts(
                    X, model_dir, exp_id, args.batch_size, args.device
                )
                if pred is not None:
                    pred_rows[exp_id] = pred
                else:
                    print(
                        f"WARNING: {exp_id}: no fold models found in {model_dir}",
                        file=sys.stderr,
                    )

    if not obs_rows:
        print("ERROR: no experiments produced observed counts", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    obs_df = pd.DataFrame(obs_rows).T  # (n_experiments, n_peaks)
    obs_df.index.name = "experiment"
    obs_path = args.out_dir / "observed_counts.tsv"
    obs_df.to_csv(obs_path, sep="\t")
    print(f"Observed counts: {obs_df.shape} → {obs_path}", file=sys.stderr)

    pred_df = None
    if pred_rows:
        # Only include experiments that have both observed and predicted
        common = [e for e in obs_df.index if e in pred_rows]
        pred_df = pd.DataFrame({e: pred_rows[e] for e in common}).T
        pred_df.index.name = "experiment"
        pred_path = args.out_dir / "predicted_counts.tsv"
        pred_df.to_csv(pred_path, sep="\t")
        print(f"Predicted counts: {pred_df.shape} → {pred_path}", file=sys.stderr)

    if len(obs_df) < 2:
        print("WARNING: fewer than 2 experiments — skipping correlation plot", file=sys.stderr)
        return

    plot_clustermaps(
        obs_df,
        pred_df,
        biosample_map,
        args.out_dir / "count_correlation.png",
    )


if __name__ == "__main__":
    main()
