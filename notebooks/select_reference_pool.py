"""Select low-activity shuffled references for a BPNet locus."""

import argparse
import gc
import gzip
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

cache_root = Path(os.environ.get("SCRATCH", "/tmp")) / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bpnetlite.bpnet import CountWrapper, ProfileWrapper
from huggingface_hub import hf_hub_download
from pyfaidx import Fasta
from tangermeme.ersatz import dinucleotide_shuffle, shuffle
from tangermeme.io import extract_loci
from tangermeme.plot import plot_logo
from tangermeme.predict import predict

from src.bpnet.attribute.deeplift import deep_lift_shap
from src.bpnet.attribute.locus_diagnostics import genomic_offsets, profile_summaries

MODEL_REPO_ID = "adamyhe/procap-atlas"
REFERENCE_FASTA_URL = "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
IN_WINDOW = 2114
OUT_WINDOW = 1000


def parse_point(region):
    chrom, position = region.replace(",", "").split(":", 1)
    return chrom, int(position) - 1


def parse_interval(region):
    chrom, interval = region.replace(",", "").split(":", 1)
    start, end = [int(value) for value in interval.split("-", 1)]
    if end < start:
        raise ValueError("Interval end must not precede its start")
    return chrom, start - 1, end


def download_first(repo_id, repo_type, filenames):
    last_error = None
    for filename in filenames:
        try:
            return Path(
                hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename)
            )
        except Exception as error:
            last_error = error
    raise FileNotFoundError(f"Could not find any of {filenames}") from last_error


def resolve_model_paths(experiment, model_dir, folds):
    if model_dir is not None:
        paths = [model_dir / f"{experiment}.fold{fold}.torch" for fold in range(folds)]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing model checkpoints: {missing}")
        return paths
    patterns = [
        "{experiment}/{experiment}.fold{fold}.torch",
        "bpnet/{experiment}/{experiment}.fold{fold}.torch",
        "models/bpnet/{experiment}/{experiment}.fold{fold}.torch",
    ]
    return [
        download_first(
            MODEL_REPO_ID,
            "model",
            [pattern.format(experiment=experiment, fold=fold) for pattern in patterns],
        )
        for fold in range(folds)
    ]


def resolve_fasta(fasta, work_dir):
    if fasta is not None:
        fasta = fasta.expanduser()
        if not fasta.exists():
            raise FileNotFoundError(fasta)
        if not Path(str(fasta) + ".fai").exists():
            Fasta(str(fasta))
        return fasta

    reference_dir = work_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    fasta = reference_dir / "hg38.fa"
    compressed = reference_dir / "hg38.fa.gz"
    if not fasta.exists():
        if not compressed.exists():
            urllib.request.urlretrieve(REFERENCE_FASTA_URL, compressed)
        with gzip.open(compressed, "rb") as source, open(fasta, "wb") as target:
            shutil.copyfileobj(source, target)
    if not Path(str(fasta) + ".fai").exists():
        Fasta(str(fasta))
    return fasta


def robust_penalty(values, high_is_bad=True):
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(values)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    z = (values - median) / scale
    if not high_is_bad:
        z = -z
    return np.maximum(z, 0)


def score_candidates(metrics):
    parts = [
        robust_penalty(np.log1p(metrics["counts"])),
        robust_penalty(metrics["max_20bp"]),
        robust_penalty(metrics["max_5bp"]),
        robust_penalty(metrics["profile_score"]),
        robust_penalty(metrics["strand_imbalance"]),
        robust_penalty(metrics["profile_entropy"], high_is_bad=False),
    ]
    return np.vstack(parts).sum(axis=0)


def select_diverse_indices(seed_frame, selected_per_seed, min_hamming_fraction, bases):
    ranked = seed_frame.sort_values("selection_score", kind="stable").candidate_index
    selected = []
    for index in ranked:
        index = int(index)
        if all(
            np.mean(bases[index] != bases[chosen]) >= min_hamming_fraction
            for chosen in selected
        ):
            selected.append(index)
            if len(selected) == selected_per_seed:
                break
    if len(selected) < selected_per_seed:
        seen = set(selected)
        selected.extend(int(index) for index in ranked if int(index) not in seen)
    return selected[:selected_per_seed]


def shuffled_candidates(X, n, random_state, mode):
    if mode == "dinucleotide":
        return dinucleotide_shuffle(X.cpu(), n=n, random_state=random_state)[0]
    if mode == "mononucleotide":
        return shuffle(X.cpu(), n=n, random_state=random_state)[0]
    raise ValueError("mode must be 'dinucleotide' or 'mononucleotide'")


def plot_metric_distributions(averaged, selected, output_dir, plot_format):
    metrics = [
        ("counts", "Predicted counts"),
        ("max_5bp", "Maximum 5 bp signal"),
        ("max_20bp", "Maximum 20 bp signal"),
        ("profile_score", "Profile concentration"),
        ("strand_imbalance", "Strand imbalance"),
        ("profile_entropy", "Profile entropy"),
        ("selection_score", "Selection score"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    selected_indices = set(selected["candidate_index"])
    all_mask = ~averaged["candidate_index"].isin(selected_indices)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        ax.hist(
            averaged.loc[all_mask, metric],
            bins=50,
            color="#4C72B0",
            alpha=0.45,
            label="not selected",
        )
        ax.hist(
            selected[metric],
            bins=25,
            color="#C44E52",
            alpha=0.7,
            label="selected",
        )
        ax.set_title(title)
        ax.set_ylabel("Candidates")
    axes[0, 0].legend(frameon=False)
    axes[-1, -1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"metric_distributions.{plot_format}", dpi=180)
    plt.close(fig)


def plot_ranked_metrics(averaged, selected, seeds, output_dir, plot_format):
    metrics = [
        ("selection_score", "Selection score", True),
        ("counts", "Predicted counts", True),
        ("max_5bp", "Maximum 5 bp signal", True),
        ("max_20bp", "Maximum 20 bp signal", True),
        ("profile_entropy", "Profile entropy", False),
    ]
    selected_indices = set(selected["candidate_index"])
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 3.8))
    for ax, (metric, title, ascending) in zip(axes, metrics):
        for seed in seeds:
            seed_frame = averaged[averaged["seed"] == seed].sort_values(
                metric, ascending=ascending, kind="stable"
            )
            values = seed_frame[metric].to_numpy()
            x = np.arange(1, len(values) + 1)
            ax.plot(x, values, linewidth=1.3, label=f"seed {seed}")
            selected_positions = np.flatnonzero(
                seed_frame["candidate_index"].isin(selected_indices).to_numpy()
            )
            ax.scatter(
                x[selected_positions],
                values[selected_positions],
                s=14,
                color="black",
                alpha=0.75,
                zorder=3,
            )
        ax.set_title(title)
        ax.set_xlabel("Candidate rank")
    axes[0].set_ylabel("Metric value")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Per-seed candidate rankings; black points are selected references")
    fig.tight_layout()
    fig.savefig(output_dir / f"ranked_reference_metrics.{plot_format}", dpi=180)
    plt.close(fig)


def plot_timing_summary(timings, output_dir, plot_format):
    rows = []
    for key, value in timings.items():
        if key == "total_seconds":
            continue
        if isinstance(value, dict):
            for fold, seconds in value.items():
                rows.append((f"fold {fold} scoring", seconds))
        else:
            rows.append((key.replace("_seconds", "").replace("_", " "), value))
    labels, seconds = zip(*rows)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, seconds, color="#55A868")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds")
    total = timings.get("total_seconds", sum(seconds))
    ax.set_title(f"Reference-pool runtime breakdown ({total:.2f}s total)")
    fig.tight_layout()
    fig.savefig(output_dir / f"timing_summary.{plot_format}", dpi=180)
    plt.close(fig)


def plot_deeplift_logos(
    attributions, seeds, logo_start, logo_end, output_dir, plot_format
):
    heads = [("profile", 0), ("count", 1)]
    for head, index in heads:
        fig, axes = plt.subplots(2, 1, figsize=(20, 5), squeeze=False, sharex=True)
        plot_logo(torch.tensor(attributions[index].mean(axis=(0, 1))), ax=axes[0, 0])
        axes[0, 0].set_title(f"Selected-reference {head} DeepLIFT/SHAP")
        mean_by_seed = attributions[index].mean(axis=0)
        seed_mean = mean_by_seed.mean(axis=0)
        seed_sd = mean_by_seed.std(axis=0)
        x = np.arange(seed_mean.shape[-1])
        axes[1, 0].plot(x, seed_mean.sum(axis=0), color="#4C72B0")
        axes[1, 0].fill_between(
            x,
            seed_mean.sum(axis=0) - seed_sd.sum(axis=0),
            seed_mean.sum(axis=0) + seed_sd.sum(axis=0),
            color="#4C72B0",
            alpha=0.25,
        )
        axes[1, 0].set_title("Mean +/- SD across selected seed banks")
        axes[1, 0].set_ylabel("Summed attribution")
        axes[1, 0].set_xlabel(f"Input positions {logo_start}-{logo_end}")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"selected_{head}_deeplift_logo.{plot_format}", dpi=180
        )
        plt.close(fig)

    fig, axes = plt.subplots(
        len(seeds) * 2,
        1,
        figsize=(20, max(2.0 * len(seeds) * 2, 4)),
        squeeze=False,
        sharex=True,
    )
    for head_index, head in enumerate(["profile", "count"]):
        for seed_index, seed in enumerate(seeds):
            ax = axes[head_index * len(seeds) + seed_index, 0]
            plot_logo(
                torch.tensor(attributions[head_index, :, seed_index].mean(axis=0)),
                ax=ax,
            )
            ax.set_title(f"{head}, seed {seed}")
    axes[-1, 0].set_xlabel(f"Input positions {logo_start}-{logo_end}")
    fig.tight_layout()
    fig.savefig(output_dir / f"selected_seed_deeplift_logos.{plot_format}", dpi=180)
    plt.close(fig)


def plot_frequency_reference_deeplift_logos(
    attributions, logo_start, logo_end, output_dir, plot_format
):
    for head, index in [("profile", 0), ("count", 1)]:
        matrix = attributions[index].mean(axis=0)
        fig, axes = plt.subplots(2, 1, figsize=(20, 5), squeeze=False, sharex=True)
        plot_logo(torch.tensor(matrix), ax=axes[0, 0])
        axes[0, 0].set_title(
            f"Observed-frequency-reference {head} DeepLIFT/SHAP"
        )
        x = np.arange(matrix.shape[-1])
        axes[1, 0].plot(x, matrix.sum(axis=0), color="#4C72B0")
        axes[1, 0].axhline(0, color="black", linewidth=0.7)
        axes[1, 0].set_title("Summed attribution track")
        axes[1, 0].set_ylabel("Summed attribution")
        axes[1, 0].set_xlabel(f"Input positions {logo_start}-{logo_end}")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"frequency_reference_{head}_deeplift_logo.{plot_format}",
            dpi=180,
        )
        plt.close(fig)


def observed_frequency_reference(X):
    frequencies = X[0].float().mean(dim=-1)
    reference = frequencies[:, None].expand_as(X[0]).clone()
    return reference.unsqueeze(0)


def selected_reference_banks(candidates, selected, seeds):
    return {
        seed: candidates[
            selected.loc[selected["seed"] == seed, "candidate_index"]
            .astype(int)
            .tolist()
        ]
        for seed in seeds
    }


def selected_deeplift_attributions(
    model_paths, X, reference_banks, seeds, logo_offsets, batch_size, device
):
    width = logo_offsets[1] - logo_offsets[0]
    attributions = np.empty(
        (2, len(model_paths), len(seeds), 4, width), dtype=np.float32
    )
    frequency_attributions = np.empty(
        (2, len(model_paths), 4, width), dtype=np.float32
    )
    frequency_reference = observed_frequency_reference(X)
    frequency_reference_bank = frequency_reference[:, None]
    if frequency_reference_bank.shape[:2] != (X.shape[0], 1):
        raise ValueError("Observed-frequency baseline must contain one reference per input")
    for fold, model_path in enumerate(model_paths):
        print(f"DeepLIFT fold {fold + 1}/{len(model_paths)}: {model_path.name}")
        model = torch.load(model_path, map_location="cpu", weights_only=False).eval()
        for head_index, wrapper in enumerate(
            [ProfileWrapper(model), CountWrapper(model)]
        ):
            for seed_index, seed in enumerate(seeds):
                attr = deep_lift_shap(
                    model=wrapper,
                    X=X,
                    references=reference_banks[seed][None],
                    batch_size=batch_size,
                    hypothetical=True,
                    warning_threshold=0.01,
                    device=device,
                )
                attributions[head_index, fold, seed_index] = (
                    (attr * X)
                    .detach()
                    .cpu()
                    .numpy()[0, :, logo_offsets[0] : logo_offsets[1]]
                )
            frequency_attr = deep_lift_shap(
                model=wrapper,
                X=X,
                references=frequency_reference_bank,
                batch_size=batch_size,
                hypothetical=True,
                warning_threshold=0.01,
                device=device,
            )
            frequency_attributions[head_index, fold] = (
                (frequency_attr * X)
                .detach()
                .cpu()
                .numpy()[0, :, logo_offsets[0] : logo_offsets[1]]
            )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return attributions, frequency_attributions, frequency_reference


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a large shuffled candidate pool for one BPNet "
            "locus, score predicted reference activity, and select low-activity "
            "references with timing reports."
        )
    )
    parser.add_argument("--experiment", default="ENCSR342WAR")
    parser.add_argument("--point-region", default="chr2:181680717")
    parser.add_argument(
        "--logo-region",
        help=(
            "genomic interval for DeepLIFT logos; defaults to a centered "
            "--logo-window around --point-region"
        ),
    )
    parser.add_argument("--candidate-seeds", default="0,1,2,3,4,6,7,42,47,100")
    parser.add_argument(
        "--candidate-mode",
        choices=("dinucleotide", "mononucleotide"),
        default="dinucleotide",
        help="shuffle type used to generate candidate references",
    )
    parser.add_argument("--candidates-per-seed", type=int, default=500)
    parser.add_argument("--selected-per-seed", type=int, default=20)
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("SCRATCH", ".cache")) / "procap_atlas_locus_viewer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: plots/bpnet/reference_pool/<experiment>/<point>/<candidate-mode>",
    )
    parser.add_argument("--min-hamming-fraction", type=float, default=0.1)
    parser.add_argument("--plot-format", choices=("png", "pdf", "svg"), default="pdf")
    parser.add_argument("--logo-window", type=int, default=200)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-deeplift", action="store_true")
    args = parser.parse_args()

    start_total = time.perf_counter()
    timings = {}
    seeds = [int(seed) for seed in args.candidate_seeds.split(",") if seed]
    if not seeds:
        raise ValueError("--candidate-seeds must contain at least one integer")
    if args.candidates_per_seed < args.selected_per_seed:
        raise ValueError("--candidates-per-seed must be at least --selected-per-seed")
    if args.folds < 1 or args.batch_size < 1:
        raise ValueError("--folds and --batch-size must be positive")
    if not 0 <= args.min_hamming_fraction <= 1:
        raise ValueError("--min-hamming-fraction must be between zero and one")

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else Path("plots/bpnet/reference_pool")
        / args.experiment
        / args.point_region.replace(":", "_").replace(",", "")
        / args.candidate_mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir.expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(
        f"Candidates: {len(seeds)} seeds x {args.candidates_per_seed} "
        f"= {len(seeds) * args.candidates_per_seed}"
    )
    print(f"Candidate mode: {args.candidate_mode}")

    stage = time.perf_counter()
    fasta = resolve_fasta(args.fasta, work_dir)
    model_paths = resolve_model_paths(args.experiment, args.model_dir, args.folds)
    timings["setup_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    chrom, center = parse_point(args.point_region)
    if args.logo_region is None:
        half = args.logo_window // 2
        logo_start = center - half
        logo_end = logo_start + args.logo_window
        logo_chrom = chrom
    else:
        logo_chrom, logo_start, logo_end = parse_interval(args.logo_region)
    if logo_chrom != chrom:
        raise ValueError(
            "--point-region and --logo-region must use the same chromosome"
        )
    logo_offsets = genomic_offsets(center, logo_start, logo_end, IN_WINDOW)
    loci = pd.DataFrame({"chrom": [chrom], "start": [center], "end": [center + 1]})
    X = extract_loci(loci, sequences=str(fasta), in_window=IN_WINDOW, ignore=["N", "n"])
    if len(X) != 1:
        raise ValueError("The requested locus could not be extracted")
    X = X.float()
    timings["extract_locus_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    candidate_blocks = []
    candidate_rows = []
    for seed in seeds:
        block = shuffled_candidates(
            X, args.candidates_per_seed, seed, args.candidate_mode
        )
        offset = sum(len(previous) for previous in candidate_blocks)
        candidate_blocks.append(block)
        candidate_rows.extend(
            {
                "candidate_index": offset + reference,
                "seed": seed,
                "seed_rank": reference,
                "candidate_mode": args.candidate_mode,
            }
            for reference in range(args.candidates_per_seed)
        )
    candidates = torch.cat(candidate_blocks).float()
    candidate_index = pd.DataFrame(candidate_rows)
    bases = candidates.argmax(dim=1).numpy()
    timings["candidate_generation_seconds"] = time.perf_counter() - stage

    fold_tables = []
    timings["fold_scoring_seconds"] = {}
    for fold, model_path in enumerate(model_paths):
        stage = time.perf_counter()
        print(f"Scoring fold {fold + 1}/{len(model_paths)}: {model_path.name}")
        model = torch.load(model_path, map_location="cpu", weights_only=False).eval()
        profile_logits, log_counts = predict(
            model=model,
            X=candidates,
            batch_size=args.batch_size,
            device=device,
        )
        profile_logits = torch.as_tensor(profile_logits)
        original_shape = profile_logits.shape
        centered = profile_logits.reshape(len(candidates), -1)
        centered = centered - centered.mean(dim=1, keepdim=True)
        probabilities = torch.softmax(centered, dim=1).reshape(original_shape)
        counts = torch.exp(
            torch.as_tensor(log_counts).reshape(len(candidates), -1)
        ).sum(dim=1)
        count_scaled = probabilities.reshape(len(candidates), -1) * counts[:, None]
        count_scaled = count_scaled.reshape(original_shape)
        summaries = profile_summaries(
            probabilities.detach().cpu().numpy(),
            count_scaled.detach().cpu().numpy(),
            windows=(1, 5, 20),
        )
        plus_mass = probabilities[:, 0].reshape(len(candidates), -1).sum(dim=1)
        minus_mass = probabilities[:, 1].reshape(len(candidates), -1).sum(dim=1)
        table = candidate_index.copy()
        table["fold"] = fold
        table["counts"] = counts.detach().cpu().numpy()
        table["profile_score"] = (
            (centered * torch.softmax(centered, dim=1))
            .sum(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        table["strand_imbalance"] = (
            torch.abs(plus_mass - minus_mass).detach().cpu().numpy()
        )
        for key, value in summaries.items():
            table[key] = value
        fold_tables.append(table)
        del (
            model,
            profile_logits,
            log_counts,
            centered,
            probabilities,
            counts,
            count_scaled,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        timings["fold_scoring_seconds"][str(fold)] = time.perf_counter() - stage

    stage = time.perf_counter()
    fold_metrics = pd.concat(fold_tables, ignore_index=True)
    averaged = (
        fold_metrics.drop(columns=["fold"])
        .groupby(
            ["candidate_index", "seed", "seed_rank", "candidate_mode"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )
    averaged["selection_score"] = score_candidates(averaged)
    selected_indices = []
    for seed in seeds:
        seed_frame = averaged[averaged["seed"] == seed]
        selected_indices.extend(
            select_diverse_indices(
                seed_frame,
                args.selected_per_seed,
                args.min_hamming_fraction,
                bases,
            )
        )
    selected_indices = np.asarray(selected_indices, dtype=int)
    selected = averaged[averaged["candidate_index"].isin(selected_indices)].copy()
    selected["selected_order"] = selected["candidate_index"].map(
        {index: order for order, index in enumerate(selected_indices)}
    )
    selected = selected.sort_values("selected_order")
    selected_refs = candidates[selected_indices].numpy().astype(np.int8)
    timings["selection_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    fold_metrics.to_csv(
        output_dir / "candidate_fold_metrics.tsv", sep="\t", index=False
    )
    averaged.to_csv(output_dir / "candidate_mean_metrics.tsv", sep="\t", index=False)
    selected.to_csv(
        output_dir / "selected_reference_metrics.tsv", sep="\t", index=False
    )
    np.savez_compressed(
        output_dir / "selected_references.npz",
        references=selected_refs,
        selected_indices=selected_indices,
        seeds=np.asarray(seeds, dtype=int),
        selected_per_seed=np.asarray(args.selected_per_seed, dtype=int),
        candidate_mode=np.asarray(args.candidate_mode),
        point_region=np.asarray(args.point_region),
        experiment=np.asarray(args.experiment),
    )
    timings["write_outputs_seconds"] = time.perf_counter() - stage

    if not args.no_plots:
        stage = time.perf_counter()
        plot_metric_distributions(averaged, selected, output_dir, args.plot_format)
        plot_ranked_metrics(averaged, selected, seeds, output_dir, args.plot_format)
        timings["plot_seconds"] = time.perf_counter() - stage

    if not args.no_deeplift and not args.no_plots:
        stage = time.perf_counter()
        banks = selected_reference_banks(candidates, selected, seeds)
        (
            selected_attributions,
            frequency_attributions,
            frequency_reference,
        ) = selected_deeplift_attributions(
            model_paths, X, banks, seeds, logo_offsets, args.batch_size, device
        )
        np.savez_compressed(
            output_dir / "selected_deeplift_attributions.npz",
            attributions=selected_attributions,
            frequency_reference_attributions=frequency_attributions,
            frequency_reference=frequency_reference.detach().cpu().numpy(),
            frequency_reference_probabilities=frequency_reference[0, :, 0]
            .detach()
            .cpu()
            .numpy(),
            n_frequency_references=np.asarray(1, dtype=int),
            seeds=np.asarray(seeds, dtype=int),
            logo_offsets=np.asarray(logo_offsets, dtype=int),
            logo_region=np.asarray(f"{logo_chrom}:{logo_start + 1}-{logo_end}"),
        )
        plot_deeplift_logos(
            selected_attributions,
            seeds,
            logo_start,
            logo_end,
            output_dir,
            args.plot_format,
        )
        plot_frequency_reference_deeplift_logos(
            frequency_attributions,
            logo_start,
            logo_end,
            output_dir,
            args.plot_format,
        )
        timings["deeplift_seconds"] = time.perf_counter() - stage

    timings["total_seconds"] = time.perf_counter() - start_total
    if not args.no_plots:
        plot_timing_summary(timings, output_dir, args.plot_format)

    summary = {
        "experiment": args.experiment,
        "point_region": args.point_region,
        "device": device,
        "folds": args.folds,
        "batch_size": args.batch_size,
        "candidate_seeds": seeds,
        "candidate_mode": args.candidate_mode,
        "logo_region": f"{logo_chrom}:{logo_start + 1}-{logo_end}",
        "logo_offsets": list(logo_offsets),
        "n_frequency_references": 1,
        "candidates_per_seed": args.candidates_per_seed,
        "selected_per_seed": args.selected_per_seed,
        "n_candidates": int(len(candidates)),
        "n_selected": int(len(selected_indices)),
        "min_hamming_fraction": args.min_hamming_fraction,
        "fasta": str(fasta),
        "model_paths": [str(path) for path in model_paths],
        "timings": timings,
    }
    with open(output_dir / "selection_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTiming summary")
    for key, value in timings.items():
        if isinstance(value, dict):
            total = sum(value.values())
            print(f"  {key}: {total:.2f}s total ({value})")
        else:
            print(f"  {key}: {value:.2f}s")
    print(f"\nWrote {output_dir}")


if __name__ == "__main__":
    main()
