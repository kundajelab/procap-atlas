"""Run the BPNet locus diagnostics notebook workflow as a batch-friendly CLI."""

import argparse
import gc
import gzip
import os
import shutil
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
import pybigtools
import seaborn as sns
import torch
import yaml
from bpnetlite.attribute import deep_lift_shap
from bpnetlite.bpnet import CountWrapper, ProfileWrapper
from huggingface_hub import hf_hub_download
from pyfaidx import Fasta
from tangermeme.io import extract_loci
from tangermeme.plot import plot_logo
from tangermeme.predict import predict

from src.bpnet.attribute.locus_diagnostics import (
    CACHE_SCHEMA_VERSION,
    StrandProfileWrapper,
    as_numpy,
    diagnostic_cache_path,
    dinucleotide_frequencies,
    fold_consensus,
    genomic_offsets,
    gradient_x_input,
    per_reference_attributions,
    point_ism,
    predicted_activity,
    reference_banks,
    reference_weight_schemes,
    reverse_complement_matrix,
    reverse_complement_peak_coordinates,
    reverse_complement_tracks,
    rolling_profile_maxima,
    weighted_strand_attributions,
    window_ism,
    window_scores_to_positions,
)

sns.set_style("whitegrid")

MODEL_REPO_ID = "adamyhe/procap-atlas"
TRACK_REPO_ID = "adamyhe/procap-atlas-tracks"
METADATA_REPO_ID = "adamyhe/procap-atlas-metadata"
REFERENCE_FASTA_URL = "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
BLACKLIST_URL = (
    "https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz"
)

parser = argparse.ArgumentParser(
    description="Generate BPNet locus predictions and attribution diagnostics."
)
parser.add_argument("--experiment", default="ENCSR342WAR")
parser.add_argument("--point-region", default="chr2:181680717")
parser.add_argument("--view-region", default="chr2:181680467-181681166")
parser.add_argument("--logo-region", default="chr2:181680467-181681167")
parser.add_argument("--reverse-complement", action="store_true")
parser.add_argument("--folds", type=int, default=7)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--reference-seeds", default="0,1,2,3,4")
parser.add_argument("--references", type=int, default=20)
parser.add_argument("--weighted-reference-seed", type=int)
parser.add_argument("--reference-weight-temperature", type=float, default=1.0)
parser.add_argument("--window-ism-width", type=int, default=10)
parser.add_argument("--window-ism-stride", type=int, default=2)
parser.add_argument("--window-ism-replacements", type=int, default=20)
parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
parser.add_argument("--force", action="store_true")
parser.add_argument(
    "--work-dir",
    type=Path,
    default=Path(os.environ.get("SCRATCH", ".cache")) / "procap_atlas_locus_viewer",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    help="figure directory (default: plots/bpnet/locus_diagnostics/<experiment>/<point>)",
)
parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
parser.add_argument("--dpi", type=int, default=180)
args = parser.parse_args()

WORK_DIR = args.work_dir.expanduser()
CACHE_DIR = WORK_DIR / "diagnostics"
EXP_ID = args.experiment
POINT_REGION = args.point_region
VIEW_REGION = args.view_region
LOGO_REGION = args.logo_region
REVERSE_COMPLEMENT = args.reverse_complement
N_FOLDS = args.folds
IN_WINDOW = 2114
OUT_WINDOW = 1000
BATCH_SIZE = args.batch_size
REFERENCE_SEEDS = [int(seed) for seed in args.reference_seeds.split(",")]
N_REFERENCES = args.references
WEIGHTED_REFERENCE_SEED = (
    REFERENCE_SEEDS[0]
    if args.weighted_reference_seed is None
    else args.weighted_reference_seed
)
REFERENCE_WEIGHT_TEMPERATURE = args.reference_weight_temperature
WINDOW_ISM_WIDTH = args.window_ism_width
WINDOW_ISM_STRIDE = args.window_ism_stride
WINDOW_ISM_REPLACEMENTS = args.window_ism_replacements
FORCE_DIAGNOSTICS = args.force
DEVICE = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
if DEVICE == "auto":
    DEVICE = "cpu"
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("--device cuda requested, but CUDA is unavailable")
if not REFERENCE_SEEDS:
    raise ValueError("--reference-seeds must contain at least one integer")
if N_FOLDS < 1 or N_REFERENCES < 1 or BATCH_SIZE < 1:
    raise ValueError("--folds, --references, and --batch-size must be positive")
if REFERENCE_WEIGHT_TEMPERATURE <= 0:
    raise ValueError("--reference-weight-temperature must be positive")

OUTPUT_DIR = (
    args.output_dir.expanduser()
    if args.output_dir is not None
    else Path("plots/bpnet/locus_diagnostics")
    / EXP_ID
    / POINT_REGION.replace(":", "_").replace(",", "")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Work directory: {WORK_DIR}")
print(f"Figure directory: {OUTPUT_DIR}")


def download_first(repo_id, repo_type, filenames):
    last_error = None
    for filename in filenames:
        try:
            return Path(
                hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename)
            )
        except Exception as error:
            last_error = error
    raise FileNotFoundError(
        f"Could not find any of {filenames} in {repo_id}"
    ) from last_error


def parse_point(region):
    chrom, position = region.replace(",", "").split(":", 1)
    return chrom, int(position) - 1


def parse_interval(region):
    chrom, interval = region.replace(",", "").split(":", 1)
    start, end = [int(value) for value in interval.split("-", 1)]
    if end < start:
        raise ValueError("Interval end must not precede its start")
    return chrom, start - 1, end


def download_reference():
    reference_dir = WORK_DIR / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    fasta = reference_dir / "hg38.fa"
    compressed = reference_dir / "hg38.fa.gz"
    blacklist = reference_dir / "hg38.blacklist.bed.gz"
    if not fasta.exists():
        if not compressed.exists():
            urllib.request.urlretrieve(REFERENCE_FASTA_URL, compressed)
        with gzip.open(compressed, "rb") as source, open(fasta, "wb") as target:
            shutil.copyfileobj(source, target)
    if not Path(str(fasta) + ".fai").exists():
        Fasta(str(fasta))
    if not blacklist.exists():
        urllib.request.urlretrieve(BLACKLIST_URL, blacklist)
    return fasta, blacklist


def download_model_paths(exp_id, n_folds=N_FOLDS):
    patterns = [
        "{exp_id}/{exp_id}.fold{fold}.torch",
        "bpnet/{exp_id}/{exp_id}.fold{fold}.torch",
        "models/bpnet/{exp_id}/{exp_id}.fold{fold}.torch",
    ]
    return [
        download_first(
            MODEL_REPO_ID,
            "model",
            [p.format(exp_id=exp_id, fold=fold) for p in patterns],
        )
        for fold in range(n_folds)
    ]


def setup_experiment(exp_id=EXP_ID):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config_path = download_first(
        METADATA_REPO_ID,
        "dataset",
        ["experiment_config.yaml", "configs/experiment_config.yaml"],
    )
    with open(config_path) as handle:
        config = yaml.safe_load(handle)["experiments"]
    if exp_id not in config:
        raise KeyError(f"{exp_id} is absent from experiment_config.yaml")
    fasta, blacklist = download_reference()
    observed = {
        "plus": download_first(
            TRACK_REPO_ID, "dataset", [f"observed/{exp_id}_pl.bigWig"]
        ),
        "minus": download_first(
            TRACK_REPO_ID, "dataset", [f"observed/{exp_id}_mn.bigWig"]
        ),
    }
    return {
        "exp_id": exp_id,
        "config": config[exp_id],
        "fasta": fasta,
        "blacklist": blacklist,
        "observed": observed,
        "model_paths": download_model_paths(exp_id),
    }


def locus_input(resources):
    chrom, center = parse_point(POINT_REGION)
    loci = pd.DataFrame({"chrom": [chrom], "start": [center], "end": [center + 1]})
    X = extract_loci(
        loci, sequences=str(resources["fasta"]), in_window=IN_WINDOW, ignore=["N", "n"]
    )
    if len(X) != 1:
        raise ValueError("The requested locus could not be extracted")
    return chrom, center, X.float()


resources = setup_experiment()
chrom, center, X = locus_input(resources)
logo_chrom, logo_start, logo_end = parse_interval(LOGO_REGION)
if logo_chrom != chrom:
    raise ValueError("POINT_REGION and LOGO_REGION must use the same chromosome")
logo_offsets = genomic_offsets(center, logo_start, logo_end, IN_WINDOW)
reference_sequences = reference_banks(X, REFERENCE_SEEDS, N_REFERENCES)
print(resources["config"].get("biosample", EXP_ID), X.shape, logo_offsets)


def run_diagnostics(force=False):
    parameters = {
        "experiment": EXP_ID,
        "cache_schema": CACHE_SCHEMA_VERSION,
        "point": POINT_REGION,
        "logo": LOGO_REGION,
        "folds": N_FOLDS,
        "seeds": REFERENCE_SEEDS,
        "references": N_REFERENCES,
        "weighted_reference_seed": WEIGHTED_REFERENCE_SEED,
        "reference_weight_temperature": REFERENCE_WEIGHT_TEMPERATURE,
        "window": WINDOW_ISM_WIDTH,
        "stride": WINDOW_ISM_STRIDE,
        "replacements": WINDOW_ISM_REPLACEMENTS,
        "models": [(p.name, p.stat().st_size) for p in resources["model_paths"]],
    }
    cache_path = diagnostic_cache_path(CACHE_DIR, EXP_ID, parameters)
    if cache_path.exists() and not force:
        print(f"Loading {cache_path}")
        return dict(np.load(cache_path, allow_pickle=False)), cache_path

    heads = ("profile", "count")
    width = logo_end - logo_start
    deep_lift = np.empty((2, N_FOLDS, len(REFERENCE_SEEDS), 4, width), dtype=np.float32)
    strand_profile_deep_lift = np.empty(
        (2, N_FOLDS, len(REFERENCE_SEEDS), 4, width), dtype=np.float32
    )
    gradients = np.empty((2, N_FOLDS, 4, width), dtype=np.float32)
    point_scores = np.empty((2, N_FOLDS, 4, width), dtype=np.float32)
    window_positions = np.arange(
        logo_offsets[0], logo_offsets[1] - WINDOW_ISM_WIDTH + 1, WINDOW_ISM_STRIDE
    )
    window_scores = np.empty((2, N_FOLDS, len(window_positions)), dtype=np.float32)
    reference_centered_logits = np.empty(
        (N_FOLDS, len(REFERENCE_SEEDS), N_REFERENCES, 2, OUT_WINDOW), dtype=np.float16
    )
    reference_probabilities = np.empty_like(reference_centered_logits)
    reference_count_scaled = np.empty_like(reference_centered_logits)
    weighted_seed_probabilities = np.empty(
        (N_FOLDS, N_REFERENCES, 2, OUT_WINDOW), dtype=np.float32
    )
    weighted_seed_counts = np.empty((N_FOLDS, N_REFERENCES), dtype=np.float32)
    per_reference_strand_deep_lift = np.empty(
        (2, N_FOLDS, N_REFERENCES, 4, IN_WINDOW), dtype=np.float32
    )
    strand_profile_deltas = np.empty((2, N_FOLDS, N_REFERENCES), dtype=np.float32)
    genomic_centered_logits = np.empty((N_FOLDS, 2, OUT_WINDOW), dtype=np.float16)
    genomic_probabilities = np.empty_like(genomic_centered_logits)
    genomic_count_scaled = np.empty_like(genomic_centered_logits)
    activity_rows = []
    if WEIGHTED_REFERENCE_SEED not in REFERENCE_SEEDS:
        raise ValueError("WEIGHTED_REFERENCE_SEED must be present in REFERENCE_SEEDS")

    for fold, model_path in enumerate(resources["model_paths"]):
        print(f"Fold {fold + 1}/{N_FOLDS}: {model_path.name}")
        model = torch.load(model_path, map_location="cpu", weights_only=False).eval()
        for seed_index, seed in enumerate(REFERENCE_SEEDS):
            activity = predicted_activity(
                model, X, reference_sequences[seed], BATCH_SIZE, DEVICE
            )
            reference_centered_logits[fold, seed_index] = activity[
                "reference_centered_logits"
            ].astype(np.float16)
            reference_probabilities[fold, seed_index] = activity[
                "reference_probabilities"
            ].astype(np.float16)
            reference_count_scaled[fold, seed_index] = activity[
                "reference_count_scaled"
            ].astype(np.float16)
            if seed == WEIGHTED_REFERENCE_SEED:
                weighted_seed_probabilities[fold] = activity["reference_probabilities"]
                weighted_seed_counts[fold] = activity["reference_counts"]
            if seed_index == 0:
                genomic_centered_logits[fold] = activity["genomic_centered_logits"][
                    0
                ].astype(np.float16)
                genomic_probabilities[fold] = activity["genomic_probabilities"][
                    0
                ].astype(np.float16)
                genomic_count_scaled[fold] = activity["genomic_count_scaled"][0].astype(
                    np.float16
                )
            for reference in range(N_REFERENCES):
                summaries = activity["reference_summaries"]
                activity_rows.append(
                    [
                        fold,
                        seed,
                        reference,
                        activity["reference_profile_score"][reference],
                        activity["reference_counts"][reference],
                        activity["reference_profile_jsd"][reference],
                        summaries["max_1bp"][reference],
                        summaries["max_5bp"][reference],
                        summaries["max_20bp"][reference],
                        summaries["peak_20bp_position"][reference],
                        summaries["peak_20bp_strand"][reference],
                        summaries["profile_entropy"][reference],
                        summaries["effective_width"][reference],
                        activity["genomic_profile_score"],
                        activity["genomic_counts"],
                    ]
                )

        for head_index, head in enumerate(heads):
            wrapper = (
                ProfileWrapper(model) if head == "profile" else CountWrapper(model)
            )
            for seed_index, seed in enumerate(REFERENCE_SEEDS):
                attr = deep_lift_shap(
                    model=wrapper,
                    X=X,
                    references=reference_sequences[seed][None],
                    batch_size=BATCH_SIZE,
                    hypothetical=True,
                    warning_threshold=0.01,
                    device=DEVICE,
                )
                observed = as_numpy(attr * X)[0, :, logo_offsets[0] : logo_offsets[1]]
                deep_lift[head_index, fold, seed_index] = observed
            gradients[head_index, fold] = as_numpy(
                gradient_x_input(wrapper, X, DEVICE)
            )[0, :, logo_offsets[0] : logo_offsets[1]]
            point_scores[head_index, fold] = as_numpy(
                point_ism(wrapper, X, *logo_offsets, BATCH_SIZE, DEVICE)
            )[0]
            positions, scores = window_ism(
                wrapper,
                X,
                *logo_offsets,
                WINDOW_ISM_WIDTH,
                WINDOW_ISM_STRIDE,
                WINDOW_ISM_REPLACEMENTS,
                1000 + fold * 10 + head_index,
                BATCH_SIZE,
                DEVICE,
            )
            window_scores[head_index, fold] = scores
        for strand in range(2):
            wrapper = StrandProfileWrapper(model, strand)
            for seed_index, seed in enumerate(REFERENCE_SEEDS):
                if seed == WEIGHTED_REFERENCE_SEED:
                    multipliers = deep_lift_shap(
                        model=wrapper,
                        X=X,
                        references=reference_sequences[seed][None],
                        batch_size=BATCH_SIZE,
                        raw_outputs=True,
                        warning_threshold=0.01,
                        device=DEVICE,
                    )
                    per_reference = as_numpy(
                        per_reference_attributions(
                            multipliers, X, reference_sequences[seed]
                        )
                    )
                    per_reference_strand_deep_lift[strand, fold] = per_reference
                    observed = per_reference.mean(axis=0)[
                        :, logo_offsets[0] : logo_offsets[1]
                    ]
                    scores = as_numpy(
                        predict(
                            model=wrapper,
                            X=torch.cat([X.cpu(), reference_sequences[seed].cpu()]),
                            batch_size=BATCH_SIZE,
                            device=DEVICE,
                        )
                    ).reshape(-1)
                    strand_profile_deltas[strand, fold] = scores[0] - scores[1:]
                else:
                    attr = deep_lift_shap(
                        model=wrapper,
                        X=X,
                        references=reference_sequences[seed][None],
                        batch_size=BATCH_SIZE,
                        hypothetical=True,
                        warning_threshold=0.01,
                        device=DEVICE,
                    )
                    observed = as_numpy(attr * X)[
                        0, :, logo_offsets[0] : logo_offsets[1]
                    ]
                strand_profile_deep_lift[strand, fold, seed_index] = observed
        del model, wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    reference_weights = reference_weight_schemes(
        weighted_seed_probabilities,
        weighted_seed_counts,
        temperature=REFERENCE_WEIGHT_TEMPERATURE,
    )
    result = {
        "deep_lift": deep_lift,
        "strand_profile_deep_lift": strand_profile_deep_lift,
        "per_reference_strand_deep_lift": per_reference_strand_deep_lift,
        "strand_profile_deltas": strand_profile_deltas,
        "reference_weight_metrics": reference_weights["metrics"],
        "reference_contamination": reference_weights["contamination"],
        "reference_weights": reference_weights["weights"],
        "gradients": gradients,
        "point_ism": point_scores,
        "window_positions": window_positions,
        "window_ism": window_scores,
        "activity": np.asarray(activity_rows, dtype=np.float64),
        "reference_centered_logits": reference_centered_logits,
        "reference_probabilities": reference_probabilities,
        "reference_count_scaled": reference_count_scaled,
        "genomic_centered_logits": genomic_centered_logits,
        "genomic_probabilities": genomic_probabilities,
        "genomic_count_scaled": genomic_count_scaled,
        "dinucleotide_frequencies": dinucleotide_frequencies(X),
    }
    temporary_cache = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_cache, **result)
    temporary_cache.replace(cache_path)
    print(f"Saved {cache_path}")
    return result, cache_path


diagnostics, diagnostic_cache = run_diagnostics(force=FORCE_DIAGNOSTICS)


def oriented_matrix(matrix):
    return reverse_complement_matrix(matrix) if REVERSE_COMPLEMENT else matrix


def logo_ticks(ax, start, end):
    width = end - start
    positions = np.linspace(0, width - 1, 5, dtype=int)
    labels = [end - p if REVERSE_COMPLEMENT else start + p + 1 for p in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{value:,}" for value in labels])


def plot_dinucleotide_spectrum():
    values = diagnostics["dinucleotide_frequencies"].reshape(-1)
    labels = [a + b for a in "ACGT" for b in "ACGT"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel("Genomic input frequency")
    ax.set_title("2114 bp genomic input dinucleotide spectrum")
    return fig, ax


def activity_frame():
    columns = [
        "fold",
        "seed",
        "reference",
        "profile_score",
        "counts",
        "profile_jsd",
        "max_1bp",
        "max_5bp",
        "max_20bp",
        "peak_20bp_position",
        "peak_20bp_strand",
        "profile_entropy",
        "effective_width",
        "genomic_profile_score",
        "genomic_counts",
    ]
    frame = pd.DataFrame(diagnostics["activity"], columns=columns)
    frame[["fold", "seed", "reference", "peak_20bp_position", "peak_20bp_strand"]] = (
        frame[
            ["fold", "seed", "reference", "peak_20bp_position", "peak_20bp_strand"]
        ].astype(int)
    )
    return frame


def plot_reference_activity():
    frame = activity_frame()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax, metric, title in zip(
        axes,
        ["profile_score", "counts", "profile_jsd"],
        ["Profile target score", "Predicted counts", "Profile JSD from genomic input"],
    ):
        sns.boxplot(data=frame, x="seed", y=metric, color="#9ECAE1", fliersize=1, ax=ax)
        if metric != "profile_jsd":
            genomic = frame[
                "genomic_profile_score"
                if metric == "profile_score"
                else "genomic_counts"
            ].mean()
            ax.axhline(genomic, color="#C44E52", linestyle="--", label="genomic input")
            ax.legend(frameon=False)
        ax.set_title(title)
    fig.tight_layout()
    return fig, axes


def reference_order():
    frame = (
        activity_frame()
        .groupby(["seed", "reference"], as_index=False)
        .mean(numeric_only=True)
    )
    ordered = []
    boundaries = []
    for seed_index, seed in enumerate(REFERENCE_SEEDS):
        subset = frame[frame["seed"] == seed].sort_values("max_20bp", ascending=False)
        ordered.extend(
            (seed_index, int(reference)) for reference in subset["reference"]
        )
        boundaries.append(len(ordered))
    return ordered, boundaries


def ordered_reference_profiles(key):
    profiles = diagnostics[key].astype(np.float32).mean(axis=0)
    rows = np.stack(
        [
            profiles[seed_index, reference]
            for seed_index, reference in reference_order()[0]
        ]
    )
    if REVERSE_COMPLEMENT:
        rows = rows[:, [1, 0], ::-1]
    return rows


def plot_reference_profile_heatmaps(key="reference_count_scaled"):
    rows = ordered_reference_profiles(key)
    _, boundaries = reference_order()
    label = (
        "Count-scaled signal"
        if key == "reference_count_scaled"
        else "Centered profile logits"
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for strand, ax in enumerate(axes):
        values = rows[:, strand]
        if key == "reference_centered_logits":
            limit = np.quantile(np.abs(values), 0.995)
            sns.heatmap(
                values,
                cmap="vlag",
                center=0,
                vmin=-limit,
                vmax=limit,
                ax=ax,
                cbar_kws={"label": label},
            )
        else:
            limit = np.quantile(values, 0.995)
            sns.heatmap(
                values,
                cmap="mako",
                vmin=0,
                vmax=limit,
                ax=ax,
                cbar_kws={"label": label},
            )
        for boundary in boundaries[:-1]:
            ax.axhline(boundary, color="white", linewidth=1)
        strand_name = ("plus", "minus")[strand]
        ax.set_title(f"{strand_name} strand")
        ax.set_xlabel("BPNet output position")
    axes[0].set_ylabel("References grouped by seed; sorted by max 20 bp activity")
    fig.suptitle(label)
    fig.tight_layout()
    return fig, axes


def plot_reference_peak_scatter():
    frame = (
        activity_frame()
        .groupby(["seed", "reference"], as_index=False)
        .mean(numeric_only=True)
    )
    genomic_profiles = diagnostics["genomic_count_scaled"].astype(np.float32)
    genomic_max, _, _ = rolling_profile_maxima(genomic_profiles, windows=(20,))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=frame, x="counts", y="max_20bp", hue="seed", palette="tab10", ax=ax
    )
    ax.scatter(
        activity_frame()["genomic_counts"].mean(),
        genomic_max[20].mean(),
        marker="*",
        s=180,
        color="black",
        label="genomic input",
    )
    strongest = frame.nlargest(8, "max_20bp")
    for row in strongest.itertuples():
        ax.annotate(
            f"s{row.seed}/r{row.reference}", (row.counts, row.max_20bp), fontsize=8
        )
    ax.set_xlabel("Predicted total counts")
    ax.set_ylabel("Maximum 20 bp count-scaled signal")
    ax.set_title("Shuffled-reference activity")
    return fig, ax


def plot_reference_position_envelope():
    profiles = (
        diagnostics["reference_count_scaled"]
        .astype(np.float32)
        .mean(axis=0)
        .reshape(-1, 2, OUT_WINDOW)
    )
    genomic = diagnostics["genomic_count_scaled"].astype(np.float32).mean(axis=0)
    if REVERSE_COMPLEMENT:
        profiles = profiles[:, [1, 0], ::-1]
        genomic = genomic[[1, 0], ::-1]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for strand, ax in enumerate(axes):
        values = profiles[:, strand]
        ax.plot(values.mean(axis=0), label="reference mean", color="#4C72B0")
        ax.plot(
            np.quantile(values, 0.95, axis=0),
            label="reference 95th percentile",
            color="#55A868",
        )
        ax.plot(values.max(axis=0), label="reference maximum", color="#C44E52")
        ax.plot(
            genomic[strand],
            label="genomic input",
            color="black",
            linestyle="--",
            alpha=0.7,
        )
        ax.set_ylabel("Count-scaled signal")
        ax.legend(frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("BPNet output position")
    fig.suptitle("Position-wise shuffled-reference activity")
    fig.tight_layout()
    return fig, axes


def plot_example_reference_profiles(selection="strongest"):
    frame = (
        activity_frame()
        .groupby(["seed", "reference"], as_index=False)
        .mean(numeric_only=True)
    )
    examples = []
    for seed in REFERENCE_SEEDS:
        seed_frame = frame[frame["seed"] == seed]
        if selection == "strongest":
            examples.append(seed_frame.loc[seed_frame["max_20bp"].idxmax()])
        elif selection == "median":
            median = seed_frame["max_20bp"].median()
            examples.append(
                seed_frame.loc[(seed_frame["max_20bp"] - median).abs().idxmin()]
            )
        else:
            raise ValueError("selection must be 'strongest' or 'median'")
    examples = pd.DataFrame(examples)
    fig, axes = plt.subplots(
        2, len(examples), figsize=(4.5 * len(examples), 6), squeeze=False, sharex=True
    )
    for column, row in enumerate(examples.itertuples()):
        seed_index = REFERENCE_SEEDS.index(row.seed)
        reference = int(row.reference)
        logits = (
            diagnostics["reference_centered_logits"][:, seed_index, reference]
            .astype(np.float32)
            .mean(axis=0)
        )
        signal = (
            diagnostics["reference_count_scaled"][:, seed_index, reference]
            .astype(np.float32)
            .mean(axis=0)
        )
        if REVERSE_COMPLEMENT:
            logits = logits[[1, 0], ::-1]
            signal = signal[[1, 0], ::-1]
        x = np.arange(OUT_WINDOW)
        axes[0, column].plot(x, logits[0], color="#C44E52", label="plus")
        axes[0, column].plot(x, logits[1], color="#4C72B0", label="minus")
        axes[0, column].axhline(0, color="black", linewidth=0.7)
        axes[0, column].set_title(
            f"seed {int(row.seed)}, reference {reference}\nmax 20 bp={row.max_20bp:.2f}, counts={row.counts:.2f}"
        )
        axes[0, column].set_ylabel("Jointly centered profile logit")
        axes[0, column].legend(frameon=False)
        axes[1, column].plot(x, signal[0], color="#C44E52", label="plus")
        axes[1, column].plot(x, -signal[1], color="#4C72B0", label="minus")
        axes[1, column].axhline(0, color="black", linewidth=0.7)
        axes[1, column].set_ylabel("Count-scaled signal")
        axes[1, column].set_xlabel("BPNet output position")
    fig.suptitle(f"{selection.capitalize()} shuffled-reference profile per seed")
    fig.tight_layout()
    return fig, axes


def plot_ranked_reference_metrics():
    frame = (
        activity_frame()
        .groupby(["seed", "reference"], as_index=False)
        .mean(numeric_only=True)
    )
    genomic_profiles = diagnostics["genomic_count_scaled"].astype(np.float32)
    genomic_maxima, _, _ = rolling_profile_maxima(genomic_profiles, windows=(1, 5, 20))
    genomic_probabilities = diagnostics["genomic_probabilities"].astype(np.float32)
    genomic_probabilities = genomic_probabilities.reshape(
        len(genomic_probabilities), -1
    )
    genomic_entropy = -(
        genomic_probabilities * np.log(genomic_probabilities.clip(min=1e-12))
    ).sum(axis=1)
    genomic_baselines = {
        "counts": activity_frame()["genomic_counts"].mean(),
        "max_1bp": genomic_maxima[1].mean(),
        "max_5bp": genomic_maxima[5].mean(),
        "max_20bp": genomic_maxima[20].mean(),
        "profile_entropy": genomic_entropy.mean(),
    }
    metrics = ["counts", "max_1bp", "max_5bp", "max_20bp", "profile_entropy"]
    titles = [
        "Counts",
        "Maximum 1 bp",
        "Maximum 5 bp",
        "Maximum 20 bp",
        "Profile entropy",
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    for ax, metric, title in zip(axes, metrics, titles):
        values = frame.sort_values(metric, ascending=False)[metric].to_numpy()
        ax.plot(np.arange(1, len(values) + 1), values, color="#4C72B0")
        ax.axhline(
            genomic_baselines[metric],
            color="#C44E52",
            linestyle="--",
            label="genomic input",
        )
        ax.set_title(title)
        ax.set_xlabel("Reference rank")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig, axes


def reference_fold_consensus():
    frame = activity_frame()
    shape = (N_FOLDS, len(REFERENCE_SEEDS), N_REFERENCES)
    positions = np.empty(shape)
    strands = np.empty(shape)
    activities = np.empty(shape)
    for row in frame.itertuples():
        seed_index = REFERENCE_SEEDS.index(row.seed)
        positions[row.fold, seed_index, row.reference] = row.peak_20bp_position
        strands[row.fold, seed_index, row.reference] = row.peak_20bp_strand
        activities[row.fold, seed_index, row.reference] = row.max_20bp
    consensus = fold_consensus(positions, strands, activities)
    seed_grid, reference_grid = np.meshgrid(
        REFERENCE_SEEDS, np.arange(N_REFERENCES), indexing="ij"
    )
    result = pd.DataFrame(
        {"seed": seed_grid.ravel(), "reference": reference_grid.ravel()}
    )
    for key, value in consensus.items():
        result[key] = value.ravel()
    mean_positions = positions.mean(axis=0)
    mean_strands = np.rint(strands.mean(axis=0)).astype(int)
    if REVERSE_COMPLEMENT:
        mean_positions, mean_strands = reverse_complement_peak_coordinates(
            mean_positions, mean_strands, OUT_WINDOW, 20
        )
    result["mean_peak_position"] = mean_positions.ravel()
    return result


def plot_fold_consensus():
    frame = reference_fold_consensus()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.scatterplot(
        data=frame,
        x="peak_position_sd",
        y="peak_strand_agreement",
        hue="peak_activity_mean",
        palette="viridis",
        ax=axes[0],
    )
    axes[0].set_title("Fold agreement of strongest 20 bp window")
    sns.scatterplot(
        data=frame,
        x="mean_peak_position",
        y="peak_activity_mean",
        hue="seed",
        palette="tab10",
        ax=axes[1],
    )
    axes[1].set_title("Recurring peak positions across folds")
    fig.tight_layout()
    return fig, axes


def seed_instability_frame(head="profile"):
    index = 0 if head == "profile" else 1
    attribution = (
        diagnostics["deep_lift"][index].mean(axis=0).reshape(len(REFERENCE_SEEDS), -1)
    )
    consensus = attribution.mean(axis=0)
    correlations = [np.corrcoef(values, consensus)[0, 1] for values in attribution]
    activity = (
        activity_frame()
        .groupby("seed")
        .agg(
            mean_counts=("counts", "mean"),
            max_counts=("counts", "max"),
            mean_max_20bp=("max_20bp", "mean"),
            max_max_20bp=("max_20bp", "max"),
            mean_entropy=("profile_entropy", "mean"),
        )
        .reset_index()
    )
    activity["attribution_disagreement"] = 1 - np.asarray(correlations)
    return activity


def plot_instability_linkage(head="profile"):
    frame = seed_instability_frame(head)
    metrics = ["mean_counts", "max_counts", "max_max_20bp", "mean_entropy"]
    titles = [
        "Mean reference counts",
        "Maximum reference counts",
        "Maximum 20 bp activity",
        "Mean profile entropy",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ax, metric, title in zip(axes, metrics, titles):
        ax.scatter(frame[metric], frame["attribution_disagreement"], color="#4C72B0")
        for row in frame.itertuples():
            ax.annotate(
                f"seed {row.seed}",
                (getattr(row, metric), row.attribution_disagreement),
                fontsize=8,
            )
        ax.set_xlabel(title)
    axes[0].set_ylabel("1 - correlation to seed-consensus attribution")
    fig.suptitle(f"{head} DeepLIFT instability versus shuffled-reference activity")
    fig.tight_layout()
    return fig, axes


def plot_deeplift_stability(head="profile"):
    head_index = 0 if head == "profile" else 1
    values = diagnostics["deep_lift"][head_index]
    seed_vectors = values.mean(axis=0).reshape(len(REFERENCE_SEEDS), -1)
    fold_vectors = values.mean(axis=1).reshape(N_FOLDS, -1)
    position_values = values.sum(axis=2)
    if REVERSE_COMPLEMENT:
        position_values = position_values[..., ::-1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.heatmap(
        np.corrcoef(seed_vectors),
        vmin=-1,
        vmax=1,
        cmap="vlag",
        xticklabels=REFERENCE_SEEDS,
        yticklabels=REFERENCE_SEEDS,
        ax=axes[0],
    )
    axes[0].set_title(f"{head} DeepLIFT seed correlation")
    sns.heatmap(np.corrcoef(fold_vectors), vmin=-1, vmax=1, cmap="vlag", ax=axes[1])
    axes[1].set_title(f"{head} DeepLIFT fold correlation")
    mean = position_values.mean(axis=(0, 1))
    std = position_values.std(axis=(0, 1))
    x = np.arange(len(mean))
    axes[2].plot(x, mean, color="#4C72B0")
    axes[2].fill_between(x, mean - std, mean + std, color="#4C72B0", alpha=0.25)
    logo_ticks(axes[2], logo_start, logo_end)
    axes[2].set_title("Mean +/- SD across folds and seeds")
    fig.tight_layout()
    return fig, axes


def plot_method_logos(head="profile"):
    index = 0 if head == "profile" else 1
    matrices = [
        diagnostics["deep_lift"][index].mean(axis=(0, 1)),
        diagnostics["gradients"][index].mean(axis=0),
        diagnostics["point_ism"][index].mean(axis=0),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    for ax, matrix, title in zip(
        axes, matrices, ["DeepLIFT/SHAP", "Gradient x input", "Single-base ISM"]
    ):
        plot_logo(torch.tensor(oriented_matrix(matrix), dtype=torch.float32), ax=ax)
        ax.set_title(f"{head}: {title}")
        logo_ticks(ax, logo_start, logo_end)
    fig.tight_layout()
    return fig, axes


def plot_seed_deeplift_logos():
    values = diagnostics["deep_lift"].mean(axis=1)
    rows = len(REFERENCE_SEEDS) * 2
    fig, axes = plt.subplots(rows, 1, figsize=(20, 2.2 * rows), sharex=True)
    for head_index, head in enumerate(("profile", "count")):
        for seed_index, seed in enumerate(REFERENCE_SEEDS):
            ax = axes[head_index * len(REFERENCE_SEEDS) + seed_index]
            plot_logo(
                torch.tensor(
                    oriented_matrix(values[head_index, seed_index]),
                    dtype=torch.float32,
                ),
                ax=ax,
            )
            ax.set_title(f"{head}, seed {seed}")
            logo_ticks(ax, logo_start, logo_end)
    fig.suptitle("DeepLIFT/SHAP by reference seed, averaged across folds")
    fig.tight_layout()
    return fig, axes


def plot_strand_profile_deeplift():
    values = diagnostics["strand_profile_deep_lift"].mean(axis=(1, 2))
    if REVERSE_COMPLEMENT:
        values = values[[1, 0]]
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    for strand, ax in enumerate(axes):
        plot_logo(
            torch.tensor(oriented_matrix(values[strand]), dtype=torch.float32), ax=ax
        )
        ax.set_title(f"{('plus', 'minus')[strand]}-strand profile DeepLIFT/SHAP")
        logo_ticks(ax, logo_start, logo_end)
    fig.suptitle("Joint-softmax strand-specific profile attribution")
    fig.tight_layout()
    return fig, axes


def plot_seed_strand_profile_deeplift():
    values = diagnostics["strand_profile_deep_lift"].mean(axis=1)
    if REVERSE_COMPLEMENT:
        values = values[[1, 0]]
    rows = len(REFERENCE_SEEDS) * 2
    fig, axes = plt.subplots(rows, 1, figsize=(20, 2.2 * rows), sharex=True)
    for strand_index, strand in enumerate(("plus", "minus")):
        for seed_index, seed in enumerate(REFERENCE_SEEDS):
            ax = axes[strand_index * len(REFERENCE_SEEDS) + seed_index]
            plot_logo(
                torch.tensor(
                    oriented_matrix(values[strand_index, seed_index]),
                    dtype=torch.float32,
                ),
                ax=ax,
            )
            ax.set_title(f"{strand}, seed {seed}")
            logo_ticks(ax, logo_start, logo_end)
    fig.suptitle(
        "Strand-specific profile DeepLIFT/SHAP by seed, averaged across folds"
    )
    fig.tight_layout()
    return fig, axes


def plot_reference_weights():
    metric_names = [
        "Maximum 20 bp probability mass",
        "Strand-mass imbalance",
        "log1p predicted counts",
    ]
    scheme_names = ["uniform", "profile-only", "profile + counts"]
    metrics = diagnostics["reference_weight_metrics"]
    weights = diagnostics["reference_weights"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for metric_index, (ax, metric_name) in enumerate(zip(axes, metric_names)):
        for scheme_index, scheme_name in enumerate(scheme_names):
            ax.scatter(
                metrics[metric_index],
                weights[scheme_index],
                s=28,
                alpha=0.8,
                label=scheme_name,
            )
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Reference weight")
        ax.set_title(f"Seed {WEIGHTED_REFERENCE_SEED}")
    axes[0].legend(frameon=False)
    fig.suptitle("Shared reference weights versus contamination metrics")
    fig.tight_layout()
    return fig, axes


def weighted_profile_attributions():
    per_reference = diagnostics["per_reference_strand_deep_lift"].astype(np.float32)
    schemes = []
    for weights in diagnostics["reference_weights"]:
        fold_values = np.stack(
            [
                np.stack(
                    [
                        weighted_strand_attributions(
                            per_reference[strand, fold], weights
                        )
                        for strand in range(2)
                    ]
                )
                for fold in range(N_FOLDS)
            ]
        )
        schemes.append(fold_values)
    return np.asarray(schemes)


def plot_weighted_profile_logos():
    scheme_names = ["uniform", "profile-only", "profile + counts"]
    values = weighted_profile_attributions().mean(axis=1)
    fig, axes = plt.subplots(3, 3, figsize=(16, 8), sharex=True)
    for column, scheme_name in enumerate(scheme_names):
        strand_values = values[column]
        if REVERSE_COMPLEMENT:
            strand_values = strand_values[[1, 0]]
        matrices = [strand_values[0], strand_values[1], strand_values.sum(axis=0)]
        for row, (matrix, row_name) in enumerate(
            zip(matrices, ["plus", "minus", "combined"])
        ):
            cropped = matrix[:, logo_offsets[0] : logo_offsets[1]]
            plot_logo(
                torch.tensor(oriented_matrix(cropped), dtype=torch.float32),
                ax=axes[row, column],
            )
            axes[row, column].set_title(f"{scheme_name}: {row_name}")
            logo_ticks(axes[row, column], logo_start, logo_end)
    fig.suptitle(
        f"Completeness-preserving profile DeepLIFT, seed {WEIGHTED_REFERENCE_SEED}"
    )
    fig.tight_layout()
    return fig, axes


def plot_weighted_completeness():
    scheme_names = ["uniform", "profile-only", "profile + counts"]
    weighted = weighted_profile_attributions()
    weights = diagnostics["reference_weights"]
    deltas = diagnostics["strand_profile_deltas"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    folds = np.arange(N_FOLDS)
    for scheme, (ax, scheme_name) in enumerate(zip(axes, scheme_names)):
        attribution_sums = weighted[scheme].sum(axis=(1, 2, 3))
        weighted_deltas = np.sum(deltas * weights[scheme][None, None, :], axis=(0, 2))
        residuals = attribution_sums - weighted_deltas
        ax.plot(folds, attribution_sums, marker="o", label="summed attribution")
        ax.plot(
            folds,
            weighted_deltas,
            marker="s",
            linestyle="--",
            label="weighted output delta",
        )
        ax.bar(folds, residuals, alpha=0.25, label="residual")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(scheme_name)
        ax.set_xlabel("Fold")
    axes[0].set_ylabel("Profile target score")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Full-input DeepLIFT completeness")
    fig.tight_layout()
    return fig, axes


def plot_window_ism():
    positions = diagnostics["window_positions"].astype(int)
    genomic_positions = center - IN_WINDOW // 2 + positions + WINDOW_ISM_WIDTH // 2
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    for index, (ax, head) in enumerate(zip(axes, ["profile", "count"])):
        values = diagnostics["window_ism"][index]
        mean, std = values.mean(axis=0), values.std(axis=0)
        x = genomic_positions.copy()
        if REVERSE_COMPLEMENT:
            x, mean, std = x[::-1], mean[::-1], std[::-1]
        ax.plot(x, mean, color="#4C72B0")
        ax.fill_between(x, mean - std, mean + std, color="#4C72B0", alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylabel("Original - perturbed")
        ax.set_title(f"{head} 10 bp Markov replacement ISM")
    axes[-1].set_xlabel(f"{chrom} position (0-based)")
    fig.tight_layout()
    return fig, axes


def plot_window_ism_logos():
    position_scores = window_scores_to_positions(
        diagnostics["window_positions"],
        diagnostics["window_ism"].mean(axis=1),
        IN_WINDOW,
        WINDOW_ISM_WIDTH,
    )[:, logo_offsets[0] : logo_offsets[1]]
    sequence = as_numpy(X)[0, :, logo_offsets[0] : logo_offsets[1]]
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    for index, (ax, head) in enumerate(zip(axes, ("profile", "count"))):
        matrix = sequence * position_scores[index][None]
        plot_logo(
            torch.tensor(oriented_matrix(matrix), dtype=torch.float32), ax=ax
        )
        ax.set_title(f"{head}: mean overlapping-window ISM score")
        logo_ticks(ax, logo_start, logo_end)
    fig.suptitle("Window ISM projected onto the genomic sequence")
    fig.tight_layout()
    return fig, axes


def bigwig_values(path, chrom, start, end):
    with pybigtools.open(str(path)) as bw:
        return np.nan_to_num(np.asarray(bw.values(chrom, start, end), dtype=float))


def scaled_prediction(model, X):
    logits, log_counts = predict(model=model, X=X, batch_size=1, device=DEVICE)
    logits = as_numpy(logits)
    flat = logits.reshape(1, -1)
    probabilities = np.exp(flat - flat.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return (probabilities * np.exp(as_numpy(log_counts).reshape(1, -1))).reshape(
        logits.shape
    )[0]


def plot_tracks():
    view_chrom, start, end = parse_interval(VIEW_REGION)
    if view_chrom != chrom:
        raise ValueError("POINT_REGION and VIEW_REGION must use the same chromosome")
    output_start = center - OUT_WINDOW // 2
    output_end = output_start + OUT_WINDOW
    if start < output_start or end > output_end:
        raise ValueError(
            f"VIEW_REGION spans {end - start} bp at {view_chrom}:{start + 1}-{end}, "
            f"but predictions cover only {view_chrom}:{output_start + 1}-{output_end}. "
            "Choose a 1-based inclusive interval inside the model output."
        )
    left, right = start - output_start, end - output_start
    predictions = []
    for path in resources["model_paths"]:
        model = torch.load(path, map_location="cpu", weights_only=False).eval()
        predictions.append(scaled_prediction(model, X))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    prediction = np.mean(predictions, axis=0)
    plus = prediction[0, left:right]
    minus = -prediction[1, left:right]
    observed_plus = bigwig_values(resources["observed"]["plus"], chrom, start, end)
    observed_minus = -np.abs(
        bigwig_values(resources["observed"]["minus"], chrom, start, end)
    )
    x = np.arange(start, end)
    lengths = {len(x), len(plus), len(minus), len(observed_plus), len(observed_minus)}
    if len(lengths) != 1:
        raise ValueError(
            f"Track lengths do not agree: coordinates={len(x)}, predicted={len(plus)}, observed={len(observed_plus)}"
        )
    if REVERSE_COMPLEMENT:
        observed_plus, observed_minus = reverse_complement_tracks(
            observed_plus, observed_minus
        )
        plus, minus = reverse_complement_tracks(plus, minus)
        x = x[::-1]
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(x, observed_plus, color="#C44E52", label="observed plus")
    ax.plot(x, observed_minus, color="#4C72B0", label="observed minus")
    ax.plot(x, plus, color="#C44E52", linestyle="--", label="predicted plus")
    ax.plot(x, minus, color="#4C72B0", linestyle="--", label="predicted minus")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlim(x[0], x[-1])
    ax.set_title(f"{EXP_ID} {VIEW_REGION}")
    ax.set_ylabel("PRO-cap signal")
    ax.legend(frameon=False, ncol=2)
    return fig, ax


def save_figure(name, plot_result):
    fig = plot_result[0]
    path = OUTPUT_DIR / f"{name}.{args.format}"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


figure_jobs = [
    ("dinucleotide_spectrum", plot_dinucleotide_spectrum),
    ("reference_activity", plot_reference_activity),
    (
        "reference_count_scaled_heatmaps",
        lambda: plot_reference_profile_heatmaps("reference_count_scaled"),
    ),
    (
        "reference_centered_logit_heatmaps",
        lambda: plot_reference_profile_heatmaps("reference_centered_logits"),
    ),
    ("reference_peak_scatter", plot_reference_peak_scatter),
    ("reference_position_envelope", plot_reference_position_envelope),
    (
        "strongest_reference_profiles",
        lambda: plot_example_reference_profiles("strongest"),
    ),
    (
        "median_reference_profiles",
        lambda: plot_example_reference_profiles("median"),
    ),
    ("ranked_reference_metrics", plot_ranked_reference_metrics),
    ("fold_consensus", plot_fold_consensus),
    (
        "profile_deeplift_stability",
        lambda: plot_deeplift_stability("profile"),
    ),
    ("count_deeplift_stability", lambda: plot_deeplift_stability("count")),
    (
        "profile_instability_linkage",
        lambda: plot_instability_linkage("profile"),
    ),
    ("count_instability_linkage", lambda: plot_instability_linkage("count")),
    ("profile_method_logos", lambda: plot_method_logos("profile")),
    ("count_method_logos", lambda: plot_method_logos("count")),
    ("seed_deeplift_logos", plot_seed_deeplift_logos),
    ("strand_profile_deeplift", plot_strand_profile_deeplift),
    ("seed_strand_profile_deeplift", plot_seed_strand_profile_deeplift),
    ("reference_weights", plot_reference_weights),
    ("weighted_profile_logos", plot_weighted_profile_logos),
    ("weighted_completeness", plot_weighted_completeness),
    ("window_ism", plot_window_ism),
    ("window_ism_logos", plot_window_ism_logos),
    ("observed_predicted_tracks", plot_tracks),
]

for figure_name, plot_function in figure_jobs:
    save_figure(figure_name, plot_function())
