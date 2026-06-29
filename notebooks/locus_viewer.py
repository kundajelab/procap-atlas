"""Helpers for the lightweight BPNet locus viewer notebook."""

from __future__ import annotations

import gc
import gzip
import os
import shutil
import urllib.request
from pathlib import Path

cache_root = Path(os.environ.get("SCRATCH", "/tmp")) / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybigtools
import torch
import yaml
from bpnetlite.bpnet import CountWrapper, ProfileWrapper
from huggingface_hub import hf_hub_download
from pyfaidx import Fasta
from tangermeme.io import extract_loci
from tangermeme.plot import plot_logo
from tangermeme.predict import predict

from src.bpnet.attribute.deeplift import deep_lift_shap
from src.bpnet.attribute.locus_diagnostics import (
    as_numpy,
    genomic_offsets,
    reverse_complement_matrix,
    reverse_complement_tracks,
)

MODEL_REPO_ID = "adamyhe/procap-atlas"
TRACK_REPO_ID = "adamyhe/procap-atlas-tracks"
METADATA_REPO_ID = "adamyhe/procap-atlas-metadata"
REFERENCE_FASTA_URL = "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
IN_WINDOW = 2114
OUT_WINDOW = 1000
POINTS_PER_INCH = 72
SUMMARY_FIGURE_SIZE_PT = (570, 120)
SUMMARY_FIGURE_SIZE_IN = tuple(value / POINTS_PER_INCH for value in SUMMARY_FIGURE_SIZE_PT)


def parse_point(region: str) -> tuple[str, int]:
    """Return a 0-based center coordinate from ``chr:start`` input."""
    chrom, position = region.replace(",", "").split(":", 1)
    return chrom, int(position) - 1


def parse_interval(region: str) -> tuple[str, int, int]:
    """Return a 0-based half-open interval from ``chr:start-end`` input."""
    chrom, interval = region.replace(",", "").split(":", 1)
    start, end = [int(value) for value in interval.split("-", 1)]
    if end < start:
        raise ValueError("Interval end must not precede its start")
    return chrom, start - 1, end


def interval_center(start: int, end: int) -> int:
    """Return the 0-based center coordinate for a half-open interval."""
    return (start + end) // 2


def region_center(region: str) -> tuple[str, int]:
    """Return the model-center coordinate implied by a point or interval string."""
    if "-" in region.split(":", 1)[1]:
        chrom, start, end = parse_interval(region)
        return chrom, interval_center(start, end)
    return parse_point(region)


def download_first(repo_id: str, repo_type: str, filenames: list[str]) -> Path:
    """Download the first matching Hugging Face path from a fallback list."""
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


def download_reference(work_dir: Path) -> Path:
    """Download and index hg38 for local sequence extraction."""
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


def download_model_paths(exp_id: str, n_folds: int) -> list[Path]:
    """Resolve fold checkpoints from the model repository."""
    patterns = [
        "{exp_id}/{exp_id}.fold{fold}.torch",
        "bpnet/{exp_id}/{exp_id}.fold{fold}.torch",
        "models/bpnet/{exp_id}/{exp_id}.fold{fold}.torch",
    ]
    return [
        download_first(
            MODEL_REPO_ID,
            "model",
            [pattern.format(exp_id=exp_id, fold=fold) for pattern in patterns],
        )
        for fold in range(n_folds)
    ]


def setup_experiment(exp_id: str, work_dir: Path, n_folds: int) -> dict:
    """Download metadata, observed tracks, reference genome, and model paths."""
    config_path = download_first(
        METADATA_REPO_ID,
        "dataset",
        ["experiment_config.yaml", "configs/experiment_config.yaml"],
    )
    with open(config_path) as handle:
        config = yaml.safe_load(handle)["experiments"]
    if exp_id not in config:
        raise KeyError(f"{exp_id} is absent from experiment_config.yaml")
    return {
        "exp_id": exp_id,
        "config": config[exp_id],
        "fasta": download_reference(work_dir),
        "observed": {
            "plus": download_first(
                TRACK_REPO_ID, "dataset", [f"observed/{exp_id}_pl.bigWig"]
            ),
            "minus": download_first(
                TRACK_REPO_ID, "dataset", [f"observed/{exp_id}_mn.bigWig"]
            ),
        },
        "model_paths": download_model_paths(exp_id, n_folds),
    }


def locus_input(resources: dict, point_region: str) -> tuple[str, int, torch.Tensor]:
    """Extract the 2,114 bp one-hot input centered on ``point_region``."""
    chrom, center = parse_point(point_region)
    loci = pd.DataFrame({"chrom": [chrom], "start": [center], "end": [center + 1]})
    X = extract_loci(
        loci,
        sequences=str(resources["fasta"]),
        in_window=IN_WINDOW,
        ignore=["N", "n"],
    )
    if len(X) != 1:
        raise ValueError("The requested locus could not be extracted")
    return chrom, center, X.float()


def region_input(resources: dict, region: str) -> tuple[str, int, int, int, torch.Tensor]:
    """Extract model input centered on a region of interest."""
    chrom, start, end = parse_interval(region)
    center = interval_center(start, end)
    loci = pd.DataFrame({"chrom": [chrom], "start": [center], "end": [center + 1]})
    X = extract_loci(
        loci,
        sequences=str(resources["fasta"]),
        in_window=IN_WINDOW,
        ignore=["N", "n"],
    )
    if len(X) != 1:
        raise ValueError("The requested region could not be extracted")
    return chrom, start, end, center, X.float()


def nucleotide_frequency_references(X: torch.Tensor) -> torch.Tensor:
    """Create one soft reference from the input-wide A/C/G/T frequencies."""
    frequencies = X.float().mean(dim=-1, keepdim=True)
    return frequencies.expand_as(X).unsqueeze(1).clone()


def logo_offsets_for_locus(
    point_region: str, logo_region: str
) -> tuple[str, int, int, tuple[int, int]]:
    """Validate the logo interval and map it onto the model input window."""
    chrom, center = parse_point(point_region)
    logo_chrom, logo_start, logo_end = parse_interval(logo_region)
    if logo_chrom != chrom:
        raise ValueError("POINT_REGION and LOGO_REGION must use the same chromosome")
    offsets = genomic_offsets(center, logo_start, logo_end, IN_WINDOW)
    return logo_chrom, logo_start, logo_end, offsets


def logo_offsets_for_region(region: str) -> tuple[str, int, int, tuple[int, int]]:
    """Map a region of interest onto the input window centered on that region."""
    chrom, start, end = parse_interval(region)
    offsets = genomic_offsets(interval_center(start, end), start, end, IN_WINDOW)
    return chrom, start, end, offsets


def scaled_prediction(model: torch.nn.Module, X: torch.Tensor, device: str) -> np.ndarray:
    """Return count-scaled strand profiles for one model."""
    logits, log_counts = predict(model=model, X=X, batch_size=1, device=device)
    logits = as_numpy(logits).astype(np.float32)
    flat = logits.reshape(logits.shape[0], -1)
    probabilities = np.exp(flat - flat.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    counts = np.exp(as_numpy(log_counts).reshape(logits.shape[0], -1))
    return (probabilities * counts).reshape(logits.shape)[0]


def ensemble_prediction(
    resources: dict,
    X: torch.Tensor,
    n_folds: int,
    device: str,
) -> np.ndarray:
    """Average count-scaled predictions across fold checkpoints."""
    predictions = []
    for fold, path in enumerate(resources["model_paths"][:n_folds]):
        print(f"Predicting fold {fold + 1}/{n_folds}: {path.name}")
        model = torch.load(path, map_location="cpu", weights_only=False).eval()
        predictions.append(scaled_prediction(model, X, device))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.mean(predictions, axis=0)


def bigwig_values(path: Path, chrom: str, start: int, end: int) -> np.ndarray:
    """Read one BigWig interval and replace missing values with zero."""
    with pybigtools.open(str(path)) as bw:
        return np.nan_to_num(np.asarray(bw.values(chrom, start, end), dtype=float))


def track_arrays(
    prediction: np.ndarray,
    resources: dict,
    center_region: str,
    view_region: str,
    reverse_complement: bool = False,
) -> dict[str, np.ndarray]:
    """Return observed and predicted plus/minus tracks for the view interval."""
    chrom, center = region_center(center_region)
    view_chrom, start, end = parse_interval(view_region)
    if view_chrom != chrom:
        raise ValueError("Center region and view region must use the same chromosome")
    output_start = center - OUT_WINDOW // 2
    output_end = output_start + OUT_WINDOW
    if start < output_start or end > output_end:
        raise ValueError(
            f"VIEW_REGION spans {view_chrom}:{start + 1}-{end}, but predictions "
            f"cover only {view_chrom}:{output_start + 1}-{output_end}."
        )
    left, right = start - output_start, end - output_start
    predicted_plus = prediction[0, left:right]
    predicted_minus = -prediction[1, left:right]
    observed_plus = bigwig_values(resources["observed"]["plus"], chrom, start, end)
    observed_minus = -np.abs(
        bigwig_values(resources["observed"]["minus"], chrom, start, end)
    )
    x = np.arange(start + 1, end + 1)
    lengths = {
        len(x),
        len(predicted_plus),
        len(predicted_minus),
        len(observed_plus),
        len(observed_minus),
    }
    if len(lengths) != 1:
        raise ValueError(f"Track lengths do not agree: {lengths}")
    if reverse_complement:
        observed_plus, observed_minus = reverse_complement_tracks(
            observed_plus, observed_minus
        )
        predicted_plus, predicted_minus = reverse_complement_tracks(
            predicted_plus, predicted_minus
        )
        x = x[::-1]
    return {
        "x": x,
        "observed_plus": observed_plus,
        "observed_minus": observed_minus,
        "predicted_plus": predicted_plus,
        "predicted_minus": predicted_minus,
    }


def signed_value_transform(values: np.ndarray, transform: str | None) -> np.ndarray:
    """Apply an optional sign-preserving display transform to signal tracks."""
    if isinstance(transform, str):
        transform = transform.lower()
    if transform is None or transform in {"none", "linear"}:
        return values
    absolute = np.abs(values)
    if transform == "sqrt":
        transformed = np.sqrt(absolute)
    elif transform in {"log", "log1p"}:
        transformed = np.log1p(absolute)
    else:
        raise ValueError("track_transform must be one of None, 'sqrt', or 'log1p'")
    return np.sign(values) * transformed


def transform_track_arrays(
    tracks: dict[str, np.ndarray],
    track_transform: str | None = None,
) -> dict[str, np.ndarray]:
    """Return tracks with an optional display transform applied to signal arrays."""
    return {
        key: (
            signed_value_transform(value, track_transform)
            if key != "x"
            else value
        )
        for key, value in tracks.items()
    }


def emphasize_left_y_axis(ax, linewidth: float = 0.8) -> None:
    """Draw the left y-axis in black while preserving the active plot style."""
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(linewidth)
    ax.tick_params(axis="y", color="black", labelcolor="black")


def format_track_axis(
    ax,
    x: np.ndarray,
    title: str,
    track_transform: str | None = None,
    show_title: bool = True,
    show_legend: bool = True,
) -> None:
    """Apply shared formatting to one plus/minus signal track axis."""
    ax.set_xlim(x[0], x[-1])
    if show_title:
        ax.set_title(title)
    ylabel = "PRO-cap signal"
    if track_transform not in {None, "none", "linear"}:
        ylabel = f"{track_transform} {ylabel}"
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(frameon=False, ncol=2)
    emphasize_left_y_axis(ax)


def shared_ticks(x_limits: tuple[float, float], n_ticks: int = 5) -> np.ndarray:
    """Return common integer genomic tick positions for stacked panels."""
    return np.linspace(x_limits[0], x_limits[1], n_ticks, dtype=int)


def apply_shared_ticks(ax, ticks: np.ndarray, show_labels: bool = False) -> None:
    """Apply the same x tick locations to one panel."""
    ax.set_xticks(ticks)
    if show_labels:
        ax.set_xticklabels([f"{value:,}" for value in ticks])
    else:
        ax.tick_params(axis="x", labelbottom=False)


def plot_tracks(
    prediction: np.ndarray,
    resources: dict,
    exp_id: str,
    point_region: str,
    view_region: str,
    reverse_complement: bool = False,
    track_transform: str | None = None,
):
    """Plot observed and predicted PRO-cap signal on separate y scales."""
    tracks = track_arrays(
        prediction, resources, point_region, view_region, reverse_complement
    )
    tracks = transform_track_arrays(tracks, track_transform)
    x = tracks["x"]
    ticks = shared_ticks((float(x[0]), float(x[-1])))
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.2), sharex=True)
    axes[0].plot(x, tracks["observed_plus"], color="#C44E52", label="observed plus")
    axes[0].plot(x, tracks["observed_minus"], color="#4C72B0", label="observed minus")
    format_track_axis(axes[0], x, f"{exp_id} observed {view_region}", track_transform)
    axes[1].plot(
        x,
        tracks["predicted_plus"],
        color="#C44E52",
        linestyle="--",
        label="predicted plus",
    )
    axes[1].plot(
        x,
        tracks["predicted_minus"],
        color="#4C72B0",
        linestyle="--",
        label="predicted minus",
    )
    format_track_axis(axes[1], x, f"{exp_id} predicted {view_region}", track_transform)
    apply_shared_ticks(axes[0], ticks)
    apply_shared_ticks(axes[1], ticks, show_labels=True)
    axes[1].set_xlabel("Genomic position")
    fig.tight_layout()
    return fig, axes


def shift_logo_to_genomic_axis(
    ax,
    logo_start: int,
    logo_end: int,
    reverse_complement: bool = False,
) -> None:
    """Move logo glyphs from logo-local x positions to genomic coordinates."""
    for collection in ax.collections:
        for path in collection.get_paths():
            vertices = path.vertices
            if reverse_complement:
                vertices[:, 0] = logo_end - vertices[:, 0]
            else:
                vertices[:, 0] = logo_start + 1 + vertices[:, 0]


def plot_logo_panel(
    ax,
    matrix: np.ndarray,
    title: str,
    logo_start: int,
    logo_end: int,
    reverse_complement: bool = False,
    x_limits: tuple[float, float] | None = None,
    ticks: np.ndarray | None = None,
    show_tick_labels: bool = False,
    show_title: bool = True,
) -> None:
    """Draw one DeepLIFT logo panel with genomic coordinate ticks."""
    plot_logo(torch.tensor(matrix, dtype=torch.float32), ax=ax)
    if show_title:
        ax.set_title(title)
    if x_limits is None:
        logo_ticks(ax, logo_start, logo_end, reverse_complement)
    else:
        shift_logo_to_genomic_axis(ax, logo_start, logo_end, reverse_complement)
        ax.set_xlim(*x_limits)
        if ticks is not None:
            apply_shared_ticks(ax, ticks, show_labels=show_tick_labels)
    emphasize_left_y_axis(ax)


def plot_locus_summary(
    prediction: np.ndarray,
    attributions: dict[str, np.ndarray],
    resources: dict,
    exp_id: str,
    point_region: str,
    view_region: str,
    logo_region: str,
    logo_start: int,
    logo_end: int,
    reverse_complement: bool = False,
    track_transform: str | None = None,
):
    """Stack observed tracks, predicted tracks, and DeepLIFT logos in one figure."""
    tracks = track_arrays(
        prediction, resources, point_region, view_region, reverse_complement
    )
    tracks = transform_track_arrays(tracks, track_transform)
    x = tracks["x"]
    x_limits = (float(x[0]), float(x[-1]))
    ticks = shared_ticks(x_limits)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=SUMMARY_FIGURE_SIZE_IN,
        gridspec_kw={"height_ratios": [1.1, 1.1, 1.0, 1.0]},
    )
    axes[0].plot(x, tracks["observed_plus"], color="#C44E52", label="observed plus")
    axes[0].plot(x, tracks["observed_minus"], color="#4C72B0", label="observed minus")
    format_track_axis(
        axes[0],
        x,
        f"{exp_id} observed {view_region}",
        track_transform,
        show_title=False,
        show_legend=False,
    )
    apply_shared_ticks(axes[0], ticks)
    axes[1].plot(
        x,
        tracks["predicted_plus"],
        color="#C44E52",
        linestyle="--",
        label="predicted plus",
    )
    axes[1].plot(
        x,
        tracks["predicted_minus"],
        color="#4C72B0",
        linestyle="--",
        label="predicted minus",
    )
    format_track_axis(
        axes[1],
        x,
        f"{exp_id} predicted {view_region}",
        track_transform,
        show_title=False,
        show_legend=False,
    )
    apply_shared_ticks(axes[1], ticks)
    for ax, head in zip(axes[2:], ["profile", "count"]):
        matrix = oriented_logo_matrix(attributions[head], reverse_complement)
        plot_logo_panel(
            ax,
            matrix,
            f"{head} DeepLIFT/SHAP, frequency reference ({logo_region})",
            logo_start,
            logo_end,
            reverse_complement,
            x_limits,
            ticks,
            show_tick_labels=ax is axes[-1],
            show_title=False,
        )
    axes[-1].set_xlabel("Genomic position")
    fig.tight_layout()
    fig.set_size_inches(*SUMMARY_FIGURE_SIZE_IN, forward=True)
    return fig, axes


def deeplift_attributions(
    resources: dict,
    X: torch.Tensor,
    logo_offsets: tuple[int, int],
    n_folds: int,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Compute fold-averaged profile/count DeepLIFT with a soft reference."""
    references = nucleotide_frequency_references(X)
    attributions = {"profile": [], "count": []}
    for fold, path in enumerate(resources["model_paths"][:n_folds]):
        print(f"DeepLIFT fold {fold + 1}/{n_folds}: {path.name}")
        model = torch.load(path, map_location="cpu", weights_only=False).eval()
        wrappers = {"profile": ProfileWrapper(model), "count": CountWrapper(model)}
        for head, wrapper in wrappers.items():
            attr = deep_lift_shap(
                model=wrapper,
                X=X,
                references=references,
                n_shuffles=1,
                batch_size=batch_size,
                hypothetical=True,
                warning_threshold=0.01,
                device=device,
            )
            observed = as_numpy(attr * X)[0, :, logo_offsets[0] : logo_offsets[1]]
            attributions[head].append(observed)
        del model, wrappers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {head: np.mean(values, axis=0) for head, values in attributions.items()}


def oriented_logo_matrix(matrix: np.ndarray, reverse_complement: bool) -> np.ndarray:
    """Reverse-complement a logo matrix when displaying the opposite strand."""
    return reverse_complement_matrix(matrix) if reverse_complement else matrix


def logo_ticks(
    ax,
    logo_start: int,
    logo_end: int,
    reverse_complement: bool = False,
) -> None:
    """Label sequence-logo ticks in genomic coordinates."""
    width = logo_end - logo_start
    positions = np.linspace(0, width - 1, 5, dtype=int)
    labels = [
        logo_end - position if reverse_complement else logo_start + position + 1
        for position in positions
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{value:,}" for value in labels])


def plot_deeplift_logos(
    attributions: dict[str, np.ndarray],
    exp_id: str,
    logo_region: str,
    logo_start: int,
    logo_end: int,
    reverse_complement: bool = False,
):
    """Plot profile/count DeepLIFT logos for the selected logo interval."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True)
    for ax, head in zip(axes, ["profile", "count"]):
        matrix = oriented_logo_matrix(attributions[head], reverse_complement)
        plot_logo_panel(
            ax,
            matrix,
            f"{head} DeepLIFT/SHAP, frequency reference",
            logo_start,
            logo_end,
            reverse_complement,
        )
    fig.suptitle(f"{exp_id} {logo_region}")
    fig.tight_layout()
    return fig, axes


def save_locus_viewer_outputs(
    output_dir: Path,
    prediction: np.ndarray,
    attributions: dict[str, np.ndarray],
    resources: dict,
    exp_id: str,
    point_region: str,
    view_region: str,
    logo_region: str,
    logo_start: int,
    logo_end: int,
    reverse_complement: bool = False,
    track_transform: str | None = None,
) -> None:
    """Save the current viewer figures and arrays for offline inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_locus_summary(
        prediction,
        attributions,
        resources,
        exp_id,
        point_region,
        view_region,
        logo_region,
        logo_start,
        logo_end,
        reverse_complement,
        track_transform,
    )[0].savefig(
        output_dir / "locus_viewer_summary.pdf",
        bbox_inches=None,
        pad_inches=0,
    )
    np.savez_compressed(
        output_dir / "locus_viewer_arrays.npz",
        prediction=prediction,
        profile_deeplift=attributions["profile"],
        count_deeplift=attributions["count"],
        point_region=np.asarray(point_region),
        view_region=np.asarray(view_region),
        logo_region=np.asarray(logo_region),
    )
    plt.close("all")
