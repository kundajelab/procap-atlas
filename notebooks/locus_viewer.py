"""Helpers for the lightweight BPNet locus viewer notebook."""

from __future__ import annotations

import gc
import gzip
import os
import shutil
import warnings
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
from src.bpnet.attribute.attribute_bpnet import (
    DEEPLIFT_NONLINEAR_OPS,
    nucleotide_frequency_references,
)
from huggingface_hub import hf_hub_download
from pyfaidx import Fasta
from tangermeme.annotate import annotate_seqlets
from tangermeme.io import extract_loci, read_meme
from tangermeme.plot import plot_logo
from tangermeme.predict import predict
from tangermeme.seqlet import recursive_seqlets

from tangermeme.deep_lift_shap import deep_lift_shap
from src.bpnet.attribute.locus_diagnostics import (
    as_numpy,
    genomic_offsets,
    reverse_complement_matrix,
    reverse_complement_tracks,
)
from src.modeling.profile import count_scaled_profile

MODEL_REPO_ID = "adamyhe/procap-atlas"
TRACK_REPO_ID = "adamyhe/procap-atlas-tracks"
METADATA_REPO_ID = "adamyhe/procap-atlas-metadata"
REFERENCE_FASTA_URL = "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
IN_WINDOW = 2114
OUT_WINDOW = 1000
# Checked into git (unlike experiment_config.yaml, which is fetched from
# HuggingFace because it can be regenerated), so it is resolved off __file__
# rather than cwd -- this module always lives at <repo_root>/notebooks/, but
# the *process* cwd is not reliably the repo root. A bare "configs/n_reads.txt"
# would resolve fine after the Colab setup cell's clone-and-chdir, but Jupyter
# on Sherlock/OnDemand opens with cwd set to the notebook's own directory
# (notebooks/), not the repo root, which would silently miss the file.
N_READS_PATH = Path(__file__).resolve().parent.parent / "configs" / "n_reads.txt"
POINTS_PER_INCH = 72
SUMMARY_FIGURE_SIZE_PT = (570, 120)
SUMMARY_FIGURE_SIZE_IN = tuple(value / POINTS_PER_INCH for value in SUMMARY_FIGURE_SIZE_PT)
SUMMARY_LABEL_SIZE = 6
SUMMARY_TICK_SIZE = 5
TRACK_SIGNAL_COLOR = "#4C72B0"
TRACK_SIGNAL_LINEWIDTH = 0.7
DEFAULT_MOTIF_PATH = Path("data/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt")
SUMMARY_SUBPLOT_ADJUST = {
    "left": 0.055,
    "right": 0.995,
    "bottom": 0.18,
    "top": 0.98,
    "hspace": 0.06,
}


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


def best_device() -> str:
    """The fastest available torch device: CUDA, then Apple Metal, then CPU.

    MPS is a real speedup over CPU on Apple silicon and gives attributions
    matching CPU to float32 rounding (~1e-7 on this path). It has no float64,
    but nothing here needs it on-device -- the fold accumulators are numpy,
    on the host.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_device_memory() -> None:
    """Drop a fold model's cached allocations between folds."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def cpm_scale_for(exp_id: str, n_reads_path: Path = N_READS_PATH) -> float | None:
    """1e6 / total_reads for `exp_id`, or None (with a warning) if unavailable.

    Every track this notebook reads or predicts -- observed/predicted PRO-cap
    coverage, and counts-head DeepLIFT -- is on a raw-count scale that a
    library's sequencing depth moves directly. Comparing two experiments (a
    differential-locus panel) without this is comparing depth as much as
    biology; two libraries at similar depth make it look harmless, but that
    is a coincidence of which experiments were picked, not something to rely
    on. Profile-head DeepLIFT is exempt: it explains a softmax output, a
    shape/probability distribution that is depth-independent by
    construction, so multiplying it by a depth-based factor would be
    meaningless rather than merely unnecessary.

    Returns None rather than raising so a locus can still be viewed (in raw
    units, with a clear warning) if the read-depth table is missing --
    matches the graceful-degradation style already used for
    SEQLET_MOTIF_PATH.
    """
    if not n_reads_path.exists():
        warnings.warn(
            f"{n_reads_path} not found; showing raw (non-CPM) values. Depth "
            "differences between experiments are not accounted for."
        )
        return None
    table = pd.read_csv(n_reads_path, sep="\t")
    row = table.loc[table["experiment"] == exp_id]
    if not len(row):
        warnings.warn(
            f"{exp_id} has no entry in {n_reads_path}; showing raw "
            "(non-CPM) values."
        )
        return None
    return 1e6 / float(row["total_reads"].iloc[0])


def region_inputs(
    resources: dict, regions: list[str]
) -> tuple[pd.DataFrame, torch.Tensor]:
    """Extract model inputs for many regions at once.

    Returns a frame of one row per region (`region`, `chrom`, `start`, `end`,
    `center`, `logo_start`, `logo_end`, `logo_offsets`) aligned row-for-row
    with the returned `(N, 4, IN_WINDOW)` batch, so downstream results index
    the same way.

    `extract_loci` is called once for the whole set rather than per region,
    and the batch then feeds `ensemble_predictions`/`deeplift_attributions_batch`,
    which load each fold model once for the entire batch instead of once per
    region. That is the difference between 7 model loads and 7N.
    """
    rows = []
    for region in regions:
        chrom, start, end = parse_interval(region)
        _, logo_start, logo_end, offsets = logo_offsets_for_region(region)
        rows.append({
            "region": region,
            "chrom": chrom,
            "start": start,
            "end": end,
            "center": interval_center(start, end),
            "logo_start": logo_start,
            "logo_end": logo_end,
            "logo_offsets": offsets,
        })
    records = pd.DataFrame(rows)
    loci = pd.DataFrame({
        "chrom": records["chrom"],
        "start": records["center"],
        "end": records["center"] + 1,
    })
    X = extract_loci(
        loci,
        sequences=str(resources["fasta"]),
        in_window=IN_WINDOW,
        ignore=["N", "n"],
    )
    if len(X) != len(records):
        raise ValueError(
            f"extract_loci returned {len(X)} of {len(records)} regions; one "
            "or more fall off a chromosome end or contain N"
        )
    return records, X.float()


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


def ensemble_predictions(
    resources: dict,
    X: torch.Tensor,
    n_folds: int,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Fold-averaged count-scaled profiles for a batch of inputs, `(N, ...)`.

    Folds are the outer loop and regions the inner one, so each fold model is
    loaded exactly once for the whole batch. Peak memory is unchanged -- still
    one model resident at a time -- but a sweep over N regions costs 7 loads
    rather than 7N.

    Averaging happens on logits and log-counts *before* `count_scaled_profile`,
    matching the single-region path: averaging count-scaled profiles instead
    would weight folds by their predicted depth.
    """
    logits_sum = None
    log_counts_sum = None
    for fold, path in enumerate(resources["model_paths"][:n_folds]):
        print(f"Predicting fold {fold + 1}/{n_folds} "
              f"({len(X)} region(s)): {path.name}")
        model = torch.load(path, map_location="cpu", weights_only=False).eval()
        logits, log_counts = predict(
            model=model, X=X, batch_size=batch_size, device=device
        )
        logits = as_numpy(logits).astype(np.float32)
        log_counts = as_numpy(log_counts).astype(np.float32)
        if logits_sum is None:
            logits_sum = np.zeros_like(logits, dtype=np.float64)
            log_counts_sum = np.zeros_like(log_counts, dtype=np.float64)
        logits_sum += logits
        log_counts_sum += log_counts
        del model
        free_device_memory()
    mean_logits = logits_sum / n_folds
    mean_log_counts = log_counts_sum / n_folds
    return count_scaled_profile(mean_logits, mean_log_counts).astype(np.float32)


def ensemble_prediction(
    resources: dict,
    X: torch.Tensor,
    n_folds: int,
    device: str,
) -> np.ndarray:
    """Fold-averaged count-scaled profile for a single-region input."""
    return ensemble_predictions(resources, X, n_folds, device, batch_size=1)[0]


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
    cpm_scale: float | None = None,
) -> dict[str, np.ndarray]:
    """Return observed and predicted plus/minus tracks for the view interval.

    `cpm_scale` (from `cpm_scale_for`) is applied to all four tracks, since
    both observed coverage and predicted coverage are on the same raw-count
    scale. `None` (the default) leaves values raw, for callers that have not
    opted in.
    """
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
    if cpm_scale is not None:
        predicted_plus = predicted_plus * cpm_scale
        predicted_minus = predicted_minus * cpm_scale
        observed_plus = observed_plus * cpm_scale
        observed_minus = observed_minus * cpm_scale
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


def clip_track_values(values: np.ndarray, clip: float | None) -> np.ndarray:
    """Clip signal tracks to a symmetric display range without changing raw arrays."""
    if clip is None:
        return values
    if clip <= 0:
        raise ValueError("track_value_clip must be positive or None")
    return np.clip(values, -clip, clip)


def clip_track_arrays(
    tracks: dict[str, np.ndarray],
    track_value_clip: float | None = None,
) -> dict[str, np.ndarray]:
    """Return tracks with optional display clipping applied to signal arrays."""
    return {
        key: (
            clip_track_values(value, track_value_clip)
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
    track_value_clip: float | None = None,
    show_title: bool = True,
    show_legend: bool = True,
    cpm_scaled: bool = False,
) -> None:
    """Apply shared formatting to one plus/minus signal track axis.

    `cpm_scaled` only changes the label -- the caller is responsible for
    having actually multiplied the plotted values by `cpm_scale_for(...)`.
    A mismatch between the two would mislabel raw counts as RPM.
    """
    ax.set_xlim(x[0], x[-1])
    if show_title:
        ax.set_title(title)
    ylabel = "PRO-cap signal (RPM)" if cpm_scaled else "PRO-cap signal"
    if track_value_clip is not None:
        ylabel = f"{ylabel} (clipped at {track_value_clip:g})"
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


def apply_compact_summary_axis_style(ax, show_x_labels: bool = False) -> None:
    """Keep summary figure labels legible at journal-column dimensions."""
    ax.set_title("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="major", labelsize=SUMMARY_TICK_SIZE, pad=1)
    ax.tick_params(axis="x", labelbottom=show_x_labels)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_fontsize(SUMMARY_TICK_SIZE)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def motif_names_from_meme(motif_path: Path) -> list[str]:
    """Return MEME motif names in the order used by tangermeme annotation."""
    return list(read_meme(str(motif_path)).keys())


def _to_numpy_array(value) -> np.ndarray:
    """Convert torch/tangermeme outputs to numpy arrays."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _seqlet_sequence_tensor(
    X: torch.Tensor,
    logo_offsets: tuple[int, int],
    reverse_complement: bool,
) -> torch.Tensor:
    """Return one cropped OHE tensor aligned to displayed logo coordinates."""
    sequence = as_numpy(X.float())[0, :, logo_offsets[0] : logo_offsets[1]]
    if reverse_complement:
        sequence = reverse_complement_matrix(sequence)
    return torch.tensor(sequence[None], dtype=torch.float32)


def call_and_annotate_seqlets(
    attributions: dict[str, np.ndarray],
    X: torch.Tensor,
    logo_offsets: tuple[int, int],
    motif_path: Path | str | None = DEFAULT_MOTIF_PATH,
    reverse_complement: bool = False,
    threshold: float = 0.01,
    additional_flanks: int = 2,
    min_seqlet_len: int = 4,
    max_seqlet_len: int = 25,
    n_nearest: int = 1,
    use_abs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Call seqlets and optionally annotate them with MEME motifs.

    Seqlets are called on per-position projected attributions for the cropped
    logo window. Motif annotation is skipped when ``motif_path`` is ``None`` or
    absent.
    """
    motif_path = Path(motif_path) if motif_path is not None else None
    motif_names = (
        motif_names_from_meme(motif_path)
        if motif_path and motif_path.exists()
        else []
    )
    sequence = _seqlet_sequence_tensor(X, logo_offsets, reverse_complement)
    annotations: dict[str, pd.DataFrame] = {}
    for head, matrix in attributions.items():
        display_matrix = oriented_logo_matrix(matrix, reverse_complement)
        projected = display_matrix.sum(axis=0)
        caller_values = np.abs(projected) if use_abs else projected
        seqlets = recursive_seqlets(
            torch.tensor(caller_values[None], dtype=torch.float32),
            threshold=threshold,
            additional_flanks=additional_flanks,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
        ).copy()
        if seqlets.empty:
            annotations[head] = seqlets
            continue
        seqlets["start"] = seqlets["start"].clip(lower=0, upper=len(projected))
        seqlets["end"] = seqlets["end"].clip(lower=0, upper=len(projected))
        seqlets["head"] = head
        seqlets["signed_attribution"] = [
            projected[int(start) : int(end)].sum()
            for start, end in zip(seqlets["start"], seqlets["end"])
        ]
        seqlets["strand"] = "."
        seqlets["score"] = seqlets["attribution"]
        seqlets["motif_idx"] = -1
        seqlets["motif_name"] = "seqlet"
        seqlets["motif_pvalue"] = np.nan
        if motif_names:
            motif_idxs, pvals = annotate_seqlets(
                sequence,
                seqlets,
                str(motif_path),
                n_nearest=n_nearest,
            )
            motif_idxs = _to_numpy_array(motif_idxs)
            pvals = _to_numpy_array(pvals)
            primary_idxs = motif_idxs[:, 0]
            seqlets["motif_idx"] = primary_idxs
            seqlets["motif_name"] = [
                motif_names[int(idx)] if int(idx) >= 0 else "unannotated"
                for idx in primary_idxs
            ]
            seqlets["motif_pvalue"] = pvals[:, 0]
        annotations[head] = seqlets
    return annotations


def plot_tracks(
    prediction: np.ndarray,
    resources: dict,
    exp_id: str,
    point_region: str,
    view_region: str,
    reverse_complement: bool = False,
    track_value_clip: float | None = None,
    cpm_scale: float | None = None,
):
    """Plot observed and predicted PRO-cap signal on separate y scales."""
    tracks = track_arrays(
        prediction, resources, point_region, view_region, reverse_complement,
        cpm_scale,
    )
    tracks = clip_track_arrays(tracks, track_value_clip)
    x = tracks["x"]
    ticks = shared_ticks((float(x[0]), float(x[-1])))
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.2), sharex=True)
    axes[0].plot(
        x,
        tracks["observed_plus"],
        color=TRACK_SIGNAL_COLOR,
        linewidth=TRACK_SIGNAL_LINEWIDTH,
        label="observed plus",
    )
    axes[0].plot(
        x,
        tracks["observed_minus"],
        color=TRACK_SIGNAL_COLOR,
        linewidth=TRACK_SIGNAL_LINEWIDTH,
        label="observed minus",
    )
    format_track_axis(axes[0], x, f"{exp_id} observed {view_region}",
                      track_value_clip, cpm_scaled=cpm_scale is not None)
    axes[1].plot(
        x,
        tracks["predicted_plus"],
        color=TRACK_SIGNAL_COLOR,
        linewidth=TRACK_SIGNAL_LINEWIDTH,
        linestyle="--",
        label="predicted plus",
    )
    axes[1].plot(
        x,
        tracks["predicted_minus"],
        color=TRACK_SIGNAL_COLOR,
        linewidth=TRACK_SIGNAL_LINEWIDTH,
        linestyle="--",
        label="predicted minus",
    )
    format_track_axis(axes[1], x, f"{exp_id} predicted {view_region}",
                      track_value_clip, cpm_scaled=cpm_scale is not None)
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
    def shift_x(values):
        if reverse_complement:
            return logo_end - values
        return logo_start + 1 + values

    for collection in ax.collections:
        for path in collection.get_paths():
            vertices = path.vertices
            vertices[:, 0] = shift_x(vertices[:, 0])
    for patch in ax.patches:
        if hasattr(patch, "get_x") and hasattr(patch, "set_x"):
            x = patch.get_x()
            if reverse_complement and hasattr(patch, "get_width"):
                patch.set_x(float(logo_end - x - patch.get_width()))
            else:
                patch.set_x(float(shift_x(x)))
    for line in ax.lines:
        line.set_xdata(shift_x(np.asarray(line.get_xdata())))
    for text in ax.texts:
        x, y = text.get_position()
        text.set_position((float(shift_x(x)), y))


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
    seqlet_annotations: pd.DataFrame | None = None,
    value_clip: float | None = None,
) -> None:
    """Draw one DeepLIFT logo panel with genomic coordinate ticks.

    `value_clip` fixes the y-axis to `(-value_clip, value_clip)` instead of
    autoscaling to this panel's own data. It does not touch `matrix` -- a
    letter taller than the clip is drawn in full and cut off by the axes
    boundary (standard matplotlib clip-to-axes behaviour), not numerically
    truncated -- so unlike `clip_track_values` this never changes what was
    computed, only what window of it is visible. That matters for comparing
    two independently-rendered logo panels (e.g. two experiments at the same
    locus): each autoscales to its own max by default, which visually erases
    a real difference in attribution magnitude between them unless both are
    pinned to the same range.
    """
    plot_kwargs = {}
    if seqlet_annotations is not None and not seqlet_annotations.empty:
        plot_kwargs = {
            "annotations": seqlet_annotations,
            "score_key": "attribution",
        }
    plot_logo(torch.tensor(matrix, dtype=torch.float32), ax=ax, **plot_kwargs)
    if show_title:
        ax.set_title(title)
    if x_limits is None:
        logo_ticks(ax, logo_start, logo_end, reverse_complement)
    else:
        shift_logo_to_genomic_axis(ax, logo_start, logo_end, reverse_complement)
        ax.set_xlim(*x_limits)
        if ticks is not None:
            apply_shared_ticks(ax, ticks, show_labels=show_tick_labels)
    if value_clip is not None:
        if value_clip <= 0:
            raise ValueError("value_clip must be positive or None")
        ax.set_ylim(-value_clip, value_clip)
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
    track_value_clip: float | None = None,
    seqlet_annotations: dict[str, pd.DataFrame] | None = None,
    cpm_scale: float | None = None,
    logo_value_clip: dict[str, float] | None = None,
):
    """Stack observed tracks, predicted tracks, and DeepLIFT logos in one figure.

    `cpm_scale` (from `cpm_scale_for`) scales the observed/predicted coverage
    tracks and the counts-DeepLIFT logo. It is applied only to the locally
    drawn copies, not to `prediction`/`attributions` themselves, so those stay
    in raw units for `save_locus_viewer_outputs` to persist unchanged.
    Profile DeepLIFT explains a softmax output (depth-independent by
    construction) and is never scaled.

    `logo_value_clip` is `{"profile": clip_or_None, "count": clip_or_None}`,
    fixing each DeepLIFT panel's y-axis instead of autoscaling it to this
    figure's own data -- see `plot_logo_panel`. Each head needs its own value
    since profile and count DeepLIFT are on unrelated scales.
    """
    logo_value_clip = logo_value_clip or {}
    tracks = track_arrays(
        prediction, resources, point_region, view_region, reverse_complement,
        cpm_scale,
    )
    tracks = clip_track_arrays(tracks, track_value_clip)
    x = tracks["x"]
    x_limits = (float(x[0]), float(x[-1]))
    ticks = shared_ticks(x_limits)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=SUMMARY_FIGURE_SIZE_IN,
        gridspec_kw={"height_ratios": [1.1, 1.1, 1.0, 1.0]},
    )
    axes[0].plot(
        x,
        tracks["observed_plus"],
        color=TRACK_SIGNAL_COLOR,
        linewidth=TRACK_SIGNAL_LINEWIDTH,
        label="observed plus",
    )
    axes[0].plot(
        x,
        tracks["observed_minus"],
        color=TRACK_SIGNAL_COLOR,
        linewidth=TRACK_SIGNAL_LINEWIDTH,
        label="observed minus",
    )
    format_track_axis(
        axes[0],
        x,
        f"{exp_id} observed {view_region}",
        track_value_clip,
        show_title=False,
        show_legend=False,
        cpm_scaled=cpm_scale is not None,
    )
    apply_shared_ticks(axes[0], ticks)
    axes[1].plot(
        x,
        tracks["predicted_plus"],
        color=TRACK_SIGNAL_COLOR,
        linewidth=TRACK_SIGNAL_LINEWIDTH,
        linestyle="--",
        label="predicted plus",
    )
    axes[1].plot(
        x,
        tracks["predicted_minus"],
        color=TRACK_SIGNAL_COLOR,
        linewidth=TRACK_SIGNAL_LINEWIDTH,
        linestyle="--",
        label="predicted minus",
    )
    format_track_axis(
        axes[1],
        x,
        f"{exp_id} predicted {view_region}",
        track_value_clip,
        show_title=False,
        show_legend=False,
        cpm_scaled=cpm_scale is not None,
    )
    apply_shared_ticks(axes[1], ticks)
    for ax, head in zip(axes[2:], ["profile", "count"]):
        matrix = oriented_logo_matrix(attributions[head], reverse_complement)
        if head == "count" and cpm_scale is not None:
            matrix = matrix * cpm_scale
        annotations = None
        if seqlet_annotations is not None:
            annotations = seqlet_annotations.get(head)
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
            seqlet_annotations=annotations,
            value_clip=logo_value_clip.get(head),
        )
    for ax in axes[:-1]:
        apply_compact_summary_axis_style(ax)
    apply_compact_summary_axis_style(axes[-1], show_x_labels=True)
    axes[-1].set_xlabel("Genomic position", fontsize=SUMMARY_LABEL_SIZE, labelpad=2)
    fig.subplots_adjust(**SUMMARY_SUBPLOT_ADJUST)
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
    return {
        head: values[0]
        for head, values in deeplift_attributions_batch(
            resources, X, logo_offsets, n_folds, batch_size, device
        ).items()
    }


def deeplift_attributions_batch(
    resources: dict,
    X: torch.Tensor,
    logo_offsets: tuple[int, int],
    n_folds: int,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Fold-averaged profile/count DeepLIFT for a batch, `{head: (N, 4, W)}`.

    Folds outer, regions inner, so each fold model is loaded once for the
    whole batch rather than once per region.

    `logo_offsets` is shared across the batch: every input is centred on its
    own region, so a fixed window on the input maps to each region's own
    coordinates. Regions of differing widths therefore need separate calls --
    use `region_inputs` and group by width, or pass the widest window and
    slice afterwards.
    """
    attributions = {"profile": [], "count": []}
    for fold, path in enumerate(resources["model_paths"][:n_folds]):
        print(f"DeepLIFT fold {fold + 1}/{n_folds} "
              f"({len(X)} region(s)): {path.name}")
        model = torch.load(path, map_location="cpu", weights_only=False).eval()
        wrappers = {"profile": ProfileWrapper(model), "count": CountWrapper(model)}
        for head, wrapper in wrappers.items():
            attr = deep_lift_shap(
                model=wrapper,
                X=X,
                # Passed as a *callable*, not a prebuilt tensor.
                # tangermeme only one-hot-validates Tensor references, so a
                # soft PFM tensor is rejected outright ("references must be
                # one-hot encoded"); a callable is invoked internally and
                # skips that check. This is what c3cbb07 fixed in production,
                # which is why the local soft-reference fork of
                # deep_lift_shap could be deleted.
                references=nucleotide_frequency_references,
                n_shuffles=1,
                batch_size=batch_size,
                hypothetical=True,
                warning_threshold=0.01,
                # BPNet's wrappers contain _ProfileLogitScaling/_Log/_Exp,
                # which DeepLIFT cannot decompose unless told they are
                # nonlinearities. attribute_bpnet.py passes the same map, so
                # notebook attributions match the production ones rather than
                # silently differing.
                additional_nonlinear_ops=DEEPLIFT_NONLINEAR_OPS,
                device=device,
            )
            observed = as_numpy(attr * X)[:, :, logo_offsets[0] : logo_offsets[1]]
            attributions[head].append(observed)
        del model, wrappers
        free_device_memory()
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
    seqlet_annotations: dict[str, pd.DataFrame] | None = None,
    cpm_scale: float | None = None,
    logo_value_clip: dict[str, float] | None = None,
):
    """Plot profile/count DeepLIFT logos for the selected logo interval.

    `cpm_scale` scales the drawn count-head matrix only, matching
    `plot_locus_summary`; `attributions` itself is left untouched.

    `logo_value_clip` is `{"profile": clip_or_None, "count": clip_or_None}`,
    i.e. one value per head rather than a single shared value -- profile and
    count DeepLIFT are on unrelated scales (see `cpm_scale_for`), so a shared
    clip would either do nothing on one head or hide everything on the other.
    """
    logo_value_clip = logo_value_clip or {}
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True)
    for ax, head in zip(axes, ["profile", "count"]):
        matrix = oriented_logo_matrix(attributions[head], reverse_complement)
        if head == "count" and cpm_scale is not None:
            matrix = matrix * cpm_scale
        annotations = None
        if seqlet_annotations is not None:
            annotations = seqlet_annotations.get(head)
        plot_logo_panel(
            ax,
            matrix,
            f"{head} DeepLIFT/SHAP, frequency reference",
            logo_start,
            logo_end,
            reverse_complement,
            seqlet_annotations=annotations,
            value_clip=logo_value_clip.get(head),
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
    track_value_clip: float | None = None,
    seqlet_annotations: dict[str, pd.DataFrame] | None = None,
    cpm_scale: float | None = None,
    logo_value_clip: dict[str, float] | None = None,
) -> None:
    """Save the current viewer figures and arrays for offline inspection.

    `prediction` and `attributions` are saved to `locus_viewer_arrays.npz` in
    their original (raw) units regardless of `cpm_scale`/`logo_value_clip` --
    only the rendered figure is scaled/clipped -- matching
    `clip_track_arrays`, which also never mutates the arrays it is given. Raw
    values stay reproducible even when the figure is display-scaled.
    """
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
        track_value_clip,
        seqlet_annotations,
        cpm_scale,
        logo_value_clip,
    )[0].savefig(
        output_dir / "locus_viewer_summary.pdf",
        bbox_inches=None,
        pad_inches=0,
    )
    if seqlet_annotations:
        seqlets = pd.concat(seqlet_annotations.values(), ignore_index=True)
        seqlets.to_csv(output_dir / "locus_viewer_seqlets.tsv", sep="\t", index=False)
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
