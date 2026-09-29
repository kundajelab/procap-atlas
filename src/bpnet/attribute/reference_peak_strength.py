"""Predicted peak strength of shuffled vs frequency references at one or more loci.

For each locus, generates many dinucleotide-shuffled sequences and one frequency
reference, predicts on all of them with all fold models, and saves the
distribution of peak strength metrics. Runs locally on MPS/CPU in seconds per
locus.

Usage:
    python src/bpnet/attribute/reference_peak_strength.py \
        --loci chr2:181,680,717 chr11:5,227,002
    python src/bpnet/attribute/reference_peak_strength.py \
        --loci chr2:181,680,717 --n-shuffles 500 --seeds 0,1,2
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

cache_root = Path(os.environ.get("SCRATCH", "/tmp")) / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.io import extract_loci
from tangermeme.predict import predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.locus_viewer import (
    best_device,
    download_model_paths,
    download_reference,
    free_device_memory,
    parse_point,
)
from src.bpnet.attribute.attribute_bpnet import nucleotide_frequency_references

IN_WINDOW = 2114
OUT_WINDOW = 1000
TRIM = (IN_WINDOW - OUT_WINDOW) // 2


def predict_peak_metrics(model, X, batch_size, device):
    """Return per-sequence peak strength metrics as numpy arrays."""
    profile_logits, log_counts = predict(
        model=model, X=X, batch_size=batch_size, device=device,
    )
    profile_logits = np.asarray(profile_logits, dtype=np.float32)
    log_counts = np.asarray(log_counts, dtype=np.float32)

    n = len(profile_logits)
    flat = profile_logits.reshape(n, -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    shifted = centered - centered.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

    counts = np.exp(log_counts).sum(axis=1)
    count_scaled = probs * counts[:, None]

    return {
        "max_prob": probs.max(axis=1),
        "counts": counts,
        "max_signal": count_scaled.max(axis=1),
        "profile_score": (centered * probs).sum(axis=1),
    }


def run_locus(X, model_paths, seeds, n_shuffles, batch_size, device, verbose):
    """Predict on shuffled and frequency inputs for one locus across all folds."""
    n_seeds = len(seeds)
    n_folds = len(model_paths)
    total_shuffles = n_seeds * n_shuffles
    metrics = ["max_prob", "counts", "max_signal", "profile_score"]

    shuffled_results = {m: np.zeros((total_shuffles, n_folds)) for m in metrics}
    frequency_results = {m: np.zeros(n_folds) for m in metrics}
    genomic_results = {m: np.zeros(n_folds) for m in metrics}

    all_shuffled = []
    for seed in seeds:
        refs = dinucleotide_shuffle(X.cpu(), n=n_shuffles, random_state=seed)
        all_shuffled.append(refs[0])
    all_shuffled = torch.cat(all_shuffled, dim=0).float()

    freq_ref = nucleotide_frequency_references(X, n=1).squeeze(1)

    for fold, model_path in enumerate(model_paths):
        if verbose:
            print(f"  Fold {fold}: {model_path.name}")
        model = torch.load(model_path, map_location="cpu", weights_only=False).eval()

        gm = predict_peak_metrics(model, X, batch_size, device)
        for m in metrics:
            genomic_results[m][fold] = gm[m][0]

        fm = predict_peak_metrics(model, freq_ref, batch_size, device)
        for m in metrics:
            frequency_results[m][fold] = fm[m][0]

        sm = predict_peak_metrics(model, all_shuffled, batch_size, device)
        for m in metrics:
            shuffled_results[m][:, fold] = sm[m]

        del model
        free_device_memory()

    return {
        "shuffled": {m: v.mean(axis=1) for m, v in shuffled_results.items()},
        "frequency": {m: v.mean() for m, v in frequency_results.items()},
        "genomic": {m: v.mean() for m, v in genomic_results.items()},
        "shuffled_per_fold": shuffled_results,
    }


def plot_locus_distributions(results, locus_label, seeds, n_shuffles, output_dir, fmt):
    """Plot peak strength distributions for one locus."""
    shuf = results["shuffled"]
    freq = results["frequency"]
    gen = results["genomic"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))

    panels = [
        ("max_prob", "Maximum predicted probability", False),
        ("counts", "Predicted total counts", True),
        ("max_signal", "Maximum count-scaled signal", True),
    ]

    for ax, (metric, title, use_log) in zip(axes, panels):
        vals = shuf[metric]
        freq_val = freq[metric]
        gen_val = gen[metric]

        if use_log:
            vals = np.log10(np.maximum(vals, 1e-6))
            freq_val = np.log10(max(freq_val, 1e-6))
            gen_val = np.log10(max(gen_val, 1e-6))
            xlabel = f"log₁₀({title.lower()})"
        else:
            xlabel = title

        ax.hist(vals, bins=40, density=True, alpha=0.55, color="#4C72B0",
                label=f"Shuffled (n={len(vals)})")
        ax.axvline(freq_val, color="#DD8452", linewidth=1.5, linestyle="-",
                   label="Frequency ref")
        ax.axvline(gen_val, color="#55A868", linewidth=1.5, linestyle="--",
                   label="Genomic")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False)

    safe_label = locus_label.replace(":", "_").replace(",", "")
    fig.suptitle(f"{locus_label}  ({len(seeds)} seeds × {n_shuffles} shuffles)",
                 fontsize=9)
    fig.tight_layout()
    path = output_dir / f"reference_peak_strength_{safe_label}.{fmt}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--loci", nargs="+", required=True,
        help="one or more 1-based point regions, e.g. chr2:181,680,717",
    )
    parser.add_argument("--experiment", default="ENCSR220XSM")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--fasta", type=Path, default=None)
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        default=REPO_ROOT / "figures" / "reference_stability",
    )
    parser.add_argument("--format", choices=("pdf", "png", "svg"), default="pdf")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    device = best_device()
    work_dir = Path(os.environ.get("SCRATCH", ".cache")) / "procap_atlas_locus_viewer"
    work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Seeds: {seeds}, Shuffles/seed: {args.n_shuffles}")
    print(f"Total shuffled references per locus: {len(seeds) * args.n_shuffles}")

    if args.fasta is not None:
        fasta = args.fasta.expanduser()
    else:
        fasta = download_reference(work_dir)

    if args.model_dir is not None:
        model_paths = [
            args.model_dir / f"{args.experiment}.fold{fold}.torch"
            for fold in range(args.folds)
        ]
    else:
        model_paths = download_model_paths(args.experiment, args.folds)

    all_results = {}
    for locus in args.loci:
        print(f"\n{locus}")
        t0 = time.perf_counter()

        chrom, center = parse_point(locus)
        loci_df = pd.DataFrame({"chrom": [chrom], "start": [center], "end": [center + 1]})
        X = extract_loci(loci_df, sequences=str(fasta), in_window=IN_WINDOW,
                         ignore=["N", "n"]).float()
        if len(X) != 1:
            print(f"  WARNING: could not extract {locus}, skipping")
            continue

        results = run_locus(
            X, model_paths, seeds, args.n_shuffles,
            args.batch_size, device, args.verbose,
        )
        plot_locus_distributions(
            results, locus, seeds, args.n_shuffles,
            args.output_dir, args.format,
        )
        all_results[locus] = results
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

    if len(all_results) > 1:
        out_path = args.output_dir / "reference_peak_strength_all.npz"
        save_dict = {}
        for locus, results in all_results.items():
            safe = locus.replace(":", "_").replace(",", "")
            for ref_type in ("shuffled", "frequency", "genomic"):
                for metric, value in results[ref_type].items():
                    save_dict[f"{safe}_{ref_type}_{metric}"] = np.asarray(value)
        np.savez_compressed(out_path, **save_dict)
        print(f"\nSaved all results to {out_path}")


if __name__ == "__main__":
    main()
