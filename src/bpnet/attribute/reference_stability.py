"""Measure reference peak strength and attribution stability across seeds.

Generates data for two supplementary analyses:
1. Distribution of predicted peak strength for shuffled vs frequency model inputs
2. Correlation between shuffled-reference peak strength variation and attribution
   cosine instability across seeds

Usage:
    python src/bpnet/attribute/reference_stability.py -e ENCSR220XSM
    python src/bpnet/attribute/reference_stability.py -e ENCSR220XSM --attributions
    python src/bpnet/attribute/reference_stability.py -e ENCSR220XSM --attributions --attribution-subset 2000
"""

import argparse
import gc
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from bpnetlite.bpnet import CountWrapper, ProfileWrapper, _ProfileLogitScaling
from bpnetlite.chrombpnet import _Exp, _Log
from tangermeme.deep_lift_shap import _nonlinear, deep_lift_shap
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.io import extract_loci
from tangermeme.predict import predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bpnet.attribute.attribute_bpnet import nucleotide_frequency_references

CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
FASTA = str(REPO_ROOT / "data" / "hg38.fa")
BLACKLIST = str(REPO_ROOT / "data" / "hg38.blacklist.bed.gz")
IN_WINDOW = 2114
OUT_WINDOW = 1000
TRIM = (IN_WINDOW - OUT_WINDOW) // 2

DEEPLIFT_NONLINEAR_OPS = {
    _ProfileLogitScaling: _nonlinear,
    _Log: _nonlinear,
    _Exp: _nonlinear,
}


def predict_metrics(model, X, batch_size, device):
    """Predict on X and return peak strength metrics as numpy arrays."""
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

    max_prob = probs.max(axis=1)
    counts = np.exp(log_counts).sum(axis=1)
    count_scaled = probs * counts[:, None]
    max_signal = count_scaled.max(axis=1)
    argmax_flat = count_scaled.argmax(axis=1)

    return {
        "max_prob": max_prob,
        "counts": counts,
        "max_signal": max_signal,
        "argmax_flat": argmax_flat,
    }


def reverse_complement_ohe(seq):
    """Reverse-complement a (4, L) one-hot encoded sequence."""
    return seq[[3, 2, 1, 0], ::-1].copy()


def accumulate_pwm(pwm, n_pwm, X_flat, argmax_flat, half_window):
    """Add base composition at each prediction argmax to the PWM accumulator."""
    window = 2 * half_window + 1
    for i in range(len(X_flat)):
        strand = int(argmax_flat[i]) // OUT_WINDOW
        out_pos = int(argmax_flat[i]) % OUT_WINDOW
        in_pos = out_pos + TRIM
        start = in_pos - half_window
        end = start + window
        if start < 0 or end > IN_WINDOW:
            continue
        seq = X_flat[i].numpy() if isinstance(X_flat[i], torch.Tensor) else X_flat[i]
        bases = seq[:, start:end]
        if strand == 1:
            bases = reverse_complement_ohe(bases)
        pwm += bases
        n_pwm[0] += 1


def cosine_similarity_batch(a, b):
    """Pairwise cosine similarity between rows of two (N, ...) arrays."""
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    dot = (a_flat * b_flat).sum(axis=1)
    norm = np.sqrt((a_flat ** 2).sum(axis=1) * (b_flat ** 2).sum(axis=1))
    return dot / np.maximum(norm, 1e-12)


def run_predictions(
    X, model_paths, seeds, n_shuffles, batch_size, chunk_size,
    pwm_half_window, device, verbose,
):
    """Phase 1: predict on genomic, shuffled, and frequency inputs for all peaks."""
    n_peaks = len(X)
    n_seeds = len(seeds)
    n_folds = len(model_paths)
    pwm_window = 2 * pwm_half_window + 1

    shuffled_max_prob = np.zeros((n_peaks, n_seeds, n_shuffles), dtype=np.float64)
    shuffled_counts = np.zeros((n_peaks, n_seeds, n_shuffles), dtype=np.float64)
    shuffled_max_signal = np.zeros((n_peaks, n_seeds, n_shuffles), dtype=np.float64)

    frequency_max_prob = np.zeros(n_peaks, dtype=np.float64)
    frequency_counts = np.zeros(n_peaks, dtype=np.float64)
    frequency_max_signal = np.zeros(n_peaks, dtype=np.float64)

    genomic_max_prob = np.zeros(n_peaks, dtype=np.float64)
    genomic_counts = np.zeros(n_peaks, dtype=np.float64)
    genomic_max_signal = np.zeros(n_peaks, dtype=np.float64)

    pwm = np.zeros((4, pwm_window), dtype=np.float64)
    n_pwm = [0]

    for fold, model_path in enumerate(model_paths):
        t0 = time.perf_counter()
        model = torch.load(model_path, map_location="cpu", weights_only=False).eval()
        if verbose:
            print(f"  Fold {fold}: loaded {model_path.name}")

        for chunk_start in range(0, n_peaks, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_peaks)
            X_chunk = X[chunk_start:chunk_end].float()
            cs = chunk_end - chunk_start

            gm = predict_metrics(model, X_chunk, batch_size, device)
            genomic_max_prob[chunk_start:chunk_end] += gm["max_prob"]
            genomic_counts[chunk_start:chunk_end] += gm["counts"]
            genomic_max_signal[chunk_start:chunk_end] += gm["max_signal"]

            freq = nucleotide_frequency_references(X_chunk, n=1).squeeze(1)
            fm = predict_metrics(model, freq, batch_size, device)
            frequency_max_prob[chunk_start:chunk_end] += fm["max_prob"]
            frequency_counts[chunk_start:chunk_end] += fm["counts"]
            frequency_max_signal[chunk_start:chunk_end] += fm["max_signal"]

            for si, seed in enumerate(seeds):
                shuffled = dinucleotide_shuffle(
                    X_chunk.cpu(), n=n_shuffles, random_state=seed,
                )
                flat = shuffled.reshape(-1, 4, IN_WINDOW).float()
                sm = predict_metrics(model, flat, batch_size, device)
                mp = sm["max_prob"].reshape(cs, n_shuffles)
                ct = sm["counts"].reshape(cs, n_shuffles)
                ms = sm["max_signal"].reshape(cs, n_shuffles)
                shuffled_max_prob[chunk_start:chunk_end, si] += mp
                shuffled_counts[chunk_start:chunk_end, si] += ct
                shuffled_max_signal[chunk_start:chunk_end, si] += ms

                accumulate_pwm(pwm, n_pwm, flat, sm["argmax_flat"], pwm_half_window)

            if verbose and (chunk_start // chunk_size) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"    chunk {chunk_start}-{chunk_end}/{n_peaks}  "
                    f"({elapsed:.0f}s elapsed)"
                )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            print(f"  Fold {fold} done in {time.perf_counter() - t0:.0f}s")

    shuffled_max_prob = (shuffled_max_prob / n_folds).astype(np.float32)
    shuffled_counts = (shuffled_counts / n_folds).astype(np.float32)
    shuffled_max_signal = (shuffled_max_signal / n_folds).astype(np.float32)
    frequency_max_prob = (frequency_max_prob / n_folds).astype(np.float32)
    frequency_counts = (frequency_counts / n_folds).astype(np.float32)
    frequency_max_signal = (frequency_max_signal / n_folds).astype(np.float32)
    genomic_max_prob = (genomic_max_prob / n_folds).astype(np.float32)
    genomic_counts = (genomic_counts / n_folds).astype(np.float32)
    genomic_max_signal = (genomic_max_signal / n_folds).astype(np.float32)

    argmax_pwm = pwm / np.maximum(pwm.sum(axis=0, keepdims=True), 1)

    return {
        "shuffled_max_prob": shuffled_max_prob,
        "shuffled_counts": shuffled_counts,
        "shuffled_max_signal": shuffled_max_signal,
        "frequency_max_prob": frequency_max_prob,
        "frequency_counts": frequency_counts,
        "frequency_max_signal": frequency_max_signal,
        "genomic_max_prob": genomic_max_prob,
        "genomic_counts": genomic_counts,
        "genomic_max_signal": genomic_max_signal,
        "argmax_pwm": argmax_pwm,
        "n_pwm": np.int64(n_pwm[0]),
    }


def run_attributions(
    X, model_paths, seeds, n_shuffles, subset_indices,
    batch_size, attr_batch_size, chunk_size, device, verbose,
):
    """Phase 2: compute inter-seed attribution cosine on a peak subset."""
    n_subset = len(subset_indices)
    n_seeds = len(seeds)
    n_folds = len(model_paths)
    n_pairs = n_seeds * (n_seeds - 1) // 2
    seed_pairs = list(combinations(range(n_seeds), 2))

    cosine_profile = np.zeros((n_subset, n_pairs, n_folds), dtype=np.float32)
    cosine_count = np.zeros((n_subset, n_pairs, n_folds), dtype=np.float32)
    subset_max_prob = np.zeros((n_subset, n_seeds, n_folds), dtype=np.float32)
    subset_max_signal = np.zeros((n_subset, n_seeds, n_folds), dtype=np.float32)

    X_subset = X[subset_indices].float()

    for fold, model_path in enumerate(model_paths):
        t0 = time.perf_counter()
        model = torch.load(model_path, map_location="cpu", weights_only=False).eval()
        profile_wrapper = ProfileWrapper(model)
        count_wrapper = CountWrapper(model)
        if verbose:
            print(f"  Attribution fold {fold}: {model_path.name}")

        for chunk_start in range(0, n_subset, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_subset)
            X_chunk = X_subset[chunk_start:chunk_end].float()
            cs = chunk_end - chunk_start

            attrs_profile = []
            attrs_count = []
            for si, seed in enumerate(seeds):
                refs = dinucleotide_shuffle(
                    X_chunk.cpu(), n=n_shuffles, random_state=seed,
                )

                flat = refs.reshape(-1, 4, IN_WINDOW).float()
                sm = predict_metrics(model, flat, batch_size, device)
                subset_max_prob[chunk_start:chunk_end, si, fold] = (
                    sm["max_prob"].reshape(cs, n_shuffles).mean(axis=1)
                )
                subset_max_signal[chunk_start:chunk_end, si, fold] = (
                    sm["max_signal"].reshape(cs, n_shuffles).mean(axis=1)
                )

                ap = deep_lift_shap(
                    model=profile_wrapper,
                    X=X_chunk,
                    references=refs,
                    hypothetical=True,
                    batch_size=attr_batch_size,
                    warning_threshold=0.01,
                    additional_nonlinear_ops=DEEPLIFT_NONLINEAR_OPS,
                    device=device,
                )
                attrs_profile.append((ap * X_chunk).detach().cpu().numpy())

                ac = deep_lift_shap(
                    model=count_wrapper,
                    X=X_chunk,
                    references=refs,
                    hypothetical=True,
                    batch_size=attr_batch_size,
                    warning_threshold=0.01,
                    additional_nonlinear_ops=DEEPLIFT_NONLINEAR_OPS,
                    device=device,
                )
                attrs_count.append((ac * X_chunk).detach().cpu().numpy())

            for pair_idx, (si, sj) in enumerate(seed_pairs):
                cosine_profile[chunk_start:chunk_end, pair_idx, fold] = (
                    cosine_similarity_batch(attrs_profile[si], attrs_profile[sj])
                )
                cosine_count[chunk_start:chunk_end, pair_idx, fold] = (
                    cosine_similarity_batch(attrs_count[si], attrs_count[sj])
                )

            if verbose:
                elapsed = time.perf_counter() - t0
                print(
                    f"    attr chunk {chunk_start}-{chunk_end}/{n_subset}  "
                    f"({elapsed:.0f}s)"
                )

        del model, profile_wrapper, count_wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            print(f"  Attribution fold {fold} done in {time.perf_counter() - t0:.0f}s")

    return {
        "cosine_inter_seed_profile": cosine_profile,
        "cosine_inter_seed_count": cosine_count,
        "subset_max_prob_per_seed": subset_max_prob,
        "subset_max_signal_per_seed": subset_max_signal,
        "subset_indices": np.asarray(subset_indices),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e", "--experiment", default="ENCSR220XSM",
        help="experiment accession (default: ENCSR220XSM, K562 PRO-cap)",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-shuffles", type=int, default=20)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--attr-batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--pwm-half-window", type=int, default=50)
    parser.add_argument(
        "--attributions", action="store_true",
        help="enable analysis 2: attribution stability correlation",
    )
    parser.add_argument(
        "--attribution-subset", type=int, default=2000,
        help="number of peaks to use for attribution analysis",
    )
    parser.add_argument("--max-peaks", type=int, default=0, help="0 = all peaks")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "analysis" / "reference_stability",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    if args.experiment not in config["experiments"]:
        print(f"Error: {args.experiment} not in config", file=sys.stderr)
        sys.exit(1)

    exp = config["experiments"][args.experiment]
    peaks_path = str(REPO_ROOT / exp["processed"]["filtered_peaks"])
    if not Path(peaks_path).exists():
        print(f"Error: peaks not found: {peaks_path}", file=sys.stderr)
        sys.exit(1)

    with open(CHROM_SPLITS_PATH) as f:
        splits = yaml.safe_load(f)
    n_folds = len(splits["folds"])
    model_dir = REPO_ROOT / "models" / "bpnet" / args.experiment
    model_paths = [
        model_dir / f"{args.experiment}.fold{fold}.torch"
        for fold in range(n_folds)
    ]
    for p in model_paths:
        if not p.exists():
            print(f"Error: model not found: {p}", file=sys.stderr)
            sys.exit(1)

    loci = pd.read_csv(
        peaks_path, sep="\t", usecols=[0, 1, 2], header=None,
        names=["chrom", "start", "end"], dtype={"chrom": str},
    )
    all_chroms = list(
        sum((v for v in splits["folds"].values()), [])
    )
    X = extract_loci(
        loci=loci, sequences=FASTA, chroms=all_chroms,
        in_window=IN_WINDOW, verbose=args.verbose,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        exclusion_lists=[BLACKLIST],
    )
    if args.max_peaks > 0 and len(X) > args.max_peaks:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(X), args.max_peaks, replace=False)
        keep.sort()
        X = X[keep]
    n_peaks = len(X)

    print(f"Experiment: {args.experiment}")
    print(f"Peaks: {n_peaks}, Seeds: {seeds}, Shuffles/seed: {args.n_shuffles}")
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Phase 1: predictions ===")
    t_start = time.perf_counter()
    pred_results = run_predictions(
        X, model_paths, seeds, args.n_shuffles, args.batch_size,
        args.chunk_size, args.pwm_half_window, device, args.verbose,
    )
    print(f"Phase 1 done in {time.perf_counter() - t_start:.0f}s")

    save_dict = {
        **pred_results,
        "seeds": np.asarray(seeds),
        "n_shuffles": np.int64(args.n_shuffles),
        "experiment": np.asarray(args.experiment),
        "n_peaks": np.int64(n_peaks),
    }

    if args.attributions:
        subset_size = min(args.attribution_subset, n_peaks)
        rng = np.random.default_rng(0)
        subset_indices = np.sort(rng.choice(n_peaks, subset_size, replace=False))

        print(f"\n=== Phase 2: attributions ({subset_size} peaks) ===")
        t_start = time.perf_counter()
        attr_results = run_attributions(
            X, model_paths, seeds, args.n_shuffles, subset_indices,
            args.batch_size, args.attr_batch_size, args.chunk_size,
            device, args.verbose,
        )
        print(f"Phase 2 done in {time.perf_counter() - t_start:.0f}s")
        save_dict.update(attr_results)

    out_path = args.output_dir / f"{args.experiment}.npz"
    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
