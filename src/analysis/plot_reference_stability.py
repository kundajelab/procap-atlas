"""Plot reference peak strength distributions and stability correlations.

Reads outputs from reference_stability.py and generates supplementary note panels:
  - Panel A: Distribution of predicted peak strength (shuffled vs frequency inputs)
  - Panel B: Distribution of predicted total counts (shuffled vs frequency)
  - Panel C: Scatter of peak strength variation vs inter-seed attribution cosine

Usage:
    python src/analysis/plot_reference_stability.py \
        analysis/reference_stability/ENCSR220XSM.npz
    python src/analysis/plot_reference_stability.py \
        analysis/reference_stability/ENCSR220XSM.npz \
        -o figures/reference_stability/ --format pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


SHUFFLED_COLOR = "#4C72B0"
FREQUENCY_COLOR = "#DD8452"
GENOMIC_COLOR = "#55A868"


def load_results(path):
    data = dict(np.load(path, allow_pickle=True))
    for key in ("experiment", "seeds"):
        if key in data and data[key].ndim == 0:
            data[key] = data[key].item()
    return data


def plot_peak_strength_distributions(data, output_dir, fmt):
    """Histograms of max predicted probability and counts for shuffled vs frequency."""
    shuf_mp = data["shuffled_max_prob"].ravel()
    freq_mp = data["frequency_max_prob"].ravel()
    genomic_mp = data["genomic_max_prob"].ravel()

    shuf_ct = data["shuffled_counts"].ravel()
    freq_ct = data["frequency_counts"].ravel()
    genomic_ct = data["genomic_counts"].ravel()

    shuf_ms = data["shuffled_max_signal"].ravel()
    freq_ms = data["frequency_max_signal"].ravel()
    genomic_ms = data["genomic_max_signal"].ravel()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    panels = [
        (shuf_mp, freq_mp, genomic_mp, "Maximum predicted probability", False),
        (shuf_ct, freq_ct, genomic_ct, "Predicted total counts", True),
        (shuf_ms, freq_ms, genomic_ms, "Maximum count-scaled signal", True),
    ]

    for ax, (shuf, freq, gen, title, use_log) in zip(axes, panels):
        if use_log:
            shuf = np.log10(np.maximum(shuf, 1e-6))
            freq = np.log10(np.maximum(freq, 1e-6))
            gen = np.log10(np.maximum(gen, 1e-6))
            xlabel = f"log₁₀({title.lower()})"
        else:
            xlabel = title

        lo = min(np.percentile(freq, 1), np.percentile(shuf, 1))
        hi = max(np.percentile(gen, 99), np.percentile(shuf, 99))
        bins = np.linspace(lo, hi, 80)

        ax.hist(
            shuf, bins=bins, density=True, alpha=0.5,
            color=SHUFFLED_COLOR, label="Shuffled",
        )
        ax.hist(
            freq, bins=bins, density=True, alpha=0.5,
            color=FREQUENCY_COLOR, label="Frequency",
        )
        ax.axvline(
            np.median(gen), color=GENOMIC_COLOR, linewidth=1.2,
            linestyle="--", label="Genomic (median)",
        )
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False)

    fig.tight_layout()
    path = output_dir / f"peak_strength_distributions.{fmt}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_argmax_pwm(data, output_dir, fmt):
    """Sequence logo of the PWM at shuffled-reference prediction argmax."""
    pwm = data["argmax_pwm"]
    n_pwm = int(data["n_pwm"])
    half = pwm.shape[1] // 2

    ic = np.log2(4) + (pwm * np.log2(np.maximum(pwm, 1e-12))).sum(axis=0)

    fig, ax = plt.subplots(figsize=(8, 2))
    x = np.arange(pwm.shape[1]) - half
    colors = {"A": "#009E73", "C": "#0072B2", "G": "#F0E442", "T": "#D55E00"}
    bases = "ACGT"

    for pos in range(pwm.shape[1]):
        heights = pwm[:, pos] * ic[pos]
        order = np.argsort(heights)
        y_offset = 0.0
        for bi in order:
            h = heights[bi]
            if h < 0.001:
                continue
            ax.bar(
                x[pos], h, bottom=y_offset, width=0.9,
                color=colors[bases[bi]], edgecolor="none",
            )
            y_offset += h

    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0, 2)
    ax.set_xlabel("Position relative to predicted argmax", fontsize=8)
    ax.set_ylabel("Information (bits)", fontsize=8)
    ax.set_title(f"Dinucleotide-shuffled reference argmax PWM (n={n_pwm:,})", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    path = output_dir / f"argmax_pwm.{fmt}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_peakiness_instability(data, output_dir, fmt):
    """Scatter: peak strength variation across seeds vs inter-seed attribution cosine."""
    if "cosine_inter_seed_profile" not in data:
        print("No attribution data found, skipping peakiness-instability plot")
        return

    cosine_p = data["cosine_inter_seed_profile"]
    cosine_c = data["cosine_inter_seed_count"]
    sub_mp = data["subset_max_prob_per_seed"]
    sub_ms = data["subset_max_signal_per_seed"]

    mean_cosine_p = cosine_p.mean(axis=(1, 2))
    mean_cosine_c = cosine_c.mean(axis=(1, 2))

    mp_sd = sub_mp.mean(axis=2).std(axis=1)
    ms_sd = sub_ms.mean(axis=2).std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, x_vals, y_vals, xlabel, ylabel, head_label in [
        (
            axes[0], mp_sd, mean_cosine_p,
            "SD of max probability across seeds",
            "Mean inter-seed profile cosine",
            "Profile head",
        ),
        (
            axes[1], ms_sd, mean_cosine_c,
            "SD of max count-scaled signal across seeds",
            "Mean inter-seed count cosine",
            "Count head",
        ),
    ]:
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        xv, yv = x_vals[valid], y_vals[valid]
        rho, pval = stats.spearmanr(xv, yv)

        ax.scatter(
            xv, yv, s=4, alpha=0.3, color=SHUFFLED_COLOR,
            edgecolors="none", rasterized=True,
        )
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(head_label, fontsize=9)
        ax.tick_params(labelsize=7)

        pval_str = f"p < 10⁻{int(-np.log10(pval))}" if pval < 1e-4 else f"p = {pval:.2g}"
        ax.text(
            0.97, 0.05, f"ρ = {rho:.3f}\n{pval_str}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    path = output_dir / f"peakiness_instability.{fmt}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="path to .npz from reference_stability.py")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="output directory (default: figures/reference_stability/)",
    )
    parser.add_argument("--format", choices=("pdf", "png", "svg"), default="pdf")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    output_dir = args.output_dir or Path("figures") / "reference_stability"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(args.input)
    experiment = data.get("experiment", args.input.stem)
    print(f"Plotting results for {experiment}")

    plot_peak_strength_distributions(data, output_dir, args.format)
    plot_argmax_pwm(data, output_dir, args.format)
    plot_peakiness_instability(data, output_dir, args.format)

    print("Done")


if __name__ == "__main__":
    main()
