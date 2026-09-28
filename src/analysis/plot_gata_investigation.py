"""Supplementary Figure: GATA motif recovery across K562 model variants.

Panel (a): GATA CWM logos from each model variant with seqlet counts.
Panel (b): Peak class composition — grouped bars showing promoter/enhancer/other
           breakdown for XSM peaks, KBX peaks, GC negatives, DHS negatives.

Reads modisco HTML reports to identify GATA patterns and extract CWM logos.
Peak class data comes from compare_k562_peak_classes.py (CSV on cluster).

Usage:
    python src/analysis/plot_gata_investigation.py \
        --modisco-dirs tmp/ENCSR220XSM_count.modisco \
                       tmp/ENCSR261KBX_count.modisco \
                       tmp/ENCSR261KBX_dnase_count.modisco \
                       tmp/ENCSR220XSMtracks_ENCSR261KBXpeaks_count.modisco \
                       tmp/ENCSR261KBXtracks_ENCSR220XSMpeaks_count.modisco \
        --peak-class-csv figures/gata_investigation/peak_class_breakdown.csv
"""

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_LABELS = {
    "ENCSR220XSM_count": "XSM\n(44M, GC neg)",
    "ENCSR261KBX_count": "KBX\n(19M, GC neg)",
    "ENCSR261KBX_dnase_count": "KBX\n(19M, DHS neg)",
    "ENCSR220XSMtracks_ENCSR261KBXpeaks_count": "XSM tracks\nKBX peaks",
    "ENCSR261KBXtracks_ENCSR220XSMpeaks_count": "KBX tracks\nXSM peaks",
}

MODEL_ORDER = [
    "ENCSR220XSM_count",
    "ENCSR261KBXtracks_ENCSR220XSMpeaks_count",
    "ENCSR220XSMtracks_ENCSR261KBXpeaks_count",
    "ENCSR261KBX_count",
    "ENCSR261KBX_dnase_count",
]

PEAK_CLASS_ORDER = [
    "XSM (44M reads, GC neg)",
    "KBX (19M reads, GC neg)",
    "KBX GC negatives",
    "KBX DNase negatives",
]

PEAK_CLASS_LABELS = {
    "XSM (44M reads, GC neg)": "XSM peaks\n(44M reads)",
    "KBX (19M reads, GC neg)": "KBX peaks\n(19M reads)",
    "KBX GC negatives": "GC\nnegatives",
    "KBX DNase negatives": "DHS\nnegatives",
}


class _ModiscoHTMLParser(HTMLParser):
    """Extract pattern name, seqlet count, top JASPAR match, and CWM path."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[tuple[str, str | None]]] = []
        self._current_row: list[tuple[str, str | None]] = []
        self._in_td = False
        self._td_text = ""
        self._td_img: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self._in_td = True
            self._td_text = ""
            self._td_img = None
        if tag == "tr":
            self._current_row = []
        if tag == "img":
            self._td_img = dict(attrs).get("src", "")

    def handle_endtag(self, tag):
        if tag == "td":
            self._in_td = False
            self._current_row.append((self._td_text.strip(), self._td_img))
        if tag == "tr" and self._current_row:
            self.rows.append(self._current_row)

    def handle_data(self, data):
        if self._in_td:
            self._td_text += data


def parse_modisco_report(report_dir: Path) -> list[dict]:
    """Parse a modisco HTML report and return per-pattern info."""
    html_path = report_dir / "motifs.html"
    parser = _ModiscoHTMLParser()
    with open(html_path) as f:
        parser.feed(f.read())

    patterns = []
    for row in parser.rows:
        texts = [c[0] for c in row]
        imgs = [c[1] for c in row]
        if not any("pattern_" in t for t in texts):
            continue
        pattern_name = next(t for t in texts if "pattern_" in t)
        seqlets = next((int(t) for t in texts if t.isdigit()), 0)
        jaspar_match = next(
            (t for t in texts if re.match(r"MA\d+\.\d+", t)), ""
        )
        cwm_fwd_img = next(
            (i for i in imgs if i and "cwm.fwd" in i), None
        )
        cwm_rev_img = next(
            (i for i in imgs if i and "cwm.rev" in i), None
        )
        patterns.append({
            "pattern": pattern_name,
            "seqlets": seqlets,
            "jaspar_match": jaspar_match,
            "cwm_fwd_path": cwm_fwd_img,
            "cwm_rev_path": cwm_rev_img,
        })
    return patterns


def find_gata_pattern(report_dir: Path) -> dict | None:
    """Find the primary GATA pattern in a modisco report."""
    patterns = parse_modisco_report(report_dir)
    gata = [p for p in patterns if "GATA" in p["jaspar_match"]]
    if not gata:
        return None
    return max(gata, key=lambda p: p["seqlets"])


def _match_orientation(
    ref_flat: np.ndarray, fwd_img: np.ndarray, rev_img: np.ndarray | None,
) -> str:
    """Pick fwd or rev CWM image to match a reference orientation.

    Correlates both images against the reference and returns 'fwd' or 'rev'.
    """
    fwd_flat = fwd_img[:, :, :3].flatten().astype(float)
    fwd_flat -= fwd_flat.mean()
    corr_fwd = np.corrcoef(ref_flat, fwd_flat)[0, 1]

    if rev_img is None:
        return "fwd"

    rev_flat = rev_img[:, :, :3].flatten().astype(float)
    rev_flat -= rev_flat.mean()
    corr_rev = np.corrcoef(ref_flat, rev_flat)[0, 1]

    return "fwd" if corr_fwd >= corr_rev else "rev"


def _crop_axes(img: np.ndarray) -> np.ndarray:
    """Crop matplotlib axis labels/ticks from a trimmed_logos CWM PNG.

    These are standardized 300x1000 matplotlib outputs. The y-axis labels
    occupy ~12% on the left and x-axis labels ~15% on the bottom. We crop
    to the logo region using fixed proportions.
    """
    h, w = img.shape[:2]
    left = int(w * 0.12)
    right = int(w * 0.98)
    top = 0
    bottom = int(h * 0.85)
    return img[top:bottom, left:right]


def panel_peak_classes(ax, csv_path: Path):
    """Panel (a): grouped bar chart of peak/negative class composition."""
    df = pd.read_csv(csv_path, index_col=0)

    present = [k for k in PEAK_CLASS_ORDER if k in df.index]
    df = df.loc[present]

    categories = ["promoter", "enhancer", "other"]
    colors = {"promoter": "#4393c3", "enhancer": "#d6604d", "other": "#cccccc"}

    x = np.arange(len(present))
    width = 0.25

    for i, cat in enumerate(categories):
        vals = df[cat].values / df["total"].values * 100
        bars = ax.bar(
            x + (i - 1) * width, vals, width,
            label=cat.capitalize(), color=colors[cat], edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=6,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [PEAK_CLASS_LABELS.get(k, k) for k in present],
        fontsize=7, ha="center",
    )
    ax.set_ylabel("Fraction of regions (%)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)


def panel_gata_logos(axes, modisco_dirs: list[Path]):
    """Panel (b): GATA CWM logos across model variants."""
    dir_lookup = {}
    for d in modisco_dirs:
        stem = d.name.replace(".modisco", "")
        dir_lookup[stem] = d

    ordered = [k for k in MODEL_ORDER if k in dir_lookup]

    # First pass: load all fwd/rev images
    images = {}
    gata_info = {}
    for key in ordered:
        report_dir = dir_lookup[key]
        gata = find_gata_pattern(report_dir)
        gata_info[key] = gata
        if gata and gata["cwm_fwd_path"]:
            fwd_path = report_dir / gata["cwm_fwd_path"].lstrip("./")
            if fwd_path.exists():
                images[key] = {"fwd": imread(str(fwd_path))}
                rev_path_str = gata.get("cwm_rev_path")
                if rev_path_str:
                    rev_path = report_dir / rev_path_str.lstrip("./")
                    if rev_path.exists():
                        images[key]["rev"] = imread(str(rev_path))

    # Use the first model's fwd CWM as orientation reference; align all others
    ref_key = ordered[0] if ordered else None
    ref_flat = None
    if ref_key and ref_key in images:
        ref_img = images[ref_key]["fwd"]
        ref_flat = ref_img[:, :, :3].flatten().astype(float)
        ref_flat -= ref_flat.mean()

    for i, key in enumerate(ordered):
        ax = axes[i]
        gata = gata_info.get(key)

        if key in images:
            if ref_flat is not None:
                choice = _match_orientation(
                    ref_flat, images[key]["fwd"], images[key].get("rev"),
                )
            else:
                choice = "fwd"
            img = images[key][choice]
            img = _crop_axes(img)
            ax.imshow(img, aspect="auto")
            ax.set_title(
                f"{gata['seqlets']:,} seqlets",
                fontsize=7.5, pad=4,
            )
        elif gata:
            ax.text(0.5, 0.5, "image\nnot found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="gray")
        else:
            ax.text(0.5, 0.5, "no GATA\npattern", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="gray")

        ax.set_xlabel(MODEL_LABELS.get(key, key), fontsize=7, labelpad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--modisco-dirs", nargs="+", type=Path, required=True,
        help="Paths to modisco report directories (*.modisco/)",
    )
    parser.add_argument(
        "--peak-class-csv", type=Path, default=None,
        help="CSV from compare_k562_peak_classes.py (omit to skip panel a)",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=REPO_ROOT / "figures" / "gata_investigation" / "fig_gata_investigation.pdf",
    )
    args = parser.parse_args()

    has_peak_classes = args.peak_class_csv and args.peak_class_csv.exists()
    n_models = len(args.modisco_dirs)

    if has_peak_classes:
        fig = plt.figure(figsize=(7.2, 4.5))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[0.8, 1], hspace=0.5)

        gs_logos = gs[0].subgridspec(1, n_models, wspace=0.15)
        logo_axes = [fig.add_subplot(gs_logos[0, i]) for i in range(n_models)]
        panel_gata_logos(logo_axes, args.modisco_dirs)
        logo_axes[0].text(-0.15, 1.15, "a", transform=logo_axes[0].transAxes,
                          fontsize=12, fontweight="bold", va="top")

        ax_bars = fig.add_subplot(gs[1])
        panel_peak_classes(ax_bars, args.peak_class_csv)
        ax_bars.text(-0.08, 1.05, "b", transform=ax_bars.transAxes,
                     fontsize=12, fontweight="bold", va="top")
    else:
        fig = plt.figure(figsize=(7.2, 2.0))
        gs = GridSpec(1, n_models, figure=fig, wspace=0.15)
        logo_axes = [fig.add_subplot(gs[0, i]) for i in range(n_models)]
        panel_gata_logos(logo_axes, args.modisco_dirs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=300)
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
