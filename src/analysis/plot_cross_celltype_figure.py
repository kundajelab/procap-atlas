#!/usr/bin/env python3
"""Assemble the cross-cell-type prediction supplementary figure.

Five panels, all from files cross_celltype_prediction.py already wrote, so
this does no analysis of its own and can be re-run freely while iterating on
layout -- exactly the same division of labour as plot_figure2.py:

  a   cross_celltype_topk.tsv                     -> draw_topk
  b1  cross_celltype_differential_pairs.tsv        -> draw_differential_tiers
  b2  cross_celltype_differential_ceiling.tsv      -> draw_differential_ceiling
  c   cross_celltype_homogenization_{specific,ubiquitous}.tsv
                                                    -> draw_homogenization x2
  d   cross_celltype_matrix.tsv (+ experiment_config.yaml for tissue groups)
                                                    -> draw_matrix

cross_celltype_prediction.py's own plot_*() functions each open, draw, and
save one standalone figure; this script instead calls the underlying
draw_*(ax, ...) helpers directly against axes carved out of one shared
GridSpec, and never touches the per-panel PDFs those functions still write on
every run -- both outputs coexist so the per-panel files stay available for
manual rearrangement.

Usage:
    python src/analysis/plot_cross_celltype_figure.py \\
        --in-dir figures/cross_celltype_all198
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _biosample_groups import load_group_map  # noqa: E402
from cross_celltype_prediction import (  # noqa: E402
    draw_differential_ceiling,
    draw_differential_tiers,
    draw_homogenization,
    draw_matrix,
    draw_topk,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"


def load_groups(experiments: list[str], biosample_groups: Path | None) -> dict[str, str]:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["experiments"]
    tissue, _ = load_group_map(cfg, biosample_groups, quiet=True)
    return {e: tissue.get(e) for e in experiments}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in-dir", type=Path, default=REPO_ROOT / "figures" / "cross_celltype_all198",
        metavar="DIR", help="directory cross_celltype_prediction.py wrote its "
        "TSVs to (default: figures/cross_celltype_all198)",
    )
    parser.add_argument(
        "--out-stem", type=Path, default=None, metavar="PATH",
        help="output path stem, writes {stem}.{pdf,png} (default: "
        "{in-dir}/cross_celltype_figure)",
    )
    parser.add_argument(
        "--biosample-groups", type=Path, default=None, metavar="PATH",
        help="curated biosample<TAB>group override table, must match whatever "
        "cross_celltype_prediction.py was run with",
    )
    parser.add_argument("--figsize", type=float, nargs=2, default=(14.0, 17.0),
                        metavar=("W", "H"))
    args = parser.parse_args()

    def read(name: str) -> pd.DataFrame:
        path = args.in_dir / name
        if not path.exists():
            print(
                f"ERROR: {path} missing -- rerun cross_celltype_prediction.py "
                f"against --in-dir {args.in_dir} first",
                file=sys.stderr,
            )
            sys.exit(1)
        return pd.read_csv(path, sep="\t")

    topk = read("cross_celltype_topk.tsv")
    diff_pairs = read("cross_celltype_differential_pairs.tsv")
    ceiling = read("cross_celltype_differential_ceiling.tsv")
    homog_specific = read("cross_celltype_homogenization_specific.tsv")
    homog_ubiquitous = read("cross_celltype_homogenization_ubiquitous.tsv")
    matrix = pd.read_csv(args.in_dir / "cross_celltype_matrix.tsv", sep="\t", index_col=0)
    groups = load_groups(list(matrix.index), args.biosample_groups)

    fig = plt.figure(figsize=tuple(args.figsize))
    # Each draw_*() helper was tuned for its own standalone axes size (roughly
    # 4x3in for a/b1/b2, 3.3x2.9in per homogenization sub-panel); giving the
    # composite generous absolute inches per cell -- rather than shrinking
    # fonts to fit -- is what keeps their legends/annotations from colliding.
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.4), hspace=0.65, wspace=0.4)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[1, 0])
    gs_c = gs[1, 1].subgridspec(1, 2, wspace=0.2)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1], sharey=ax_c1)
    ax_d = fig.add_subplot(gs[2, :])

    draw_topk(ax_a, topk)
    draw_differential_tiers(ax_b1, diff_pairs)
    draw_differential_ceiling(ax_b2, ceiling)
    draw_homogenization(ax_c1, homog_specific, "Tissue-specific peaks", annotate_gap=True)
    draw_homogenization(ax_c2, homog_ubiquitous, "Ubiquitous peaks")
    ax_c2.set_ylabel("")
    legend = ax_c2.get_legend()
    if legend is not None:
        legend.remove()
    draw_matrix(ax_d, matrix, groups)

    # Above each panel's own title (loc="left", y~1.0), not overlapping it.
    for ax, label in ((ax_a, "a"), (ax_b1, "b1"), (ax_b2, "b2"), (ax_c1, "c"), (ax_d, "d")):
        ax.text(-0.15, 1.18, label, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom")

    out_stem = args.out_stem or (args.in_dir / "cross_celltype_figure")
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_stem}.{{pdf,png}}", file=sys.stderr)
    print(
        "Per-panel PDFs from cross_celltype_prediction.py are untouched in "
        f"{args.in_dir} for manual rearrangement.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
