#!/usr/bin/env python3
"""Assemble Figure 2: the motif lexicon, its tissue structure, and what the
motifs actually are.

Three panels, all from files other scripts already wrote, so this does no
analysis of its own and can be re-run freely while iterating on layout:

  a  plot_motif_rarefaction.py  -> motif_rarefaction_{head}.tsv
  b  motif_group_concentration.py --save-null-draws
                                 -> ..._{level}_nulldraws.tsv + _swapnull.tsv
  c  select_motif_exemplars.py   -> motif_exemplars_{head}_*.tsv
                                    plus cluster-average CWMs from the h5

Panel c is built as a stack of labelled rows rather than one grid, so the
profile head drops in as an extra row once its compendium is built without any
redesign: pass --profile-exemplars and --profile-h5. That row is the point of
the panel -- profile-head motifs should be universal core promoter elements
against the count head's lineage-restricted TFs, which is the two-lexicon
claim in visual form.

Note that profile-head core promoter elements (Inr, TATA, DPE) are absent from
JASPAR2026 entirely, so they arrive in the *unmatched* class and need
select_motif_exemplars.py --include-unmatched plus hand-naming via
make_annotation_scaffold.py. The TF-matched default that is right for the count
head will silently drop every interesting profile motif.

Usage:
    python src/analysis/plot_figure2.py --head count \\
        --modisco-h5 compendium/motifcompendium_count_cluster_averages.h5
    python src/analysis/plot_figure2.py --head count --modisco-h5 auto \\
        --profile-exemplars figures/motif_atlas/motif_exemplars_profile_restricted.tsv \\
        --profile-h5 motifcompendium/bpnet/motifcompendium_profile_cluster_averages.h5
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motif_redundancy import trim_cwm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet"
DEFAULT_DIR = REPO_ROOT / "figures" / "motif_atlas"

SCHEME_STYLE = {
    "diverse": dict(color="#1b7837", lw=1.8, ls="-", label="diverse (round-robin tissues)"),
    "uniform": dict(color="#777777", lw=1.2, ls="-", label="uniform (random)"),
    "redundant": dict(color="#b2182b", lw=1.8, ls="--", label="redundant (one tissue first)"),
}

# Tissue labels as they should read in a figure, not as _biosample_groups.py
# spells them internally.
GROUP_LABEL = {
    "blood_immune": "blood / immune",
    "stem_ipsc": "stem / iPSC",
    "neural": "neural",
    "gi_tract": "GI tract",
    "liver_biliary": "liver",
    "metastatic_carcinoma": "metastatic",
    "heart": "heart",
    "pancreas": "pancreas",
    "reproductive": "reproductive",
    "hek": "HEK293",
}


def load_cwm(h5_path: Path, cluster_id: int, posneg: str = "pos") -> np.ndarray | None:
    """A cluster's average contribution-score matrix, as (length, 4).

    MotifCompendium writes `{pos,neg}_patterns/{cluster_final}/contrib_scores`
    with shape (length, 4). trim_cwm expects (4, length), so callers must
    transpose -- getting that backwards silently reduces the per-position
    magnitude to four numbers and trims to nonsense.
    """
    group = f"{posneg}_patterns"
    with h5py.File(h5_path, "r") as f:
        if group not in f or str(cluster_id) not in f[group]:
            return None
        return np.asarray(f[group][str(cluster_id)]["contrib_scores"][:])


def trimmed_cwm(
    h5_path: Path, cluster_id: int, posneg: str = "pos",
    threshold: float = 0.3, min_len: int = 8, pad: int = 1,
) -> pd.DataFrame | None:
    """Trimmed CWM as a logomaker-ready frame, or None if the cluster is absent."""
    cwm = load_cwm(h5_path, cluster_id, posneg)
    if cwm is None:
        return None
    start, end = trim_cwm(cwm.T, threshold=threshold, min_len=min_len)
    start, end = max(0, start - pad), min(len(cwm), end + pad)
    return pd.DataFrame(cwm[start:end], columns=["A", "C", "G", "T"])


def panel_rarefaction(ax, curves: pd.DataFrame, mark_k: int = 5) -> None:
    all_cls = curves[curves["motif_class"] == "__all__"]
    total = float(all_cls["n_clusters_total"].iloc[0])

    for scheme, style in SCHEME_STYLE.items():
        sub = all_cls[all_cls["scheme"] == scheme].sort_values("k")
        if not len(sub):
            continue
        ax.plot(sub["k"], sub["mean"], **style)
        if {"lo", "hi"} <= set(sub.columns):
            ax.fill_between(sub["k"], sub["lo"], sub["hi"],
                            color=style["color"], alpha=0.13, lw=0)

    ax.axhline(total, color="black", lw=0.8, ls=":", zorder=1)
    ax.text(ax.get_xlim()[1], total, f" {total:.0f}", va="center", ha="left",
            fontsize=7, clip_on=False)

    uni = all_cls[all_cls["scheme"] == "uniform"].sort_values("k")
    at_k = uni[uni["k"] == mark_k]
    if len(at_k):
        y = float(at_k["mean"].iloc[0])
        ax.plot([mark_k], [y], "o", color="#333333", ms=4, zorder=5)
        ax.annotate(
            f"{mark_k} experiments\n{y / total:.0%} of lexicon",
            xy=(mark_k, y), xytext=(mark_k + 12, y - total * 0.13),
            fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6, color="#333333"),
        )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=total * 1.08)
    ax.set_xlabel("Experiments sampled")
    ax.set_ylabel("Motifs recovered")
    ax.set_title("a   Lexicon growth", loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)


def panel_concentration(
    ax, draws: pd.DataFrame, swap: pd.DataFrame, motif_class: str = "TF-matched",
    biosample_swap: pd.DataFrame | None = None,
) -> None:
    d = draws[draws["motif_class"] == motif_class]["n_single_group"].to_numpy()
    row = swap[swap["motif_class"] == motif_class].iloc[0]
    obs = int(row["obs_single_group"])

    bins = np.arange(d.min() - 0.5, max(d.max(), obs) + 1.5, 1.0)
    ax.hist(d, bins=bins, color="#bbbbbb", edgecolor="white", lw=0.3,
            label=f"degree-preserving null\n(n={len(d)} permutations)")
    ax.axvline(obs, color="#b2182b", lw=2, zorder=5)
    # Directly beside the line rather than on an arrow: the arrow's tail
    # landed on the legend, which sits upper-left because the null mode does.
    ax.text(
        obs - 0.8, ax.get_ylim()[1] * 0.97, f"observed\n{obs}",
        color="#b2182b", fontsize=8, fontweight="bold", ha="right", va="top",
    )

    # Against the swap null's own mean, not the exact Poisson-binomial
    # expectation (8.50 -> 5.3x). Both are correct and they differ, so the
    # label has to say which null it means or it contradicts the text.
    null_mean = float(row["null_single_group_mean"])
    parts = [
        f"{obs} vs {null_mean:.1f} expected",
        f"{obs / null_mean:.1f}× vs degree-preserving null",
        f"p < {1.0 / (len(d) + 1):.3f}",
        f"tissue concentration {row['swap_concentration']:.2f}",
    ]
    if biosample_swap is not None and len(biosample_swap):
        brow = biosample_swap[biosample_swap["motif_class"] == motif_class]
        if len(brow):
            parts.append(
                f"biosample concentration {brow.iloc[0]['swap_concentration']:.2f}"
            )
    ax.text(0.97, 0.42, "\n".join(parts), transform=ax.transAxes, ha="right",
            va="top", fontsize=7.5)

    ax.set_xlabel("Motifs confined to a single tissue group")
    ax.set_ylabel("Permutations")
    ax.set_title("b   Discovery is lineage-confined", loc="left",
                 fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)


def _logo_grid(fig, spec, rows, h5_path, subtitle, label_fn, trim_kwargs,
               per_row=6, label_fontsize=6.2):
    """A wrapped grid of logos inside `spec`, labelled down the left edge.

    Wraps rather than using one long row because logos stay legible when tiny:
    10-15 per category fits, which is enough to show a lexicon rather than a
    handful of picked examples.
    """
    import logomaker

    n = len(rows)
    n_sub = max(1, -(-n // per_row))
    inner = GridSpecFromSubplotSpec(
        n_sub, per_row, subplot_spec=spec, wspace=0.30, hspace=1.05
    )
    drew = 0
    for i, r in enumerate(rows.itertuples()):
        ax = fig.add_subplot(inner[i // per_row, i % per_row])
        posneg = getattr(r, "posneg", "pos") or "pos"
        df = trimmed_cwm(h5_path, int(r.cluster_final), posneg, **trim_kwargs)
        if df is None:
            ax.text(0.5, 0.5, f"cl {r.cluster_final}\nnot in h5", ha="center",
                    va="center", fontsize=5, color="#b2182b",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        logomaker.Logo(df, ax=ax, shade_below=0.0, fade_below=0.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.set_title(label_fn(r), fontsize=label_fontsize, pad=1.2,
                     linespacing=1.2)
        drew += 1
        if i == 0:
            ax.text(-0.12, 0.5, subtitle, transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=7, color="#555555")
    return drew


def panel_exemplars(fig, spec, ubiquitous, restricted, h5_path,
                    profile_rows=None, profile_h5=None, trim_kwargs=None,
                    n_ubiquitous=12, n_restricted=12, per_row=6):
    trim_kwargs = trim_kwargs or {}
    rows = []
    if profile_rows is not None and len(profile_rows) and profile_h5:
        rows.append(("profile head: initiation shape", profile_rows.head(n_restricted),
                     profile_h5, lambda r: str(r.jaspar_name)))
    rows.append((
        "count head: ubiquitous",
        ubiquitous.head(n_ubiquitous),
        h5_path,
        lambda r: f"{r.jaspar_name}\n{int(r.prevalence)} exp",
    ))
    rows.append((
        "count head: lineage-restricted",
        restricted.head(n_restricted),
        h5_path,
        lambda r: (
            f"{r.jaspar_name}\n"
            f"{GROUP_LABEL.get(str(r.sole_group), str(r.sole_group))}"
            f", {int(r.prevalence)}"
        ),
    ))

    # Height per category in proportion to how many sub-rows it needs, so a
    # 15-motif category is not squeezed into the same band as a 5-motif one.
    sub_rows = [max(1, -(-len(df) // per_row)) for _, df, _, _ in rows]
    inner = GridSpecFromSubplotSpec(
        len(rows), 1, subplot_spec=spec, hspace=0.55, height_ratios=sub_rows
    )
    total = 0
    for i, (subtitle, df, h5, label_fn) in enumerate(rows):
        total += _logo_grid(fig, inner[i, 0], df, h5, subtitle, label_fn,
                            trim_kwargs, per_row=per_row)
    return total


def save_panel(draw, path_stem: Path, figsize, exts=("pdf", "png")) -> list[str]:
    """Render one panel into its own figure.

    Panels are written separately as well as composited because the composite
    is for judging the story and the separate files are what get hand-aligned
    in a vector editor. Text stays as text in the PDF (no rasterization), so
    fonts remain editable; logomaker glyphs come through as paths, which is
    what a vector editor wants anyway.
    """
    fig = plt.figure(figsize=figsize)
    draw(fig)
    written = []
    for ext in exts:
        path = f"{path_stem}.{ext}"
        fig.savefig(path, dpi=400, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def save_individual_logos(
    rows, h5_path, out_dir: Path, prefix: str, trim_kwargs, figsize=(1.5, 1.0)
) -> list[str]:
    """One small PDF per motif logo, unlabelled, for free rearrangement."""
    import logomaker

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for r in rows.itertuples():
        posneg = getattr(r, "posneg", "pos") or "pos"
        df = trimmed_cwm(h5_path, int(r.cluster_final), posneg, **trim_kwargs)
        if df is None:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        logomaker.Logo(df, ax=ax, shade_below=0.0, fade_below=0.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        name = str(r.jaspar_name).replace("::", "-").replace("/", "-")
        stem = out_dir / f"{prefix}_cl{int(r.cluster_final)}_{name}"
        for ext in ("pdf", "png"):
            path = f"{stem}.{ext}"
            fig.savefig(path, dpi=400, bbox_inches="tight", transparent=True)
            written.append(path)
        plt.close(fig)
    return written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--head", default="count", choices=["profile", "count"])
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_DIR, metavar="DIR",
                        help="where the other scripts wrote their tables")
    parser.add_argument("--modisco-h5", type=Path, required=True, metavar="PATH",
                        help="motifcompendium_{head}_cluster_averages.h5, or "
                             "'auto' to look under motifcompendium/bpnet/")
    parser.add_argument("--group-level", default="tissue",
                        choices=["tissue", "biosample"])
    parser.add_argument("--motif-class", default="TF-matched")
    parser.add_argument("--profile-exemplars", type=Path, default=None, metavar="PATH")
    parser.add_argument("--profile-h5", type=Path, default=None, metavar="PATH")
    parser.add_argument("--trim-threshold", type=float, default=0.3, metavar="F")
    parser.add_argument("--min-trim-len", type=int, default=8, metavar="BP")
    parser.add_argument("--n-ubiquitous", type=int, default=12, metavar="N")
    parser.add_argument("--n-restricted", type=int, default=12, metavar="N")
    parser.add_argument("--logos-per-row", type=int, default=6, metavar="N",
                        help="wrap each category's logos at this width "
                             "(default: 6)")
    parser.add_argument("--mark-k", type=int, default=5, metavar="K")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7.4, 6.2),
                        metavar=("W", "H"), help="inches (default: 7.4 6.2)")
    parser.add_argument("--out-stem", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--no-split-panels", action="store_true",
        help="skip the per-panel files; by default each panel is also written "
             "on its own for hand-alignment in a vector editor",
    )
    parser.add_argument(
        "--split-logos", action="store_true",
        help="also write one small transparent PDF per motif logo, unlabelled, "
             "so panel c can be rearranged freely",
    )
    args = parser.parse_args()

    h5_path = args.modisco_h5
    if str(h5_path) == "auto":
        h5_path = MC_DIR / f"motifcompendium_{args.head}_cluster_averages.h5"

    needed = {
        "curves": args.in_dir / f"motif_rarefaction_{args.head}.tsv",
        "draws": args.in_dir / (
            f"motif_concentration_{args.head}_{args.group_level}_nulldraws.tsv"
        ),
        "swap": args.in_dir / (
            f"motif_concentration_{args.head}_{args.group_level}_swapnull.tsv"
        ),
        "ubiquitous": args.in_dir / f"motif_exemplars_{args.head}_ubiquitous.tsv",
        "restricted": args.in_dir / f"motif_exemplars_{args.head}_restricted.tsv",
    }
    missing = {k: v for k, v in needed.items() if not v.exists()}
    if missing or not Path(h5_path).exists():
        for k, v in missing.items():
            print(f"ERROR: missing {k}: {v}", file=sys.stderr)
        if not Path(h5_path).exists():
            print(f"ERROR: missing cluster-average h5: {h5_path}", file=sys.stderr)
        print(
            "\nRun, in order:\n"
            "  plot_motif_rarefaction.py --head HEAD --min-cluster-experiments 2\n"
            "  motif_group_concentration.py --head HEAD --group-level tissue "
            "--save-null-draws\n"
            "  select_motif_exemplars.py --head HEAD --logo-paths ...",
            file=sys.stderr,
        )
        sys.exit(1)

    tables = {k: pd.read_csv(v, sep="\t") for k, v in needed.items()}
    bio_swap_path = args.in_dir / f"motif_concentration_{args.head}_biosample_swapnull.tsv"
    bio_swap = pd.read_csv(bio_swap_path, sep="\t") if bio_swap_path.exists() else None
    profile_rows = (
        pd.read_csv(args.profile_exemplars, sep="\t")
        if args.profile_exemplars and args.profile_exemplars.exists()
        else None
    )
    if args.profile_exemplars and profile_rows is None:
        print(f"WARNING: {args.profile_exemplars} not found; omitting the "
              "profile row", file=sys.stderr)

    fig = plt.figure(figsize=tuple(args.figsize))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.15],
                  hspace=0.78, wspace=0.26,
                  left=0.09, right=0.965, top=0.94, bottom=0.04)

    panel_rarefaction(fig.add_subplot(gs[0, 0]), tables["curves"], args.mark_k)
    panel_concentration(
        fig.add_subplot(gs[0, 1]), tables["draws"], tables["swap"],
        args.motif_class, bio_swap,
    )

    sub = gs[1, :].subgridspec(1, 1)
    n_logos = panel_exemplars(
        fig, sub[0, 0], tables["ubiquitous"], tables["restricted"], Path(h5_path),
        profile_rows=profile_rows, profile_h5=args.profile_h5,
        trim_kwargs=dict(threshold=args.trim_threshold, min_len=args.min_trim_len),
        n_ubiquitous=args.n_ubiquitous, n_restricted=args.n_restricted,
        per_row=args.logos_per_row,
    )
    # Anchored to the bottom row's own extent. A hardcoded y collided with
    # panel a's x-axis label as soon as the height ratios changed.
    box = gs[1, :].get_position(fig)
    fig.text(
        box.x0 - 0.075, box.y1 + 0.075,
        "c   The two lexicons" if profile_rows is not None
        else "c   Ubiquitous vs lineage-restricted motifs",
        fontweight="bold", fontsize=10, ha="left", va="bottom",
    )

    stem = args.out_stem or (args.in_dir / f"figure2_{args.head}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=400, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)

    if not args.no_split_panels:
        w, h = args.figsize
        written = []
        written += save_panel(
            lambda f: panel_rarefaction(f.add_subplot(111), tables["curves"],
                                        args.mark_k),
            Path(f"{stem}_a_rarefaction"), (w * 0.52, h * 0.46),
        )
        written += save_panel(
            lambda f: panel_concentration(
                f.add_subplot(111), tables["draws"], tables["swap"],
                args.motif_class, bio_swap),
            Path(f"{stem}_b_concentration"), (w * 0.52, h * 0.46),
        )
        written += save_panel(
            lambda f: panel_exemplars(
                f, GridSpec(1, 1, figure=f)[0, 0], tables["ubiquitous"],
                tables["restricted"], Path(h5_path),
                profile_rows=profile_rows, profile_h5=args.profile_h5,
                trim_kwargs=dict(threshold=args.trim_threshold,
                                 min_len=args.min_trim_len),
                n_ubiquitous=args.n_ubiquitous, n_restricted=args.n_restricted,
                per_row=args.logos_per_row),
            Path(f"{stem}_c_exemplars"), (w, h * 0.56),
        )
        for path in written:
            print(f"Saved {path}")

    if args.split_logos:
        logo_dir = Path(f"{stem}_logos")
        trim_kwargs = dict(threshold=args.trim_threshold, min_len=args.min_trim_len)
        paths = save_individual_logos(
            tables["ubiquitous"].head(args.n_ubiquitous), Path(h5_path),
            logo_dir, "ubiquitous", trim_kwargs,
        )
        paths += save_individual_logos(
            tables["restricted"].head(args.n_restricted), Path(h5_path),
            logo_dir, "restricted", trim_kwargs,
        )
        if profile_rows is not None and args.profile_h5:
            paths += save_individual_logos(
                profile_rows.head(args.n_restricted), Path(args.profile_h5),
                logo_dir, "profile", trim_kwargs,
            )
        print(f"Saved {len(paths)} individual logo files under {logo_dir}/")

    if profile_rows is None:
        print(
            "\nPanel c has no profile row. Once the profile compendium is built:\n"
            "  select_motif_exemplars.py --head profile --include-unmatched\n"
            "  plot_figure2.py ... --profile-exemplars ... --profile-h5 ...\n"
            "JASPAR2026 has no Inr/TATA/DPE entries, so --include-unmatched is\n"
            "required or every core promoter motif is dropped.",
            file=sys.stderr,
        )
    print(f"{n_logos} logos drawn")


if __name__ == "__main__":
    main()
