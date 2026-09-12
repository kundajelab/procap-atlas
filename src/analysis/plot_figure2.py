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
    # Haematopoietic lineages, split out of the former "blood_immune" group.
    "lymphoid_t": "T / NK",
    "lymphoid_b": "B lymphoid",
    "myeloid_erythroid": "myeloid / erythroid",
    "lymphoid_bulk": "lymphoid tissue",
    # Retained so captions rendered from pre-split outputs still resolve.
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
    "muscle": "muscle",
    "vascular": "vascular",
    "lung_airway": "lung",
    "kidney_urinary": "kidney",
    "breast": "breast",
    "bone": "bone",
    "skin": "skin",
    "adipose": "adipose",
    "endocrine": "endocrine",
}


# Compact forms for captions. A three-group lineage spelled out in full
# ("blood / immune + GI tract + metastatic") is wider than the logo above it
# and collides with its neighbours.
SHORT_GROUP_LABEL = {
    "lymphoid_t": "T/NK",
    "lymphoid_b": "B",
    "myeloid_erythroid": "myeloid",
    "lymphoid_bulk": "lymph",
    "blood_immune": "blood",
    "gi_tract": "GI",
    "liver_biliary": "liver",
    "kidney_urinary": "kidney",
    "lung_airway": "lung",
    "stem_ipsc": "stem",
    "reproductive": "repro",
    "pancreas": "panc",
    "vascular": "vasc",
    "endocrine": "endo",
    "hek": "HEK",
}


# Groups omitted from figure captions. Empty now that metastases are grouped
# by tissue of origin rather than into a `metastatic_carcinoma` bucket, so
# every group left is a real tissue worth naming. The mechanism stays because
# suppressing a group in a caption is a presentation choice that should not
# require touching the grouping.
CAPTION_OMIT_GROUPS: tuple[str, ...] = ()


def lineage_caption(value, max_groups: int = 3, short: bool = True,
                    omit: tuple[str, ...] = CAPTION_OMIT_GROUPS,
                    wrap_at: int = 14) -> str:
    """Render a lineage for a figure caption.

    `lineage` is a single group name or a comma-joined list, so the naive
    `sole_group` lookup renders "nan" for every multi-group cluster -- which is
    exactly the set worth showing once --max-groups exceeds 1 (MEF2A in
    heart+muscle, HNF1B in GI+liver+pancreas).
    """
    parts = [p for p in str(value).split(",") if p and p != "nan"]
    if not parts:
        return ""
    kept = [p for p in parts if p not in omit]
    # Fall back to the unfiltered list rather than rendering a blank caption
    # for a motif whose only group is an omitted one.
    parts = kept or parts
    table = SHORT_GROUP_LABEL if short else GROUP_LABEL
    labels = [table.get(p, GROUP_LABEL.get(p, p.replace("_", " ")))
              for p in parts]
    if len(labels) > max_groups:
        return f"{len(labels)} tissues"
    if not short:
        return " + ".join(labels)
    # Wrap rather than truncate: three-group lineages are the interesting ones
    # now that metastases are filed by origin ("GI+liver+panc" is endoderm),
    # and the widest, "blood+breast+kidney", overruns its column at one line.
    text, line = [], ""
    for label in labels:
        candidate = f"{line}+{label}" if line else label
        if line and len(candidate) > wrap_at:
            text.append(line + "+")
            line = label
        else:
            line = candidate
    text.append(line)
    return "\n".join(text)


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


def panel_lexicon_bracket(
    ax, cluster_curves: pd.DataFrame, name_curves: pd.DataFrame,
    mark_k: int = 5,
) -> None:
    """Supplementary: the lexicon size as a bracket, not a single number.

    Answers the obvious reviewer question -- how many of the 343 clusters are
    actually distinct motifs -- by plotting both readings at once. Cluster
    level is the upper bound; collapsing clusters that share a JASPAR name is
    the lower bound, with no threshold to defend. The shaded band between them
    is where the true lexicon size sits.

    The point of the panel is that the *claim* does not depend on which end you
    take: the curve is far from its own asymptote at small k either way, so a
    small study misses most of the lexicon whichever bound is used.
    """
    levels = [
        ("by cluster (upper bound)", cluster_curves, "#1b7837", "-"),
        ("by JASPAR name (lower bound)", name_curves, "#762a83", "--"),
    ]
    totals = []
    series = []
    for label, curves, color, ls in levels:
        all_cls = curves[curves["motif_class"] == "__all__"]
        uni = all_cls[all_cls["scheme"] == "uniform"].sort_values("k")
        total = float(all_cls["n_clusters_total"].iloc[0])
        totals.append(total)
        series.append((uni["k"].to_numpy(), uni["mean"].to_numpy()))
        ax.plot(uni["k"], uni["mean"], color=color, ls=ls, lw=1.8, label=label)
        ax.axhline(total, color=color, lw=0.7, ls=":", zorder=1)
        ax.text(ax.get_xlim()[1], total, f" {total:.0f}", va="center",
                ha="left", fontsize=7, color=color, clip_on=False)

    # The band is only meaningful where both curves are defined.
    (k_hi, y_hi), (k_lo, y_lo) = series
    if len(k_hi) == len(k_lo) and (k_hi == k_lo).all():
        ax.fill_between(k_hi, y_lo, y_hi, color="#999999", alpha=0.12, lw=0)

    for (k_arr, y_arr), total, color in zip(series, totals,
                                           ["#1b7837", "#762a83"]):
        hit = np.flatnonzero(k_arr == mark_k)
        if len(hit):
            y = float(y_arr[hit[0]])
            ax.plot([mark_k], [y], "o", color=color, ms=4, zorder=5)
            ax.annotate(
                f"{y / total:.0%}", xy=(mark_k, y),
                xytext=(mark_k + 9, y - max(totals) * 0.055),
                fontsize=7, color=color,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=color),
            )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=max(totals) * 1.08)
    ax.set_xlabel("Experiments sampled")
    ax.set_ylabel("Motifs recovered")
    ax.set_title(f"Lexicon size is a bracket ({mark_k}-experiment recovery marked)",
                 loc="left", fontweight="bold", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)


def panel_concentration(
    ax, draws: pd.DataFrame, swap: pd.DataFrame, motif_class: str = "TF-matched",
    biosample_swap: pd.DataFrame | None = None,
) -> None:
    sub = swap[swap["motif_class"] == motif_class]
    if not len(sub):
        raise ValueError(
            f"motif class {motif_class!r} is absent from the swap-null table "
            f"(it has {sorted(swap['motif_class'].unique())}). A "
            "--collapse-by/--drop-unnamed run writes the same filenames, so "
            "check that the table came from the canonical run."
        )
    row = sub.iloc[0]
    obs = int(row["obs_single_group"])

    d = draws[draws["motif_class"] == motif_class]["n_single_group"].to_numpy()
    if not len(d):
        raise ValueError(
            f"no null draws for motif class {motif_class!r}; rerun "
            "motif_group_concentration.py with --save-null-draws"
        )
    # The histogram and the annotations must come from the SAME run. They are
    # separate files written by the same invocation, so a rerun that omits
    # --save-null-draws leaves a stale histogram beside fresh numbers -- which
    # is exactly what happened when the tissue grouping changed from 18 to 21
    # groups: the drawn null still centred on 8.31 while the caption said 6.07.
    # Nothing downstream notices, and the panel is simply wrong.
    expected_mean = float(row["null_single_group_mean"])
    if abs(d.mean() - expected_mean) > 0.05:
        raise ValueError(
            f"null draws disagree with the swap-null table for "
            f"{motif_class!r}: draws mean {d.mean():.3f} vs reported "
            f"{expected_mean:.3f}. The two files are from different runs. "
            "Rerun motif_group_concentration.py with --save-null-draws so "
            "both are written together."
        )

    bins = np.arange(d.min() - 0.5, max(d.max(), obs) + 1.5, 1.0)
    ax.hist(d, bins=bins, color="#bbbbbb", edgecolor="white", lw=0.3,
            label=f"degree-preserving null\n(n={len(d)} permutations)")
    # Headroom for the legend, which sits upper-left because the null mode
    # does. Without it the mode's bar runs into the legend text -- and the mode
    # moves whenever the grouping changes, so this cannot be a fixed limit.
    counts, _ = np.histogram(d, bins=bins)
    ax.set_ylim(top=counts.max() * 1.28)
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
    # Anchored to the observed line in data coordinates, not to 0.97 of the
    # axes: the line sits at the right edge, so an axes-fraction anchor put the
    # last line of text underneath it.
    ax.set_xlim(right=max(ax.get_xlim()[1], obs + 1.5))
    ax.text(
        obs - 1.2, 0.42, "\n".join(parts),
        transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=7.5,
    )

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

    # Row spacing has to be solved in inches, not set as a constant.
    # `hspace` is a fraction of the *axis* height, while a caption is sized in
    # points, so any fixed value that clears three lines (name / lineage /
    # n exp) in the standalone panel draws them straight through the logos of
    # the row above once the same grid is packed into the combined figure,
    # where the band is roughly a third as tall.
    #
    # For n rows at hspace h: ax = band/(n + (n-1)h) and gap = h*ax, so
    # requiring gap >= caption gives h = c*n / (band - c*(n-1)).
    caption_lines = max(
        (str(label_fn(r)).count("\n") + 1 for r in rows.itertuples()),
        default=1,
    )
    band = spec.get_position(fig)
    band_h = band.height * fig.get_size_inches()[1]
    hspace = 1.05
    if n_sub > 1:
        while label_fontsize > 4.0:
            caption_h = caption_lines * label_fontsize * 1.2 / 72.0 + 0.015
            denom = band_h - caption_h * (n_sub - 1)
            if denom > 0:
                needed = caption_h * n_sub / denom
                # A band tall enough to satisfy this still has to leave the
                # logos legible; past ~3.5 the axes are thinner than the
                # captions and shrinking the text is the better trade.
                if needed <= 3.5:
                    hspace = max(hspace, needed)
                    break
            label_fontsize -= 0.4
    inner = GridSpecFromSubplotSpec(
        n_sub, per_row, subplot_spec=spec, wspace=0.30, hspace=hspace
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
    # Centred on the band rather than on its first sub-row. Anchored to the
    # first axis, the two category labels were both centred on a single logo
    # row and overlapped each other whenever the label was taller than one row.
    if drew:
        fig.text(
            band.x0 - 0.018, band.y0 + band.height / 2, subtitle,
            rotation=90, ha="right", va="center", fontsize=7, color="#555555",
        )
    return drew


def rank_for_panel(df: pd.DataFrame, n: int, one_per_name: bool = True):
    """The n best-supported rows, one per JASPAR name.

    select_motif_exemplars.py sorts its table by lineage so it reads well as a
    table, which means a naive head(n) takes whichever lineages sort first
    alphabetically -- blood_immune, every time. Re-rank by support here.
    """
    out = df.sort_values("total_seqlets", ascending=False)
    if one_per_name and "jaspar_name" in out.columns:
        out = out.drop_duplicates(subset="jaspar_name", keep="first")
    return out.head(n)


def panel_exemplars(fig, spec, ubiquitous, restricted, h5_path,
                    profile_rows=None, profile_h5=None, trim_kwargs=None,
                    n_ubiquitous=12, n_restricted=12, per_row=6):
    trim_kwargs = trim_kwargs or {}
    rows = []
    # The "count head:" prefix is only informative when a profile row is
    # present to contrast with. Without one, both labels carry it, and they
    # are then longer than the bands they label and overlap each other.
    have_profile = bool(
        profile_rows is not None and len(profile_rows) and profile_h5
    )
    prefix = "count head: " if have_profile else ""
    if have_profile:
        rows.append(("profile head: initiation shape",
                     rank_for_panel(profile_rows, n_restricted), profile_h5,
                     lambda r: str(r.jaspar_name)))
    rows.append((
        f"{prefix}ubiquitous",
        rank_for_panel(ubiquitous, n_ubiquitous),
        h5_path,
        lambda r: f"{r.jaspar_name}\n{int(r.prevalence)} exp",
    ))
    rows.append((
        f"{prefix}lineage-restricted",
        rank_for_panel(restricted, n_restricted),
        h5_path,
        lambda r: (
            f"{r.jaspar_name}\n"
            f"{lineage_caption(getattr(r, 'lineage', getattr(r, 'sole_group', '')))}"
            f"\n{int(r.prevalence)} exp"
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
    parser.add_argument(
        "--collapse-curves", type=Path, default=None, metavar="PATH",
        help="motif_rarefaction_{head}.tsv from a --collapse-by jaspar_name "
             "run. With it, a supplementary panel is written showing the "
             "cluster-level and name-level curves as a bracket, which is the "
             "answer to 'how many of these clusters are really distinct "
             "motifs'. Generate it with: plot_motif_rarefaction.py --head "
             "count --min-cluster-experiments 2 --collapse-by jaspar_name "
             "--out-dir DIR",
    )
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
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.45],
                  hspace=0.52, wspace=0.26,
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
        if args.collapse_curves is not None:
            if not args.collapse_curves.exists():
                print(f"ERROR: missing --collapse-curves: "
                      f"{args.collapse_curves}", file=sys.stderr)
            else:
                collapsed = pd.read_csv(args.collapse_curves, sep="\t")
                written += save_panel(
                    lambda f: panel_lexicon_bracket(
                        f.add_subplot(111), tables["curves"], collapsed,
                        args.mark_k),
                    Path(f"{stem}_s_lexicon_bracket"), (w * 0.52, h * 0.46),
                )
        for path in written:
            print(f"Saved {path}")

    if args.split_logos:
        logo_dir = Path(f"{stem}_logos")
        trim_kwargs = dict(threshold=args.trim_threshold, min_len=args.min_trim_len)
        paths = save_individual_logos(
            rank_for_panel(tables["ubiquitous"], args.n_ubiquitous),
            Path(h5_path), logo_dir, "ubiquitous", trim_kwargs,
        )
        paths += save_individual_logos(
            rank_for_panel(tables["restricted"], args.n_restricted),
            Path(h5_path), logo_dir, "restricted", trim_kwargs,
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
