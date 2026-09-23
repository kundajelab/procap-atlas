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

# Floor on `_logo_grid`'s row gap, as a fraction of axes height. Named because
# `panel_exemplars` has to charge each band for its own internal gaps when
# equalizing logo heights, and must use the same number.
INNER_HSPACE_FLOOR = 1.05


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


# Which caption lines each block gets. Three lines (name / lineage /
# prevalence) is right for a figure read at arm's length and too much for a
# projected slide, so this is a presentation choice, not a property of the data.
LABEL_FIELD_SETS = {
    # Current manuscript behaviour, preserved byte-for-byte.
    "default": {"ubiquitous": ("name", "prevalence"),
                "restricted": ("name", "lineage", "prevalence"),
                "profile": ("name",)},
    # Two lines each: every block keeps the one field carrying its own claim --
    # how many datasets for the ubiquitous block, which lineage for the other.
    # Lineage on both count blocks: "21 tissues" next to "heart+muscle" makes
    # the ubiquitous/restricted contrast legible without reading any numbers.
    "auto": {"ubiquitous": ("name", "lineage"),
             "restricted": ("name", "lineage"),
             "profile": ("name",)},
}


def motif_display_name(r, names=None, uppercase: bool = False) -> str:
    """A logo's display name, with a fallback for clusters JASPAR cannot name.

    The core promoter motifs are exactly the unnameable set -- JASPAR2026 has
    no Inr/TATA/DPE entries -- so the profile head is selected with
    `--include-unmatched` and arrives with a null `jaspar_name`. Rendering
    `str(r.jaspar_name)` put the literal string "nan" under those logos.
    """
    cluster = int(getattr(r, "cluster_final"))
    # Hand-written names are used verbatim. `uppercase` exists to homogenize
    # JASPAR gene symbols, whose mouse/human conventions mix; it must not
    # touch a name the caller typed, where "CA-Inr" is the correct casing and
    # "CA-INR" is wrong. Cluster-id fallbacks are ids, not symbols, so they
    # are left alone too.
    if names and cluster in names:
        return names[cluster]
    raw = getattr(r, "jaspar_name", None)
    text = str(raw).strip()
    if raw is None or text == "" or text.lower() == "nan":
        return f"cl{cluster}"
    return text.upper() if uppercase else text


def motif_label(fields, uppercase: bool = False, names=None):
    """Build a logo-caption renderer over a chosen subset of fields."""
    def render(r):
        parts = []
        if "name" in fields:
            parts.append(motif_display_name(r, names, uppercase))
        if "lineage" in fields:
            caption = lineage_caption(
                getattr(r, "lineage", getattr(r, "sole_group", ""))
            )
            if caption:
                parts.append(caption)
        if "prevalence" in fields:
            parts.append(f"{int(r.prevalence)} exp")
        return "\n".join(parts)
    return render


def resolve_label_fields(spec):
    """`--label-fields` -> per-block field tuples."""
    if spec in LABEL_FIELD_SETS:
        return LABEL_FIELD_SETS[spec]
    chosen = tuple(f.strip() for f in str(spec).split(",") if f.strip())
    unknown = set(chosen) - {"name", "lineage", "prevalence"}
    if unknown:
        raise SystemExit(
            f"--label-fields: unknown field(s) {sorted(unknown)}; choose from "
            "name, lineage, prevalence (or a preset: "
            f"{', '.join(LABEL_FIELD_SETS)})"
        )
    return {"ubiquitous": chosen, "restricted": chosen, "profile": chosen}


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
    pad_to: int | None = None,
) -> pd.DataFrame | None:
    """Trimmed CWM as a logomaker-ready frame, or None if the cluster is absent.

    `pad_to` centres the trimmed CWM in a window of that many positions,
    padding with zeros. Trimmed lengths run 10-25bp across the atlas, and a
    grid of equal-width axes therefore draws a 25bp motif's glyphs 2.5x
    narrower than a 10bp one -- which makes it impossible to pick a single
    text size on a slide. Padding equalizes the glyph scale instead of
    equalizing the axes. Never truncates: a `pad_to` below the trimmed length
    is ignored, so signal is not silently cropped.
    """
    cwm = load_cwm(h5_path, cluster_id, posneg)
    if cwm is None:
        return None
    start, end = trim_cwm(cwm.T, threshold=threshold, min_len=min_len)
    start, end = max(0, start - pad), min(len(cwm), end + pad)
    df = pd.DataFrame(cwm[start:end], columns=["A", "C", "G", "T"])
    if pad_to and pad_to > len(df):
        extra = pad_to - len(df)
        left = extra // 2
        blank = pd.DataFrame(0.0, index=range(extra),
                             columns=["A", "C", "G", "T"])
        df = pd.concat([blank.iloc[:left], df, blank.iloc[left:]],
                       ignore_index=True)
        assert len(df) == pad_to, (len(df), pad_to)
    return df


def max_trimmed_len(specs, trim_kwargs) -> int:
    """Longest trimmed CWM over (rows, h5) pairs, for a shared `pad_to`.

    Computed across every block that will be drawn -- including ones going to
    a separate file -- so glyphs are the same size in all of them.
    """
    kwargs = {k: v for k, v in trim_kwargs.items() if k != "pad_to"}
    best = 0
    for rows, h5 in specs:
        if rows is None or not len(rows):
            continue
        if h5 is None:
            continue
        for r in rows.itertuples():
            df = trimmed_cwm(Path(h5), int(r.cluster_final),
                             getattr(r, "posneg", "pos") or "pos", **kwargs)
            if df is not None:
                best = max(best, len(df))
    return best


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
    title: str = "b   Discovery is lineage-confined",
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
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)


def _logo_grid(fig, spec, rows, h5_path, subtitle, label_fn, trim_kwargs,
               per_row=6, label_fontsize=6.2, min_label_fontsize=4.0,
               category_style="rotated", min_hspace=None):
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
    hspace = INNER_HSPACE_FLOOR if min_hspace is None else min_hspace
    if n_sub > 1:
        # Runs at least once even when shrinking is disallowed. As a `while
        # label_fontsize > min_label_fontsize` loop this body was skipped
        # entirely whenever the caller pinned the font (as --presentation
        # does), so no clearance was computed and captions landed on the
        # logos above -- masked until now by the 1.05 default floor.
        while True:
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
            if label_fontsize <= min_label_fontsize:
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
    if drew and category_style == "rotated":
        fig.text(
            band.x0 - 0.018, band.y0 + band.height / 2, subtitle,
            rotation=90, ha="right", va="center", fontsize=7, color="#555555",
        )
    elif drew and category_style == "header":
        # Horizontal and above the band. Projected, a rotated 7pt grey label in
        # the left margin is the least legible thing on the slide, and the
        # ubiquitous/lineage-restricted split is the panel's entire claim.
        #
        # Clearing the band is not enough: each logo's caption is drawn above
        # its axes, so a header at band.y1 lands on top of the first row's
        # caption. Offset by the caption's own height, in figure coordinates.
        caption_h = caption_lines * label_fontsize * 1.2 / 72.0
        pad = caption_h / fig.get_size_inches()[1] + 0.012
        fig.text(
            band.x0, band.y1 + pad, subtitle, ha="left", va="bottom",
            fontsize=label_fontsize * 1.15, color="#222222",
            fontweight="bold",
        )
    return drew


def _logo_metaplot_grid(fig, spec, rows, h5_path, subtitle, label_fn, trim_kwargs,
                        head, config, per_row=6, label_fontsize=6.2,
                        category_style="rotated", metaplot_window=200,
                        metaplot_bin_size=5, metaplot_min_trim_len=None,
                        mapping_tsv=None):
    """Like _logo_grid, but each cell places a logo beside its own
    compendium-wide observed-signal metaplot (metaplot_motif.py's
    compendium-seqlets source) -- fusing the CWM and the signal that
    select_motif_exemplars.py's --with-metaplots report lets you inspect
    separately, into one panel-c cell. Side by side, not stacked: a motif
    and its signal are two views of the same row, and stacking them
    doubled the vertical space every row needed for no benefit -- callers
    must size the figure for cells roughly twice as *wide* as a plain logo
    instead (see main()'s with_metaplots figsize handling).

    Deliberately a sibling function rather than a `_logo_grid` parameter:
    `_logo_grid`'s adaptive font-shrinking spacing is tuned and covered by a
    byte-identical-output test for the default (no-metaplot) panel c, and a
    much simpler fixed layout here (no adaptive shrinking) keeps this
    opt-in path from ever touching that tested one. --with-metaplots is for
    exploring the figure with real signal attached, not (yet) the polished
    default.
    """
    import logomaker

    sys.path.insert(0, str(REPO_ROOT / "src" / "bpnet" / "hitcall"))
    from metaplot_motif import collect_metaplot, draw_metaplot

    n = len(rows)
    n_sub = max(1, -(-n // per_row))
    inner = GridSpecFromSubplotSpec(
        n_sub, per_row, subplot_spec=spec, wspace=0.55, hspace=0.9,
    )
    band = spec.get_position(fig)
    drew = 0
    for i, r in enumerate(rows.itertuples()):
        cell = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=inner[i // per_row, i % per_row],
            width_ratios=(1.0, 0.85), wspace=0.12,
        )
        posneg = getattr(r, "posneg", "pos") or "pos"

        ax_logo = fig.add_subplot(cell[0, 0])
        ax_meta = fig.add_subplot(cell[0, 1])
        df = trimmed_cwm(h5_path, int(r.cluster_final), posneg, **trim_kwargs)
        if df is None:
            ax_logo.text(0.5, 0.5, f"cl {r.cluster_final}\nnot in h5", ha="center",
                        va="center", fontsize=5, color="#b2182b",
                        transform=ax_logo.transAxes)
            ax_logo.set_axis_off()
            ax_meta.set_axis_off()
            continue
        logomaker.Logo(df, ax=ax_logo, shade_below=0.0, fade_below=0.0)
        ax_logo.set_xticks([])
        ax_logo.set_yticks([])
        ax_logo.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax_logo.set_title(label_fn(r), fontsize=label_fontsize, pad=1.2,
                          linespacing=1.2, loc="left")

        compendium_name = f"{posneg}_patterns.{int(r.cluster_final)}"
        try:
            sense, antisense, n_inst = collect_metaplot(
                "compendium-seqlets", head, config,
                compendium_motif_name=compendium_name, mapping_tsv=mapping_tsv,
                min_trim_len=metaplot_min_trim_len, window=metaplot_window,
                bin_size=metaplot_bin_size,
            )
            draw_metaplot(ax_meta, sense, antisense, metaplot_window, metaplot_bin_size)
            ax_meta.text(0.97, 0.92, f"n={n_inst:,}", transform=ax_meta.transAxes,
                        fontsize=4.5, ha="right", va="top", color="#555555")
        except SystemExit:
            ax_meta.text(0.5, 0.5, "no signal yet", ha="center", va="center",
                        fontsize=5, color="#999999", transform=ax_meta.transAxes)
        ax_meta.set_xticks([])
        ax_meta.set_yticks([])
        ax_meta.spines[["top", "right", "left", "bottom"]].set_visible(False)
        drew += 1

    if drew and category_style == "rotated":
        fig.text(
            band.x0 - 0.018, band.y0 + band.height / 2, subtitle,
            rotation=90, ha="right", va="center", fontsize=7, color="#555555",
        )
    elif drew:
        fig.text(
            band.x0, band.y1 + 0.02, subtitle, ha="left", va="bottom",
            fontsize=label_fontsize * 1.15, color="#222222", fontweight="bold",
        )
    return drew


def panel_sidebar(fig, spec, ubiquitous, restricted, h5_path,
                  profile_rows=None, profile_h5=None, trim_kwargs=None,
                  cols=8, band_rows=2, label_fontsize=11.0,
                  label_fields="auto", uppercase_names=True,
                  profile_names=None, header_fontsize=None):
    """Count bands stacked on the left, core promoter as a column on the right.

    Everything sits in ONE grid rather than nested sub-grids: shared rows and
    columns make every logo the same size by construction, instead of by
    matching two grids' geometry after the fact.
    """
    import logomaker

    trim_kwargs = trim_kwargs or {}
    fields = resolve_label_fields(label_fields)
    if "lineage" in fields["ubiquitous"] and "lineage" not in fields["profile"]:
        fields = dict(fields, profile=fields["profile"] + ("lineage",))

    ubi = rank_for_panel(ubiquitous, 0)
    res = rank_for_panel(restricted, 0)
    prof = profile_rows if profile_rows is not None else pd.DataFrame()

    # Each band gets the column count that fills `band_rows` rows exactly --
    # 10 ubiquitous over 2 rows is 5 wide, 15 lineage-restricted is 8. A
    # single shared column count cannot fill both (at 8, the ubiquitous band's
    # second row holds two logos and six blanks), and a full band reads far
    # better than a rectangular outline with holes in it.
    if band_rows:
        ubi_cols = max(1, -(-len(ubi) // band_rows))
        res_cols = max(1, -(-len(res) // band_rows))
    else:
        ubi_cols = res_cols = cols
    ubi_rows = max(1, -(-len(ubi) // ubi_cols))
    res_rows = max(1, -(-len(res) // res_cols))
    # A blank row separates the two count bands, giving the lower band's
    # header somewhere to sit that is not on top of the row above.
    body_cols = max(ubi_cols, res_cols)
    n_rows = max(ubi_rows + 1 + res_rows, len(prof))
    n_cols = body_cols + (1 if len(prof) else 0)

    inner = GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=spec, wspace=0.30,
        hspace=INNER_HSPACE_FLOOR,
    )
    header_fontsize = header_fontsize or label_fontsize * 1.15
    drew = 0

    def place(rows_df, h5, row0, col0, per_row, label_fn, header):
        nonlocal drew
        first_ax = None
        for i, r in enumerate(rows_df.itertuples()):
            ax = fig.add_subplot(
                inner[row0 + i // per_row, col0 + i % per_row])
            first_ax = first_ax or ax
            df = trimmed_cwm(Path(h5), int(r.cluster_final),
                             getattr(r, "posneg", "pos") or "pos",
                             **trim_kwargs)
            if df is None:
                ax.set_axis_off()
                continue
            logomaker.Logo(df, ax=ax, shade_below=0.0, fade_below=0.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.set_title(label_fn(r), fontsize=label_fontsize, pad=1.2,
                         linespacing=1.2)
            drew += 1
        if first_ax is not None and header:
            fig.canvas.draw()
            box = first_ax.get_position()
            # Clear the caption, which is drawn above the axes, not inside it.
            lines = max(str(label_fn(r)).count("\n") + 1
                        for r in rows_df.itertuples())
            pad = lines * label_fontsize * 1.2 / 72.0 / fig.get_size_inches()[1]
            fig.text(box.x0, box.y1 + pad + 0.012, header, ha="left",
                     va="bottom", fontsize=header_fontsize, color="#222222",
                     fontweight="bold")

    place(ubi, h5_path, 0, 0, ubi_cols,
          motif_label(fields["ubiquitous"], uppercase_names), "ubiquitous")
    place(res, h5_path, ubi_rows + 1, 0, res_cols,
          motif_label(fields["restricted"], uppercase_names),
          "lineage-restricted")
    if len(prof):
        place(prof, profile_h5, 0, body_cols, 1,
              motif_label(fields["profile"], uppercase_names, profile_names),
              "core promoter")
    return drew


def rank_for_panel(df: pd.DataFrame, n: int, one_per_name: bool = True):
    """The n best-supported rows, one per JASPAR name.

    select_motif_exemplars.py sorts its table by lineage so it reads well as a
    table, which means a naive head(n) takes whichever lineages sort first
    alphabetically -- blood_immune, every time. Re-rank by support here.
    """
    out = df.sort_values("total_seqlets", ascending=False)
    # n <= 0 (or None) means the full set, which is what a talk slide wants:
    # the lexicon, not a picked handful.
    take_all = n is None or n <= 0
    if one_per_name and "jaspar_name" in out.columns:
        # `drop_duplicates` treats nulls as equal to each other, so on the
        # profile table -- where every core promoter cluster is unnamed -- a
        # naive dedup kept exactly one of them. Dedup the named rows only.
        named = out["jaspar_name"].notna() & (
            out["jaspar_name"].astype(str).str.strip().str.lower() != "nan"
        )
        out = pd.concat([
            out[named].drop_duplicates(subset="jaspar_name", keep="first"),
            out[~named],
        ]).sort_values("total_seqlets", ascending=False)
    return out if take_all else out.head(n)


def panel_exemplars(fig, spec, ubiquitous, restricted, h5_path,
                    profile_rows=None, profile_h5=None, trim_kwargs=None,
                    n_ubiquitous=12, n_restricted=12, per_row=6,
                    label_fontsize=6.2, min_label_fontsize=4.0,
                    category_style="rotated", label_fields="default",
                    uppercase_names=False, hspace=0.55,
                    profile_names=None, n_profile=None, blocks=None,
                    head_prefixes=True, equalize_band_heights=False,
                    min_hspace=None, head="count", with_metaplots=False,
                    metaplot_config=None, metaplot_window=200,
                    metaplot_bin_size=5, metaplot_min_trim_len=None,
                    metaplot_mapping_tsv=None):
    trim_kwargs = trim_kwargs or {}
    if with_metaplots and metaplot_config is None:
        import yaml
        with open(REPO_ROOT / "configs" / "experiment_config.yaml") as f:
            metaplot_config = yaml.safe_load(f)
    fields = resolve_label_fields(label_fields)
    have_profile = bool(
        profile_rows is not None and len(profile_rows) and profile_h5
    )
    wanted = blocks or ("profile", "ubiquitous", "restricted")
    show_profile = have_profile and "profile" in wanted
    show_counts = any(b in wanted for b in ("ubiquitous", "restricted"))
    # The head prefixes are only informative when both heads are on the same
    # figure to contrast with. Alone, every label carries one, and they are
    # then longer than the bands they label and overlap each other. A slide
    # does not want them at all: the biological grouping is the point, and
    # which attribution head produced it is an internal detail.
    both = show_profile and show_counts
    count_prefix = "count head: " if (both and head_prefixes) else ""

    rows = []
    if show_profile:
        rows.append((
            # Named for what it is rather than where it came from whenever the
            # head is not the distinction being drawn.
            "profile head: initiation shape"
            if (both and head_prefixes) else "core promoter",
            rank_for_panel(profile_rows, n_profile if n_profile is not None
                           else n_restricted),
            profile_h5,
            motif_label(fields["profile"], uppercase_names, profile_names),
            "profile",
        ))
    if "ubiquitous" in wanted:
        rows.append((
            f"{count_prefix}ubiquitous",
            rank_for_panel(ubiquitous, n_ubiquitous),
            h5_path,
            motif_label(fields["ubiquitous"], uppercase_names),
            head,
        ))
    if "restricted" in wanted:
        rows.append((
            f"{count_prefix}lineage-restricted",
            rank_for_panel(restricted, n_restricted),
            h5_path,
            motif_label(fields["restricted"], uppercase_names),
            head,
        ))

    # Height per category in proportion to how many sub-rows it needs, so a
    # 15-motif category is not squeezed into the same band as a 5-motif one.
    sub_rows = [max(1, -(-len(df) // per_row)) for _, df, _, _, _ in rows]
    ratios = sub_rows
    if equalize_band_heights:
        # Sub-row count alone is not the right ratio: within a band of n
        # sub-rows the axes height is band/(n + (n-1)*h), so bands of 1, 2 and
        # 3 sub-rows came out at 0.78, 0.51 and 0.46 inches. Charging each
        # band for its own internal gaps equalizes the logo height -- which is
        # the whole point when one text size has to fit every band. The gap
        # that matters is `_logo_grid`'s internal one, not the `hspace`
        # separating the bands.
        floor = INNER_HSPACE_FLOOR if min_hspace is None else min_hspace
        ratios = [n + (n - 1) * floor for n in sub_rows]
    inner = GridSpecFromSubplotSpec(
        len(rows), 1, subplot_spec=spec, hspace=hspace, height_ratios=ratios
    )
    total = 0
    for i, (subtitle, df, h5, label_fn, row_head) in enumerate(rows):
        if with_metaplots:
            total += _logo_metaplot_grid(
                fig, inner[i, 0], df, h5, subtitle, label_fn, trim_kwargs,
                row_head, metaplot_config, per_row=per_row,
                label_fontsize=label_fontsize, category_style=category_style,
                metaplot_window=metaplot_window,
                metaplot_bin_size=metaplot_bin_size,
                metaplot_min_trim_len=metaplot_min_trim_len,
                mapping_tsv=metaplot_mapping_tsv,
            )
        else:
            total += _logo_grid(fig, inner[i, 0], df, h5, subtitle, label_fn,
                                trim_kwargs, per_row=per_row,
                                label_fontsize=label_fontsize,
                                min_label_fontsize=min_label_fontsize,
                                category_style=category_style,
                                min_hspace=min_hspace)
    return total


def logo_axes_size(draw, figsize) -> tuple[float, float]:
    """(width, height) in inches of the smallest axes `draw` produces."""
    fig = plt.figure(figsize=figsize)
    draw(fig)
    fig.canvas.draw()
    fw, fh = fig.get_size_inches()
    boxes = [(a.get_position().width * fw, a.get_position().height * fh)
             for a in fig.axes]
    plt.close(fig)
    if not boxes:
        return (0.0, 0.0)
    return (min(w for w, _ in boxes), min(h for _, h in boxes))


def solve_figure_height(draw, width: float, target_h: float,
                        lo: float = 1.0, hi: float = 4.0) -> float:
    """Figure height whose logo axes are `target_h` inches tall.

    Axes height is affine in figure height -- the grid scales, while captions
    are sized in points and do not -- so two probes determine it exactly. This
    is solved rather than guessed because the whole point of the separate
    files is that a base pair and a glyph are the same size in both, so one
    text size works for the entire slide.
    """
    h_lo = logo_axes_size(draw, (width, lo))[1]
    h_hi = logo_axes_size(draw, (width, hi))[1]
    if h_hi == h_lo:
        return hi
    slope = (h_hi - h_lo) / (hi - lo)
    guess = max(0.6, lo + (target_h - h_lo) / slope)
    # The relationship is only piecewise affine: the caption font-shrink loop
    # and caption line count move the fixed offset. One secant step from the
    # estimate removes the residual, which is several percent otherwise.
    h_guess = logo_axes_size(draw, (width, guess))[1]
    if h_guess and abs(h_guess - target_h) > 1e-4:
        span = guess - hi
        if abs(span) > 1e-9 and abs(h_guess - h_hi) > 1e-9:
            slope2 = (h_guess - h_hi) / span
            guess = max(0.6, guess + (target_h - h_guess) / slope2)
    return guess


def save_panel(draw, path_stem: Path, figsize, exts=("pdf", "png"),
               transparent: bool = False) -> list[str]:
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
        fig.savefig(path, dpi=400, bbox_inches="tight",
                    transparent=transparent)
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
    parser.add_argument(
        "--with-metaplots", action="store_true",
        help="panel c: stack a compendium-wide observed-signal metaplot "
             "under each logo (metaplot_motif.py's compendium-seqlets "
             "source). Needs finemo/h5py (Linux only) and queries every "
             "contributing experiment's own bigwigs per motif, so this is "
             "much slower than the default logo-only panel. Opt-in and "
             "uses a simpler fixed layout than the default -- see "
             "_logo_metaplot_grid's docstring",
    )
    parser.add_argument("--metaplot-window", type=int, default=200, metavar="BP")
    parser.add_argument("--metaplot-bin-size", type=int, default=5, metavar="BP")
    parser.add_argument(
        "--metaplot-min-trim-len", type=int, default=None, metavar="BP",
        help="must match the value hitcall/launch.py was run with, if any -- "
             "unrelated to --min-trim-len below, which controls CWM display "
             "trimming, not which hitcall directory metaplots are read from",
    )
    parser.add_argument("--metaplot-mapping-tsv", type=Path, default=None)
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
    parser.add_argument(
        "--collapse-concentration", type=Path, default=None, metavar="DIR",
        help="directory holding motif_concentration_{head}_*.tsv from a "
             "--collapse-by jaspar_name run (with --save-null-draws). With "
             "it, a supplementary panel b is written at JASPAR-name level, "
             "which shows the concentration result is not an artifact of "
             "counting splits of one motif as separate restricted motifs.",
    )
    parser.add_argument("--profile-exemplars", type=Path, default=None, metavar="PATH")
    parser.add_argument("--profile-h5", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--profile-names", type=Path, default=None, metavar="TSV",
        help="two-column TSV (cluster_final, name) giving display names for "
             "profile clusters JASPAR cannot name. JASPAR2026 has no "
             "Inr/TATA/DPE entries, so without this the core promoter motifs "
             "render as cl<id>.",
    )
    parser.add_argument(
        "--profile-cols", type=int, default=None, metavar="N",
        help="columns in the separate core promoter figure; 1 makes it a "
             "vertical column, which drops onto a slide beside the count "
             "figure (default: as many columns as there are logos)",
    )
    parser.add_argument("--n-profile", type=int, default=None, metavar="N",
                        help="logos in the profile block (default: "
                             "--n-restricted)")
    parser.add_argument(
        "--row-gap", type=float, default=None, metavar="F",
        help="gap between logo rows, as a fraction of logo height. The "
             "caption height still sets a floor, so a small value means "
             "'no more than needed' (default 1.05; 0.15 with --presentation)",
    )
    parser.add_argument(
        "--band-gap", type=float, default=None, metavar="F",
        help="gap between the ubiquitous and lineage-restricted bands, as a "
             "fraction of logo height (default 0.55; 0.30 with "
             "--presentation)",
    )
    parser.add_argument(
        "--logo-length", type=int, default=None, metavar="BP",
        help="pad every trimmed CWM to this many positions so all logos draw "
             "at the same glyph size; 0 = auto (the longest motif being "
             "drawn). Default off; auto with --presentation.",
    )
    parser.add_argument("--trim-threshold", type=float, default=0.3, metavar="F")
    parser.add_argument("--min-trim-len", type=int, default=8, metavar="BP")
    parser.add_argument("--n-ubiquitous", type=int, default=None, metavar="N",
                        help="logos in the ubiquitous block; 0 = every row "
                             "(default 12; 0 with --presentation)")
    parser.add_argument("--n-restricted", type=int, default=None, metavar="N",
                        help="logos in the lineage-restricted block; 0 = every "
                             "row (default 12; 0 with --presentation)")
    parser.add_argument("--logos-per-row", type=int, default=6, metavar="N",
                        help="wrap each category's logos at this width "
                             "(default: 6)")
    parser.add_argument("--mark-k", type=int, default=5, metavar="K")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7.4, 6.2),
                        metavar=("W", "H"), help="inches (default: 7.4 6.2)")
    parser.add_argument("--out-stem", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--band-rows", type=int, default=2, metavar="N",
        help="with --layout sidebar: how many rows each count band fills. "
             "Column count is derived per band so each fills its rows "
             "exactly (default 2). 0 uses --logos-per-row for both.",
    )
    parser.add_argument(
        "--layout", default="stacked", choices=["stacked", "sidebar"],
        help="with --presentation --consolidate: 'stacked' puts the three "
             "bands one above another; 'sidebar' stacks the two count bands "
             "on the left and runs the core promoter motifs down a column on "
             "the right, which fits a slide's aspect better.",
    )
    parser.add_argument(
        "--consolidate", action="store_true",
        help="with --presentation, draw every motif in ONE rectangular grid "
             "instead of separate count/profile files. Category bands are "
             "dropped, so the class shows up in each caption's second line "
             "(core promoter / N tissues / a lineage). Column count comes "
             "from --logos-per-row, nudged to divide the motif count evenly.",
    )
    parser.add_argument(
        "--presentation", action="store_true",
        help="write only the exemplar panel, sized and styled for a slide: "
             "larger captions, horizontal category headers, two caption lines "
             "per logo, uppercased names, transparent background. Bundles the "
             "defaults of the five options below; each can still be set "
             "explicitly to override.",
    )
    parser.add_argument("--label-fontsize", type=float, default=None,
                        metavar="PT",
                        help="logo caption size (default 6.2; 11 with "
                             "--presentation)")
    parser.add_argument(
        "--label-fields", default=None, metavar="SPEC",
        help="caption lines per logo: a comma-list of name,lineage,prevalence "
             "applied to both blocks, or a preset -- 'default' (manuscript) "
             "or 'auto' (two lines, each block keeping the field that carries "
             "its claim; the --presentation default)",
    )
    parser.add_argument(
        "--category-label", default=None,
        choices=["rotated", "header", "none"],
        help="how to label the ubiquitous/lineage-restricted blocks (default "
             "rotated; 'header' with --presentation)",
    )
    parser.add_argument("--uppercase-names", action="store_true", default=None,
                        help="uppercase motif names, so mouse- and "
                             "human-convention JASPAR names stop mixing "
                             "(default off; on with --presentation)")
    parser.add_argument(
        "--logo-size", type=float, nargs=2, default=None, metavar=("W", "H"),
        help="physical size of ONE logo's axes, in inches (default 1.25 0.5 "
             "with --consolidate). The figure size is then derived from the "
             "grid, so changing --logos-per-row rearranges the layout "
             "without resizing the motifs.",
    )
    parser.add_argument("--exemplar-figsize", type=float, nargs=2,
                        default=None, metavar=("W", "H"),
                        help="exemplar-panel size in inches (default scales "
                             "from --figsize; 10 5 with --presentation)")
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

    # Each knob keeps its manuscript default unless --presentation is set and
    # the user has not chosen a value, so an explicit flag always wins.
    presentation_defaults = {
        # A logo is a fixed physical size; the figure grows to fit the grid.
        "logo_size": (1.25, 0.5),
        # Let the caption height set the row gap, rather than a fixed floor
        # tuned for the manuscript's three-line captions.
        "row_gap": 0.15,
        # The band gap has to clear the next band's header *and* its first
        # caption (~0.58in at 11pt), while the row gap only clears a caption.
        # Tying them together is what made the whole figure loose. Note the
        # units differ: matplotlib measures this one against average *band*
        # height (~1.8in here), not logo height.
        "band_gap": 0.45,
        # Uniform glyph size, so one text size works for the whole slide.
        "logo_length": 0,
        # The full lexicon, not a picked handful.
        "n_ubiquitous": 0,
        "n_restricted": 0,
        "label_fontsize": 11.0,
        "label_fields": "auto",
        "category_label": "header",
        "uppercase_names": True,
        "exemplar_figsize": (10.0, 7.5),
    }
    manuscript_defaults = {
        "logo_size": None,
        "row_gap": None,
        "band_gap": 0.55,
        "logo_length": None,
        "n_ubiquitous": 12,
        "n_restricted": 12,
        "label_fontsize": 6.2,
        "label_fields": "default",
        "category_label": "rotated",
        "uppercase_names": False,
        "exemplar_figsize": None,
    }
    category_label_explicit = args.category_label is not None
    chosen = presentation_defaults if args.presentation else manuscript_defaults
    for key, value in chosen.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    # Fails here on a typo rather than after the h5 reads.
    resolve_label_fields(args.label_fields)

    # Rotated multi-word band labels ("profile head: initiation shape") need
    # more vertical run length than a short band has room for -- panel c's
    # profile row is a single row of logos, nowhere near enough height for
    # ~30 characters of rotated text, and it visibly collided with the
    # neighboring band's label. Header style puts the label on its own
    # horizontal line above the band instead, which any band has room for.
    if args.with_metaplots and not category_label_explicit:
        args.category_label = "header"

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
    # --presentation draws only the exemplar panel, so the rarefaction and
    # concentration tables are not inputs to it.
    if args.presentation:
        needed = {k: v for k, v in needed.items()
                  if k in ("ubiquitous", "restricted")}
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
    profile_names = None
    if args.profile_names:
        if not args.profile_names.exists():
            print(f"ERROR: missing --profile-names: {args.profile_names}",
                  file=sys.stderr)
            sys.exit(1)
        names = pd.read_csv(args.profile_names, sep="\t")
        missing_cols = {"cluster_final", "name"} - set(names.columns)
        if missing_cols:
            print(f"ERROR: --profile-names needs columns cluster_final, name; "
                  f"missing {sorted(missing_cols)}", file=sys.stderr)
            sys.exit(1)
        profile_names = dict(
            zip(names["cluster_final"].astype(int), names["name"].astype(str))
        )
    stem = args.out_stem or (args.in_dir / f"figure2_{args.head}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    exemplar_kwargs = dict(
        profile_rows=profile_rows, profile_h5=args.profile_h5,
        trim_kwargs=dict(threshold=args.trim_threshold,
                         min_len=args.min_trim_len),
        n_ubiquitous=args.n_ubiquitous, n_restricted=args.n_restricted,
        per_row=args.logos_per_row, label_fontsize=args.label_fontsize,
        category_style=args.category_label,
        label_fields=args.label_fields,
        uppercase_names=args.uppercase_names,
        profile_names=profile_names, n_profile=args.n_profile,
        head_prefixes=not args.presentation,
        equalize_band_heights=args.presentation,
        head=args.head, with_metaplots=args.with_metaplots,
        metaplot_window=args.metaplot_window,
        metaplot_bin_size=args.metaplot_bin_size,
        metaplot_min_trim_len=args.metaplot_min_trim_len,
        metaplot_mapping_tsv=args.metaplot_mapping_tsv,
    )

    if args.profile_exemplars and profile_rows is None:
        print(f"WARNING: {args.profile_exemplars} not found; omitting the "
              "profile row", file=sys.stderr)

    if args.presentation:
        # Only the exemplar panel: the rarefaction and concentration panels are
        # arguments about sampling that a talk makes verbally, and compositing
        # them here would just shrink the logos.
        def sub_rows_for(rows, n):
            picked = len(rank_for_panel(rows, n)) if rows is not None else 0
            return max(1, -(-picked // args.logos_per_row)) if picked else 0

        have_profile = profile_rows is not None and args.profile_h5
        n_prof = (args.n_profile if args.n_profile is not None
                  else args.n_restricted)

        # One `pad_to` across every band, including any going to another file,
        # so a base pair is the same width in all of them.
        if args.logo_length is not None:
            specs = [
                (rank_for_panel(tables["ubiquitous"], args.n_ubiquitous),
                 h5_path),
                (rank_for_panel(tables["restricted"], args.n_restricted),
                 h5_path),
            ]
            if have_profile:
                specs.append(
                    (rank_for_panel(profile_rows, n_prof), args.profile_h5))
            pad_to = args.logo_length or max_trimmed_len(
                specs, exemplar_kwargs["trim_kwargs"])
            exemplar_kwargs["trim_kwargs"]["pad_to"] = pad_to
            print(f"padding every logo to {pad_to} bp for a uniform "
                  "glyph size")

        def make_draw(blocks, per_row=None):
            state = {"n": 0}
            kwargs = dict(exemplar_kwargs)
            if per_row is not None:
                kwargs["per_row"] = per_row

            def draw(f):
                state["n"] = panel_exemplars(
                    f, GridSpec(1, 1, figure=f)[0, 0], tables["ubiquitous"],
                    tables["restricted"], Path(h5_path),
                    hspace=args.band_gap,
                    min_hspace=args.row_gap,
                    # Hold the requested size: the caller asked for legible
                    # text, so if it will not fit, that is a cue to lower
                    # --n-ubiquitous/--n-restricted, not to shrink captions.
                    min_label_fontsize=args.label_fontsize,
                    blocks=blocks, **kwargs,
                )

            return draw, state

        def emit(draw, state, suffix, figsize):
            for path in save_panel(draw, Path(f"{stem}_{suffix}"), figsize,
                                   transparent=True):
                print(f"Saved {path}")
            return state["n"]

        w, h = args.exemplar_figsize
        # Per-logo geometry is pinned to what the default 10x7.5 layout gives,
        # so every presentation figure shares one glyph and text size.
        target_h = 0.5 * w / 10.0

        if args.consolidate:
            # All three bands in one file on a single column grid, so every
            # logo lands on the same pitch -- but the bands stay separate and
            # labelled. An undifferentiated array would lose the core
            # promoter / ubiquitous / lineage-restricted split, which is the
            # claim the panel exists to make.
            if args.layout == "sidebar":
                prof_sel = (rank_for_panel(profile_rows, n_prof)
                            if have_profile else None)
                state = {"n": 0}

                def draw(f):
                    state["n"] = panel_sidebar(
                        f, GridSpec(1, 1, figure=f)[0, 0],
                        tables["ubiquitous"], tables["restricted"],
                        Path(h5_path), profile_rows=prof_sel,
                        profile_h5=args.profile_h5,
                        trim_kwargs=exemplar_kwargs["trim_kwargs"],
                        cols=args.logos_per_row,
                        band_rows=args.band_rows,
                        label_fontsize=args.label_fontsize,
                        label_fields=args.label_fields,
                        uppercase_names=args.uppercase_names,
                        profile_names=profile_names,
                    )

                n_sub = ((args.band_rows or 1) * 2 + 1) if args.band_rows \
                    else 5
            else:
                draw, state = make_draw(
                    ("profile", "ubiquitous", "restricted"))
                n_sub = (
                    sub_rows_for(tables["ubiquitous"], args.n_ubiquitous)
                    + sub_rows_for(tables["restricted"], args.n_restricted))
                if have_profile:
                    n_sub += sub_rows_for(profile_rows, n_prof)

            # Logo size is the invariant and the figure is derived from it, so
            # --logos-per-row rearranges the layout instead of rescaling the
            # motifs. Axes width is strictly proportional to figure width, so
            # one probe fixes the width; the height is then solved.
            want_w, want_h = args.logo_size
            width = w
            probe_w, _ = logo_axes_size(draw, (width, 1.6 * n_sub))
            if probe_w > 0:
                width *= want_w / probe_w
            height = solve_figure_height(draw, width, want_h,
                                         lo=1.2 * n_sub, hi=2.2 * n_sub)
            total = emit(draw, state, "presentation_all", (width, height))
            aw, ah = logo_axes_size(draw, (width, height))
            print(f"figure {width:.2f} x {height:.2f} in "
                  f"({width / height:.2f}:1); logo axes "
                  f"{aw:.3f} x {ah:.3f} in; {total} logos drawn")
            return

        # Otherwise the count and profile lexicons go to separate files: on a
        # slide the count lexicon is the main figure and the core promoter
        # motifs are a different claim, so compositing only shrinks both.
        counts_draw, counts_state = make_draw(("ubiquitous", "restricted"))
        # Derive the figure from the logo, not the other way round: with the
        # figure pinned, tightening the row gaps just inflated the axes (a
        # 1.25 x 0.92in logo instead of 1.25 x 0.50in).
        want_w, want_h = args.logo_size
        n_sub = (sub_rows_for(tables["ubiquitous"], args.n_ubiquitous)
                 + sub_rows_for(tables["restricted"], args.n_restricted))
        probe_w, _ = logo_axes_size(counts_draw, (w, 1.6 * n_sub))
        if probe_w > 0:
            w = w * want_w / probe_w
        h = solve_figure_height(counts_draw, w, want_h,
                                lo=1.2 * n_sub, hi=2.2 * n_sub)
        axes_w, axes_h = logo_axes_size(counts_draw, (w, h))
        total = emit(counts_draw, counts_state, "presentation_counts", (w, h))
        print(f"counts figure {w:.2f} x {h:.2f} in ({w / h:.2f}:1)")

        if have_profile:
            n_logos_prof = len(rank_for_panel(profile_rows, n_prof))
            # Give the profile grid only the columns it fills, and a width in
            # the same proportion, so a column -- and therefore a base pair --
            # is exactly as wide as in the counts figure. Keeping five columns
            # for three logos made each one 40% narrower per bp.
            cols = args.profile_cols or max(
                1, min(n_logos_prof, args.logos_per_row))
            cols = max(1, min(cols, n_logos_prof))
            prof_w = w * cols / args.logos_per_row
            prof_draw, prof_state = make_draw(("profile",), per_row=cols)
            # `wspace` is a fraction of axes width, so a 3-column grid does not
            # divide its figure the way a 5-column one does -- the
            # proportional width lands ~3% off. Axes width is strictly
            # proportional to figure width, so one probe corrects it exactly.
            probe_w, _ = logo_axes_size(prof_draw, (prof_w, 2.0))
            if probe_w > 0:
                prof_w *= axes_w / probe_w
            prof_rows = max(1, -(-n_logos_prof // cols))
            prof_h = solve_figure_height(prof_draw, prof_w, axes_h,
                                         lo=1.2 * prof_rows,
                                         hi=2.6 * prof_rows)
            total += emit(prof_draw, prof_state, "presentation_profile",
                          (prof_w, prof_h))
            got_w, got_h = logo_axes_size(prof_draw, (prof_w, prof_h))
            print(f"logo axes: counts {axes_w:.3f} x {axes_h:.3f} in, "
                  f"profile {got_w:.3f} x {got_h:.3f} in")
        print(f"{total} logos drawn")
        return

    # --with-metaplots cells are ~1.85x as wide as a plain logo (logo beside
    # its metaplot, not stacked -- see _logo_metaplot_grid) and still need
    # room for a 3-line caption, so the manuscript default --figsize
    # (7.4x6.2in, sized for panel c's plain single-row-per-cell layout)
    # starves it badly enough to visibly overlap rows. Auto-scale from the
    # actual row count instead of guessing a fixed bigger constant, unless
    # the user already overrode --figsize.
    c_height_ratio = 1.45
    if args.with_metaplots and tuple(args.figsize) == (7.4, 6.2):
        def _sub_rows(rows, n):
            picked = len(rank_for_panel(rows, n)) if rows is not None else 0
            return max(1, -(-picked // args.logos_per_row)) if picked else 0

        n_sub_total = (
            _sub_rows(tables["ubiquitous"], args.n_ubiquitous)
            + _sub_rows(tables["restricted"], args.n_restricted)
        )
        if profile_rows is not None and args.profile_h5:
            n_prof = (args.n_profile if args.n_profile is not None
                      else args.n_restricted)
            n_sub_total += _sub_rows(profile_rows, n_prof)

        ab_height = 3.0  # fixed: panels a/b don't grow with panel c's content
        c_height = 1.6 * max(n_sub_total, 1)
        c_height_ratio = c_height / ab_height
        args.figsize = [max(14.0, 2.3 * args.logos_per_row),
                        ab_height + c_height + 1.0]

    fig = plt.figure(figsize=tuple(args.figsize))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, c_height_ratio],
                  hspace=0.52, wspace=0.26,
                  left=0.09, right=0.965, top=0.94, bottom=0.04)

    panel_rarefaction(fig.add_subplot(gs[0, 0]), tables["curves"], args.mark_k)
    panel_concentration(
        fig.add_subplot(gs[0, 1]), tables["draws"], tables["swap"],
        args.motif_class, bio_swap,
    )

    sub = gs[1, :].subgridspec(1, 1)
    n_logos = panel_exemplars(
        fig, sub[0, 0], tables["ubiquitous"], tables["restricted"],
        Path(h5_path), **exemplar_kwargs,
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
                tables["restricted"], Path(h5_path), **exemplar_kwargs),
            Path(f"{stem}_c_exemplars"), (w, h * 0.56),
        )
        if args.collapse_concentration is not None:
            cd = args.collapse_concentration
            cdraws = cd / f"motif_concentration_{args.head}_tissue_nulldraws.tsv"
            cswap = cd / f"motif_concentration_{args.head}_tissue_swapnull.tsv"
            cbio = cd / f"motif_concentration_{args.head}_biosample_swapnull.tsv"
            missing_c = [q for q in (cdraws, cswap) if not q.exists()]
            if missing_c:
                for q in missing_c:
                    print(f"ERROR: missing {q}", file=sys.stderr)
                print("Generate with: motif_group_concentration.py --head "
                      f"{args.head} --group-level tissue --collapse-by "
                      f"jaspar_name --save-null-draws --out-dir {cd}",
                      file=sys.stderr)
            else:
                written += save_panel(
                    lambda f: panel_concentration(
                        f.add_subplot(111),
                        pd.read_csv(cdraws, sep="\t"),
                        pd.read_csv(cswap, sep="\t"),
                        args.motif_class,
                        pd.read_csv(cbio, sep="\t") if cbio.exists() else None,
                        title="Discovery is lineage-confined\n"
                              "(one unit per JASPAR name)"),
                    Path(f"{stem}_s_concentration_jaspar_name"),
                    (w * 0.52, h * 0.46),
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
