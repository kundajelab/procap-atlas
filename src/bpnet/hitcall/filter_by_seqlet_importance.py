"""Drop hits whose per-hit score falls below what even the weakest
TF-MoDISco discovery seqlets for that motif would support.

filter_low_confidence_hits.py looks for a bimodal split *within* a motif's
own hit population, but not every motif has one (e.g. TATA in K562
ENCSR220XSM showed no separable low/high mode by hit_correlation even after
filter_repeat_density.py, despite being called at ~50% peak prevalence --
implausibly high for a canonical single-copy element). This script instead
anchors the cutoff to real ground truth: the seqlets TF-MoDISco already
trusted enough to build this motif's CWM from in the first place. Any hit
scoring below what even the weakest ~1% of those discovery examples showed
is hard to defend as a real instance of the motif, independent of whether
the hit population itself shows any visible internal structure.

Mirrors Kelly Cochran's ProCapNet hit-filtering notebook
(filter_and_merge_finemo_results_across_cells.ipynb in
kellycochran/procapnet_allscripts), which filters per-hit on
hit_importance/hit_score_combo (hit_correlation * hit_importance) too --
but her thresholds (0.01, 0.015 for Inr) were fixed constants chosen by
eyeballing per-motif histograms. This is that same idea made data-driven and
motif-identity-agnostic: derive the floor from each motif's own seqlets
instead of a hand-picked constant.

hit_importance (Fi-NeMo's own per-hit column) is exactly sum(|contribution|)
over the hit's trimmed span -- see hitcaller.py's fit_contribs, no
coefficient scaling or CWM-shape weighting involved -- so it's directly and
exactly reproducible for seqlets too, using regions.npz's raw contribution
track and seqlets.tsv's own trimmed start/end coordinates (already written
by report_bpnet.py's `finemo report` call, independent of any hit-calling
settings). hit_correlation is *not* reproducible this way: it depends on
importance_scale, a per-window normalization computed inside Fi-NeMo's
iterative optimizer, with no closed form outside it. So when --score-column
is hit_score_combo (hit_correlation * hit_importance, matching Kelly's own
definition), the seqlet-side reference value approximates each seqlet's
correlation as ~1.0 (defensible since a seqlet is by construction one of
the examples that defined the motif's own consensus) rather than computing
it -- this script says so explicitly at runtime, it isn't silently assumed.

Runs a self-check before trusting any of this: recomputes hit_importance
for the *existing* hits directly from regions.npz and compares against
Fi-NeMo's own recorded value, refusing to proceed if they don't match
closely (protects against a subtle reimplementation bug silently producing
wrong seqlet thresholds).

Run after filter_low_confidence_hits.py (reads its output if present, else
filter_repeat_density.py's, else hits_unique.tsv) and before
report_bpnet.py, which prefers this script's output
(hits_seqlet_filtered.tsv) when present. Requires report_bpnet.py to have
already been run at least once for this experiment/head, to produce
report/seqlets.tsv.

Usage:
    python src/bpnet/hitcall/filter_by_seqlet_importance.py -e ENCSR882DWM
    python src/bpnet/hitcall/filter_by_seqlet_importance.py -e ENCSR882DWM --min-trim-len 6
    python src/bpnet/hitcall/filter_by_seqlet_importance.py -e ENCSR882DWM --percentile 1 --percentile-multiplier 0.5
    python src/bpnet/hitcall/filter_by_seqlet_importance.py -e ENCSR882DWM --score-column hit_score_combo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from finemo.data_io import load_regions_npz

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, resolve_hits_path, trim_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_PERCENTILE = 1.0
DEFAULT_PERCENTILE_MULTIPLIER = 0.5
DEFAULT_MIN_SEQLETS = 20
DEFAULT_SCORE_COLUMN = "hit_importance"
VALIDATION_MIN_CORRELATION = 0.999
VALIDATION_MAX_MEDIAN_REL_ERR = 0.01


def build_peak_row_index(peaks_df):
    """Map peak_id -> row index into the regions.npz contribution array."""
    peak_ids = peaks_df["peak_id"].to_numpy()
    return {int(pid): i for i, pid in enumerate(peak_ids)}


def project_contribs(contribs, sequences):
    """Collapse hypothetical (N, 4, L) contribs to projected (N, L) by
    masking with the one-hot sequence, matching finemo.main.report()'s own
    reshape. No-op if contribs is already (N, L).
    """
    if contribs.ndim == 2:
        return contribs
    return (contribs * sequences).sum(axis=1)


def compute_importance(df, contribs, peak_row_index, peak_region_starts):
    """sum(|contribution|) over each row's [start, end) span, in its own
    peak's local coordinates. Vectorized per-row via peak_region_start,
    since spans are ragged (can't be sliced as one array op).
    """
    out = np.empty(len(df), dtype=np.float64)
    for i, row in enumerate(df.itertuples(index=False)):
        r = peak_row_index[int(row.peak_id)]
        local_start = row.start - peak_region_starts[r]
        local_end = row.end - peak_region_starts[r]
        out[i] = np.abs(contribs[r, local_start:local_end]).sum()
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e", "--experiment", type=str, required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
    )
    parser.add_argument(
        "-m", "--model-dir", type=str, default=None,
        help="override model directory, default derived from config",
    )
    parser.add_argument(
        "--head", type=str, default="profile", choices=["profile", "count"],
        help="attribution/motif head the hits were called against (default: profile)",
    )
    parser.add_argument(
        "--min-trim-len", type=int, default=None, metavar="BP",
        help=(
            "must match the value hitcall/launch.py was run with, if any -- "
            "resolves the same per-experiment trim-coords-suffixed output "
            "directory rather than the plain {model_dir_name}_{head}/ one."
        ),
    )
    parser.add_argument(
        "--score-column", type=str, default=DEFAULT_SCORE_COLUMN,
        choices=["hit_importance", "hit_score_combo"],
        help=(
            f"per-hit column to threshold (default: {DEFAULT_SCORE_COLUMN}). "
            "hit_score_combo = hit_correlation * hit_importance (Kelly "
            "Cochran's ProCapNet definition); its seqlet-side reference "
            "approximates seqlet correlation as ~1.0 (see module docstring)"
        ),
    )
    parser.add_argument(
        "--percentile", type=float, default=DEFAULT_PERCENTILE,
        help=f"seqlet-importance percentile to anchor the floor to (default: {DEFAULT_PERCENTILE})",
    )
    parser.add_argument(
        "--percentile-multiplier", type=float, default=DEFAULT_PERCENTILE_MULTIPLIER,
        help=(
            "multiply the seqlet percentile by this to get the actual floor "
            f"(default: {DEFAULT_PERCENTILE_MULTIPLIER}, i.e. half the weakest "
            "1% of discovery seqlets)"
        ),
    )
    parser.add_argument(
        "--min-seqlets", type=int, default=DEFAULT_MIN_SEQLETS,
        help=(
            "skip a motif entirely if it has fewer seqlets than this -- too "
            f"few for a reliable percentile (default: {DEFAULT_MIN_SEQLETS})"
        ),
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="skip the hit_importance self-check against regions.npz (not recommended)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    model_dir_name = Path(args.model_dir).name if args.model_dir else args.experiment

    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    trim_coords = (
        modisco_dir / f"{args.experiment}_{args.head}_trim_coords_min{args.min_trim_len}bp.tsv"
        if args.min_trim_len is not None
        else None
    )
    suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, trim_coords)
    exp_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{args.head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir

    regions_npz = exp_dir / "regions.npz"
    seqlets_path = hits_dir / "report" / "seqlets.tsv"
    hits_path = resolve_hits_path(
        hits_dir,
        stages=["hits_confidence_filtered.tsv", "hits_dedensified.tsv", "hits_unique.tsv"],
        verbose=args.verbose,
    )

    for path, label, hint in [
        (regions_npz, "regions.npz", "Run call_hits_bpnet.py first."),
        (seqlets_path, "report/seqlets.tsv", "Run report_bpnet.py first (writes seqlets.tsv as a side effect even with cwm_similarity QC enabled)."),
    ]:
        if not path.exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            print(hint, file=sys.stderr)
            sys.exit(1)
    if hits_path is None:
        print(f"Error: no hits found in {hits_dir}", file=sys.stderr)
        print("Run call_hits_bpnet.py first.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Reading hits from {hits_path}")
        print(f"Reading seqlets from {seqlets_path}")
        print(f"Reading regions from {regions_npz}")

    sequences, contribs, peaks_df, _ = load_regions_npz(str(regions_npz))
    contribs = project_contribs(contribs, sequences)
    peak_row_index = build_peak_row_index(peaks_df)
    peak_region_starts = peaks_df["peak_region_start"].to_numpy()

    hits = pd.read_csv(hits_path, sep="\t")
    seqlets = pd.read_csv(seqlets_path, sep="\t")

    # Self-check: recompute hit_importance for the existing hits directly
    # from regions.npz and compare against Fi-NeMo's own recorded value.
    if not args.skip_validation:
        sample = hits if len(hits) <= 20000 else hits.sample(20000, random_state=0)
        recomputed = compute_importance(sample, contribs, peak_row_index, peak_region_starts)
        actual = sample["hit_importance"].to_numpy()
        corr = np.corrcoef(recomputed, actual)[0, 1]
        # Correlation alone can't catch a constant additive/multiplicative
        # offset (still perfectly correlated) -- median relative error
        # catches that, and is robust to a few near-zero actual values
        # blowing up individual ratios the way a max() would.
        rel_err = np.abs(recomputed - actual) / np.maximum(np.abs(actual), 1e-6)
        median_rel_err = np.median(rel_err)
        if args.verbose:
            print(
                f"Validation: recomputed vs. reported hit_importance "
                f"correlation={corr:.6f}, median_rel_err={median_rel_err:.6f} "
                f"(n={len(sample)})"
            )
        if corr < VALIDATION_MIN_CORRELATION or median_rel_err > VALIDATION_MAX_MEDIAN_REL_ERR:
            print(
                f"Error: recomputed hit_importance doesn't match Fi-NeMo's "
                f"own reported values closely enough (correlation={corr:.6f}, "
                f"need >= {VALIDATION_MIN_CORRELATION}; median_rel_err="
                f"{median_rel_err:.6f}, need <= {VALIDATION_MAX_MEDIAN_REL_ERR}). "
                "Refusing to trust seqlet-importance thresholds derived the "
                "same way. Pass --skip-validation to override, but "
                "investigate first -- this usually means a mismatch in how "
                "contribs/coordinates are being read.",
                file=sys.stderr,
            )
            sys.exit(1)

    seqlet_importance = compute_importance(seqlets, contribs, peak_row_index, peak_region_starts)
    seqlets = seqlets.assign(_importance=seqlet_importance)

    if args.score_column == "hit_score_combo":
        hit_scores = hits["hit_correlation"] * hits["hit_importance"]
        if args.verbose:
            print(
                "Using hit_score_combo = hit_correlation * hit_importance; "
                "seqlet-side reference approximates seqlet correlation as "
                "~1.0 (not computed -- see module docstring)."
            )
    else:
        hit_scores = hits["hit_importance"]

    keep_mask = np.ones(len(hits), dtype=bool)
    filtered_motifs = []
    skipped_motifs = []
    for motif_name, seqlet_group in seqlets.groupby("motif_name"):
        if len(seqlet_group) < args.min_seqlets:
            skipped_motifs.append((motif_name, len(seqlet_group)))
            continue

        floor = np.percentile(seqlet_group["_importance"], args.percentile) * args.percentile_multiplier
        hit_idx = hits.index[hits["motif_name"] == motif_name]
        if len(hit_idx) == 0:
            continue

        below_floor = hit_idx[hit_scores.loc[hit_idx] < floor]
        keep_mask[below_floor] = False
        filtered_motifs.append(
            dict(
                motif_name=motif_name,
                floor=floor,
                n_seqlets=len(seqlet_group),
                n_total=len(hit_idx),
                n_dropped=len(below_floor),
            )
        )

    kept = hits[keep_mask]

    if filtered_motifs:
        print(
            f"Applied a seqlet-anchored floor ({args.percentile_multiplier}x "
            f"the {args.percentile}th percentile of seqlet importance) by "
            f"{args.score_column} to {len(filtered_motifs)} motif(s):"
        )
        for r in sorted(filtered_motifs, key=lambda r: -r["n_dropped"]):
            frac = r["n_dropped"] / r["n_total"] if r["n_total"] else 0.0
            print(
                f"  {r['motif_name']}: floor={r['floor']:.4f} "
                f"({r['n_seqlets']} seqlets) -- dropped {r['n_dropped']}/"
                f"{r['n_total']} hits ({frac:.1%})"
            )
    else:
        print("No motif had enough seqlets to anchor a floor; nothing dropped.")

    if args.verbose and skipped_motifs:
        print(
            f"\n{len(skipped_motifs)} motif(s) skipped (fewer than "
            f"{args.min_seqlets} seqlets): {skipped_motifs}"
        )

    out_path = hits_dir / "hits_seqlet_filtered.tsv"
    kept.to_csv(out_path, sep="\t", index=False)
    print(
        f"\nKept {len(kept)}/{len(hits)} hits "
        f"({len(hits) - len(kept)} dropped from {len(filtered_motifs)} motifs)"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
