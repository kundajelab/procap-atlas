"""Drop hits whose sequence context outside the trimmed core doesn't match
the motif's real flanking pattern -- even though the trimmed core itself
fits well.

Direct review of the real CWM comparison data (report_bpnet.py's
hits_fc.txt vs. modisco_fc.txt) for TATA/GATA/TA-Inr in K562 ENCSR220XSM
showed a consistent signature: the trimmed core (the ~6bp window
--min-trim-len/cwm-trim-threshold actually fits against) matches almost
perfectly (per-position cosine similarity 0.85-1.0), but the ~44bp of
flanking sequence outside that core is dominated by generic AT-repeat
content that actively disagrees with the real motif's flanking pattern
(cosine similarity down to -0.92 for GATA). filter_by_seqlet_importance.py
and filter_low_confidence_hits.py were both structural no-ops on this
problem because every score Fi-NeMo computes per hit (hit_correlation,
hit_importance, hit_similarity) is computed only over that same trimmed
core -- there is no per-hit signal that ever looks at the flanks, so no
threshold on those columns can separate a real motif in its real context
from a core-matching hexamer sitting in the wrong (repeat/AT-rich) context.

Forcing a wider fitting window at call-hits time (raising
--min-trim-len/--cwm-trim-threshold globally) was considered and rejected:
CWM magnitude decays gradually and similarly across nearly every motif, so
there's no single threshold that widens only the broken motifs -- it
widens (and adds fitting collinearity risk to) every motif in every
experiment. See the PR discussion / project memory for the numbers.

This script instead adds the missing signal as a separate, motif-identity-
agnostic post-hoc check: for each hit, extend out to the *full* (untrimmed)
CWM window -- hits already carry start_untrimmed/end_untrimmed for exactly
this span, confirmed against Fi-NeMo's own write_hits() -- and compute
cosine similarity between the hit's own observed contribution track and
the motif's full CWM (loaded directly from the .modisco.h5 via
finemo.data_io.load_modisco_motifs, motif_type="cwm", matching the "pp"
hit-calling mode this pipeline uses: both sides are the *projected*
(true-base-only) contribution, not the hypothetical one). This is exactly
the per-instance analog of what produces hits_fc.txt/cwm_similarity in
aggregate, just computed per hit instead of averaged over all of them --
so it directly measures the thing cwm_similarity fails on, unlike
hit_importance/hit_correlation.

The floor is anchored to each motif's own TF-MoDISco discovery seqlets
(same architecture as filter_by_seqlet_importance.py): seqlets carry the
same start_untrimmed/end_untrimmed/strand schema, so the identical
full-window similarity score is computable for them too, giving a
ground-truth reference distribution per motif with no consensus/JASPAR
identity knowledge involved.

Run after filter_by_seqlet_importance.py (reads its output if present) and
before report_bpnet.py, which prefers this script's output
(hits_flank_filtered.tsv) when present. Requires report_bpnet.py to have
already been run at least once for this experiment/head, to produce
report/seqlets.tsv.

Usage:
    python src/bpnet/hitcall/filter_by_flank_consistency.py -e ENCSR220XSM
    python src/bpnet/hitcall/filter_by_flank_consistency.py -e ENCSR220XSM --min-trim-len 6 -v
    python src/bpnet/hitcall/filter_by_flank_consistency.py -e ENCSR220XSM --percentile 1 --percentile-multiplier 0.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from finemo.data_io import load_modisco_motifs, load_regions_npz

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, resolve_hits_path, trim_suffix
from filter_by_seqlet_importance import build_peak_row_index, project_contribs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_PERCENTILE = 1.0
DEFAULT_PERCENTILE_MULTIPLIER = 0.5
DEFAULT_MIN_SEQLETS = 20


def build_cwm_lookup(modisco_h5_path):
    """(motif_name, strand) -> full, untrimmed (4, W) CWM, matching the
    'cwm' (projected, not hypothetical) motif type this pipeline's default
    "pp" call-hits mode fits against. trim_threshold_default is passed but
    unused here -- it only affects the trim-boundary metadata columns this
    script ignores, never the returned CWM arrays themselves.
    """
    motifs_df, cwms, _, _ = load_modisco_motifs(
        modisco_h5_path=str(modisco_h5_path),
        trim_coords=None,
        trim_thresholds=None,
        trim_threshold_default=DEFAULT_CWM_TRIM_THRESHOLD,
        motif_type="cwm",
        motifs_include=None,
        motif_name_map=None,
        motif_lambdas=None,
        motif_lambda_default=0.0,
        include_rc=True,
    )
    motifs_pd = motifs_df.to_pandas()
    lookup = {
        (row.motif_name, row.strand): cwms[i]
        for i, row in enumerate(motifs_pd.itertuples(index=False))
    }
    motif_width = cwms.shape[2]
    return lookup, motif_width


def compute_flank_similarity(
    df, contribs, sequences, peak_row_index, peak_region_starts, cwm_lookup, motif_width
):
    """Cosine similarity between each row's observed, projected contribution
    track over its full [start_untrimmed, end_untrimmed) span and the
    motif's own full CWM projected through that same row's true sequence --
    the per-instance analog of what aggregates into hits_fc.txt. NaN for
    rows whose motif/strand isn't found or whose untrimmed span falls
    outside the saved contribution track (peak-edge cases) -- callers treat
    NaN as "unscoreable, don't drop" rather than as a failure.
    """
    n = len(df)
    scores = np.full(n, np.nan, dtype=np.float64)
    for i, row in enumerate(df.itertuples(index=False)):
        cwm = cwm_lookup.get((row.motif_name, row.strand))
        if cwm is None:
            continue
        r = peak_row_index.get(int(row.peak_id))
        if r is None:
            continue
        local_start = row.start_untrimmed - peak_region_starts[r]
        local_end = local_start + motif_width
        if local_start < 0 or local_end > contribs.shape[1]:
            continue

        seq_slice = sequences[r, :, local_start:local_end]
        obs = contribs[r, local_start:local_end]
        cwm_proj = (cwm.astype(np.float64) * seq_slice).sum(axis=0)

        denom = np.linalg.norm(obs) * np.linalg.norm(cwm_proj)
        if denom <= 0:
            continue
        scores[i] = float(np.dot(obs, cwm_proj) / denom)
    return scores


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
        "--modisco-h5", type=str, default=None,
        help="override path to the .modisco.h5 file, default derived from config",
    )
    parser.add_argument(
        "--percentile", type=float, default=DEFAULT_PERCENTILE,
        help=f"seqlet full-window-similarity percentile to anchor the floor to (default: {DEFAULT_PERCENTILE})",
    )
    parser.add_argument(
        "--percentile-multiplier", type=float, default=DEFAULT_PERCENTILE_MULTIPLIER,
        help=(
            "how much slack to give below the seqlet percentile, applied as "
            "a distance-from-a-perfect-match (1.0) scaling rather than a "
            "plain multiplier -- this metric is a cosine similarity bounded "
            "at 1.0 and real seqlets cluster close to it (unlike "
            "hit_importance's unbounded magnitude), so 'multiply the "
            "percentile by 0.5' would barely move the floor at all. Floor = "
            "1 - (1 - percentile_value) / multiplier, so smaller values are "
            f"more lenient, same direction as before (default: "
            f"{DEFAULT_PERCENTILE_MULTIPLIER}, i.e. allow twice the distance "
            "from a perfect match that the weakest 1% of seqlets show)"
        ),
    )
    parser.add_argument(
        "--min-seqlets", type=int, default=DEFAULT_MIN_SEQLETS,
        help=(
            "skip a motif entirely if it has fewer seqlets than this -- too "
            f"few for a reliable percentile (default: {DEFAULT_MIN_SEQLETS})"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.percentile_multiplier <= 0:
        print("Error: --percentile-multiplier must be > 0", file=sys.stderr)
        sys.exit(1)

    with open(REPO_ROOT / "configs" / "experiment_config.yaml") as f:
        config = yaml.safe_load(f)
    if args.experiment not in config["experiments"]:
        print(f"Error: {args.experiment} not found in config", file=sys.stderr)
        sys.exit(1)

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
    modisco_h5 = (
        Path(args.modisco_h5) if args.modisco_h5
        else modisco_dir / f"{args.experiment}_{args.head}.modisco.h5"
    )
    hits_path = resolve_hits_path(
        hits_dir,
        stages=["hits_seqlet_filtered.tsv", "hits_confidence_filtered.tsv", "hits_dedensified.tsv", "hits_unique.tsv"],
        verbose=args.verbose,
    )

    for path, label, hint in [
        (regions_npz, "regions.npz", "Run call_hits_bpnet.py first."),
        (seqlets_path, "report/seqlets.tsv", "Run report_bpnet.py first (writes seqlets.tsv as a side effect even with cwm_similarity QC enabled)."),
        (modisco_h5, "motif CWMs (.modisco.h5)", "Run MoDISco first."),
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
        print(f"Reading motif CWMs from {modisco_h5}")

    sequences, contribs, peaks_df, _ = load_regions_npz(str(regions_npz))
    contribs = project_contribs(contribs, sequences)
    peak_row_index = build_peak_row_index(peaks_df)
    peak_region_starts = peaks_df["peak_region_start"].to_numpy()
    cwm_lookup, motif_width = build_cwm_lookup(modisco_h5)

    hits = pd.read_csv(hits_path, sep="\t")
    seqlets = pd.read_csv(seqlets_path, sep="\t")

    hit_scores = compute_flank_similarity(
        hits, contribs, sequences, peak_row_index, peak_region_starts, cwm_lookup, motif_width
    )
    seqlet_scores = compute_flank_similarity(
        seqlets, contribs, sequences, peak_row_index, peak_region_starts, cwm_lookup, motif_width
    )
    n_unscoreable = int(np.isnan(hit_scores).sum())
    if args.verbose or n_unscoreable:
        print(
            f"{n_unscoreable}/{len(hits)} hits ({n_unscoreable / max(len(hits), 1):.1%}) "
            "could not be scored (untrimmed span outside the saved contribution "
            "track, e.g. near a peak edge) -- kept regardless of score"
        )

    # Soft sanity check: this new metric has no exact closed-form ground
    # truth to validate against the way hit_importance did, but if
    # report_bpnet.py has already run, its motif_report.tsv gives an
    # independent cwm_similarity per motif we can eyeball this against --
    # motifs with low cwm_similarity should show a lower/wider-spread mean
    # hit-level full-window similarity too.
    motif_report_path = hits_dir / "report" / "motif_report.tsv"
    if args.verbose and motif_report_path.exists():
        motif_report = pd.read_csv(motif_report_path, sep="\t").set_index("motif_name")
        print("\nSanity check vs. existing report/motif_report.tsv (not a hard gate):")
        print(f"  {'motif_name':<24} {'cwm_similarity':>14} {'mean_flank_sim':>14}")
        for motif_name, group in hits.groupby("motif_name"):
            if motif_name not in motif_report.index:
                continue
            mean_sim = np.nanmean(hit_scores[group.index.to_numpy()])
            print(
                f"  {motif_name:<24} {motif_report.loc[motif_name, 'cwm_similarity']:>14.3f} "
                f"{mean_sim:>14.3f}"
            )

    keep_mask = np.ones(len(hits), dtype=bool)
    filtered_motifs = []
    skipped_motifs = []
    for motif_name, seqlet_group in seqlets.groupby("motif_name"):
        seqlet_idx = seqlet_group.index.to_numpy()
        group_scores = seqlet_scores[seqlet_idx]
        group_scores = group_scores[~np.isnan(group_scores)]
        if len(group_scores) < args.min_seqlets:
            skipped_motifs.append((motif_name, len(group_scores)))
            continue

        percentile_value = np.percentile(group_scores, args.percentile)
        floor = 1.0 - (1.0 - percentile_value) / args.percentile_multiplier
        hit_idx = hits.index[hits["motif_name"] == motif_name]
        if len(hit_idx) == 0:
            continue

        below_floor = hit_idx[hit_scores[hit_idx] < floor]
        keep_mask[below_floor] = False
        filtered_motifs.append(
            dict(
                motif_name=motif_name,
                floor=floor,
                n_seqlets=len(group_scores),
                n_total=len(hit_idx),
                n_dropped=len(below_floor),
            )
        )

    kept = hits[keep_mask]

    if filtered_motifs:
        print(
            f"Applied a seqlet-anchored full-window-similarity floor "
            f"({args.percentile_multiplier}x the {args.percentile}th percentile "
            f"of seqlet full-window similarity) to {len(filtered_motifs)} motif(s):"
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
            f"{args.min_seqlets} scoreable seqlets): {skipped_motifs}"
        )

    out_path = hits_dir / "hits_flank_filtered.tsv"
    kept.to_csv(out_path, sep="\t", index=False)
    print(
        f"\nKept {len(kept)}/{len(hits)} hits "
        f"({len(hits) - len(kept)} dropped from {len(filtered_motifs)} motifs)"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
