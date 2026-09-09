"""Drop the low-confidence mode of a motif's hits, when one is detectable.

Reviewed real per-hit hit_correlation distributions for K562 ENCSR220XSM's
profile head (three motifs failing report_bpnet.py's cwm_similarity QC even
after filter_repeat_density.py): GATA showed a clear bimodal split -- a large
bulk of ambiguous hits plus a distinct, separable population of very
high-confidence hits (correlation trough around the 90th percentile, then
rising again toward the max) -- while TA-Initiator showed no such split at
all (a smooth, monotonically decaying unimodal distribution). That matters:
a motif's aggregate cwm_similarity can be dragged down by a large low-
confidence tail even when a real, legitimate high-confidence subset exists
under the same motif_name, but filtering only helps where that split is
actually there to find.

This is deliberately identity-agnostic (no hardcoded motif names/thresholds,
same reasoning as filter_repeat_density.py): for each motif, it builds a
histogram of --score-column (default hit_correlation), looks for a genuine
local dip-then-rise (trough after the primary mode, followed by a
sufficiently large and sufficiently separated secondary mode), and drops
hits below that trough only when one is found. Motifs with a smooth/unimodal
distribution (e.g. TA-Inr in the case above) are left untouched -- there's no
data-driven cutoff to apply, and filtering them anyway would just be an
arbitrary top-K cut with no principled justification.

--score-column hit_flank_similarity computes a score not natively present in
hits.tsv: filter_by_flank_consistency.py's per-hit cosine similarity between
the observed contribution track over a hit's *full* (untrimmed) CWM window
and the motif's full CWM -- unlike hit_correlation/hit_importance, which are
computed only over the trimmed core and so can't see a real-motif-in-the-
wrong-flanking-context problem (see that script's docstring). Anchoring that
score's drop floor to TF-MoDISco discovery seqlets (filter_by_
seqlet_importance.py's approach) turned out to be the wrong reference
population for it: real per-hit data (K562 ENCSR220XSM) showed seqlets
scoring systematically *lower* than hits on this metric for every motif
checked, including clearly healthy ones -- expected, since MoDISco seqlets
are an intentionally diverse cluster of variant/degenerate instances
averaged into one consensus CWM, while Fi-NeMo hit-calling's sparse
regression explicitly searches for windows maximizing fit to that one
template. Detecting bimodality within the hit population itself (this
script's existing machinery) is the right level to look for a real/noise
split on this score, the same way it already is for hit_correlation.

--score-column hit_summit_proximity is a different, non-attribution-based
axis entirely: diagnose_hit_summit_distance.py's signed distance from each
hit to the real PRO-cap TSS summit (not Fi-NeMo's synthetic peak-midpoint
"summit"), negated and made unsigned (-abs(signed_distance)) so higher is
still "better"/closer, matching every other score column's convention here.
A real core-promoter element like TATA should sit at a fixed, narrow offset
upstream of the TSS; repeat-context noise elsewhere in the peak has no
reason to respect that offset. Checks for a genuine narrow near-summit mode
sitting on top of an otherwise diffuse/background spread, the same
find_peaks-based logic as every other score here, just applied to this
different signal in case it's separable even where attribution-based scores
(hit_correlation/hit_importance/hit_flank_similarity) weren't.

--score-column hit_seq_complexity is a third, sequence-intrinsic axis --
independent of attribution magnitude, shape, and position entirely. Direct
visual review of real ENCSR220XSM TATA hits (logo plots of the actual
observed contribution track, dump_tata_logos.py-style) showed a clean,
consistent qualitative difference: real core hits (~-30bp from the TSS)
show one compact, isolated "TATAAA"-like word sitting in an otherwise quiet
flanking background, while both the unexplained downstream-hump hits and
far-background hits sit inside long, dense, low-complexity stretches --
extended homopolymer or dinucleotide-repeat runs spanning most or all of
the window, with no single discrete feature standing out. Fi-NeMo's own
optimizer (hitcaller.py's fit_contribs/prox_grad_step) has no mechanism
that would catch this: sparsity comes only from a single global per-motif
L1 penalty and a global per-hit correlation floor, never a check on
whether a position's importance stands out from its own local
neighborhood, so a long repetitive stretch can keep yielding weak-but-
passing hits indefinitely. hit_seq_complexity is
-(longest homopolymer-or-dinucleotide-repeat run within
--complexity-window bp of the hit's center, decoded directly from
regions.npz's one-hot sequence), so a short run (compact motif in normal-
complexity flanks) scores near 0 and a long repeat run scores very
negative, matching every other score column's higher-is-better convention.
Needs only regions.npz, no other files.

--score-column hit_seqlet_confidence is a fourth axis, and the first that
isn't blind to *local prominence relative to background*. Direct visual
review of a larger, unbiased logo sample (dump_tata_logos_v2.py, 20 hits per
signed-distance category instead of the original 6) showed the real
distinguishing property isn't sequence-level repeat structure after all
(hit_seq_complexity came back null on real data, no separation between
categories) -- it's whether the hit's own attribution track is a genuine
local spike or just an unremarkable blip inside generally noisy background,
and critically, *both* types occur in every position category (near-summit,
downstream-hump, far-background alike), which is exactly why every position-
based and magnitude-based score tried so far averaged the signal away.
tangermeme.seqlet.recursive_seqlets is an independent seqlet caller whose
"recursive" property requires every internal sub-span down to
min_seqlet_len to also independently pass, giving a real statement about
local prominence -- this is exactly the mechanism Fi-NeMo's own optimizer
(hitcaller.py's fit_contribs/prox_grad_step) lacks (sparsity comes only from
a single global per-motif L1 penalty and a global per-hit correlation
floor, confirmed directly from source -- never a comparison to a hit's own
local neighborhood), and matches why prior methods without a shared global
sparsity fit (recursive_seqlets itself, CWM scanning) didn't show this
overcalling problem for TATA. The CLIPNET paper (Cochran/Cochran-adjacent
methods, via Schreiber2025-bb) independently confirms recursive_seqlets was
used successfully for exactly these motifs: "High importance profile motifs
such as the TATA box and initiator elements were called using the
recursive_seqlets approach" at a p-value threshold of 0.05 (tangermeme's own
default is 0.01) -- see --seqlet-threshold.

Its null-distribution histogram bins the *entire* flattened input in one
global range (xmax, xmin = X.max(), X.min() over every region passed to one
call, not per-region -- found by reading recursive_seqlets' actual numba-
jitted source, not assumed), so a single extreme outlier value anywhere in
the ~90k-region genome-wide batch collapses bin resolution for every other
region too -- reproduced directly on synthetic data (one contaminating
outlier dropped calls on 100 otherwise-trivially-callable clean spikes from
100/100 to 0/100) and is exactly why the first real run of this column came
back with 100% of hits completely uncorroborated. --seqlet-clip-percentile
clips |contribs| before the call to fix this while keeping one fast,
well-powered call over the whole batch (a per-region-only null was
considered and rejected: much less powered, only ~2100 positions per
region's own histogram, vs. this approach's single global fix).

hit_seqlet_confidence = -log10(p-value) of the best (lowest-p) recursive
seqlet overlapping the hit's trimmed [start, end) span, or exactly 0.0 if
no seqlet call overlaps at all -- "no call" is a real, meaningful negative
signal here (recursive_seqlets calls nothing in flat background), not
missing/unscoreable data. Used identity-agnostically (no tomtom/motif-
matching step, unlike prior hand-tuning-heavy attempts, and unlike
CLIPNET's own full pipeline which adds Tomtom-lite identity matching plus a
seqlet-importance floor on top): we only ask whether *any* locally
prominent attribution feature overlaps the hit's own trimmed span, not what
it matches, since Fi-NeMo's own hit_correlation already establishes the
CWM-shape match.

Real per-hit data (K562 ENCSR220XSM, threshold=0.05, clip_percentile=99.99)
split by ground-truth signed distance to the real PRO-cap TSS summit showed
corroboration RATE differs ~2-3x between near-summit "core" hits for both
TATA and TA-Inr (~58-59%) and their downstream-hump/far-background hits
(~20-29%) -- real, useful structure -- while the score's *magnitude* when
corroborated (mean ~1.34-1.41) is nearly identical across every category:
all the information is in whether a call exists at all, not in how strong
it is. detect_low_confidence_cutoff's find_peaks-based trough-then-rise
search is the wrong tool for this shape (a spike at exactly 0.0 followed by
a monotonically *decaying* tail, not a dip-then-second-peak) and reports
"no secondary mode" even though this real structure exists, so this column
bypasses that machinery entirely and applies a direct, unconditional
hit_seqlet_confidence > 0 floor to every motif with enough hits instead
(--log-scale/--n-bins/--smoothing-window/--min-rise-frac/--min-bin-frac are
all ignored for this column). This is not a clean separator (~41% of real
core hits would still be dropped as false negatives; ~20-29% of spurious
hits still survive as corroborated by chance), and CLIPNET's own methods
note recursive_seqlets "struggled to consistently identify relatively low
importance profile motifs" (DPR, activator elements) independent of any
contamination problem, so this floor costs real recall there too --
accepted deliberately as a scaling decision: this needs to run identity-
agnostically across the whole atlas, and consistent recall on core promoter
motifs (TATA/Inr) was judged to matter more than recall on low-importance
profile motifs. More compute-intensive than the other columns -- runs a
full recursive_seqlets pass over every region in regions.npz once.

Run after filter_repeat_density.py (reads hits_dedensified.tsv if present,
else hits_unique.tsv) and before report_bpnet.py, which prefers this script's
output (hits_confidence_filtered.tsv) when present.

Usage:
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --head count
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --min-trim-len 6
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --score-column hit_similarity
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --score-column hit_importance --log-scale
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --score-column hit_flank_similarity
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --score-column hit_summit_proximity
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --score-column hit_seq_complexity
    python src/bpnet/hitcall/filter_low_confidence_hits.py -e ENCSR882DWM --score-column hit_seqlet_confidence
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from finemo.data_io import load_regions_npz
from scipy.signal import find_peaks
from tangermeme.seqlet import recursive_seqlets

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, resolve_hits_path, trim_suffix
from diagnose_hit_summit_distance import (
    build_summit_lookup,
    compute_distances,
    infer_coordinate_mode,
    load_filtered_peaks,
)
from filter_by_flank_consistency import build_cwm_lookup, compute_flank_similarity
from filter_by_seqlet_importance import build_peak_row_index, project_contribs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
DEFAULT_SCORE_COLUMN = "hit_correlation"
FLANK_SIMILARITY_COLUMN = "hit_flank_similarity"
SUMMIT_PROXIMITY_COLUMN = "hit_summit_proximity"
SEQ_COMPLEXITY_COLUMN = "hit_seq_complexity"
DEFAULT_COMPLEXITY_WINDOW = 100
SEQLET_CONFIDENCE_COLUMN = "hit_seqlet_confidence"
DEFAULT_SEQLET_THRESHOLD = 0.05
DEFAULT_MIN_SEQLET_LEN = 4
DEFAULT_MAX_SEQLET_LEN = 25
DEFAULT_SEQLET_ADDITIONAL_FLANKS = 0
DEFAULT_SEQLET_CLIP_PERCENTILE = 99.99
DEFAULT_N_BINS = 50
DEFAULT_SMOOTHING_WINDOW = 3
DEFAULT_MIN_RISE_FRAC = 1.15
DEFAULT_MIN_BIN_FRAC = 0.01
DEFAULT_MIN_TOTAL_HITS = 200
BASE_ALPHABET = np.array(list("ACGT"))


def decode_sequence(seq_slice):
    """(4, W) one-hot -> ACGT string. Positions with no base set (all-zero
    column, e.g. padding) decode to whatever argmax(0) picks -- callers
    should only pass slices fully inside a real region.
    """
    idx = np.argmax(seq_slice, axis=0)
    return "".join(BASE_ALPHABET[idx])


def longest_periodic_run(seq_str, period):
    """Longest run of a period-`period` repeat (period=1: homopolymer,
    period=2: dinucleotide repeat like ATATAT...) anywhere in seq_str.
    """
    n = len(seq_str)
    if n <= period:
        return n
    best = period
    cur = period
    for i in range(period, n):
        if seq_str[i] == seq_str[i - period]:
            cur += 1
        else:
            best = max(best, cur)
            cur = period
    return max(best, cur)


def compute_seq_complexity(df, sequences, peak_row_index, peak_region_starts, window=DEFAULT_COMPLEXITY_WINDOW):
    """-(longest homopolymer-or-dinucleotide-repeat run within `window` bp
    of each row's center), decoded directly from the one-hot sequence in
    regions.npz. NaN if too close to the region's own edge to get a full
    window on both sides.
    """
    n = len(df)
    scores = np.full(n, np.nan)
    region_width = sequences.shape[2]
    for i, row in enumerate(df.itertuples(index=False)):
        r = peak_row_index.get(int(row.peak_id))
        if r is None:
            continue
        center = int((row.start + row.end) / 2) - peak_region_starts[r]
        lo = center - window
        hi = center + window
        if lo < 0 or hi > region_width:
            continue
        seq_str = decode_sequence(sequences[r, :, lo:hi])
        run1 = longest_periodic_run(seq_str, 1)
        run2 = longest_periodic_run(seq_str, 2)
        scores[i] = -float(max(run1, run2))
    return scores


def compute_seqlet_confidence(
    df,
    contribs,
    peak_row_index,
    peak_region_starts,
    threshold=DEFAULT_SEQLET_THRESHOLD,
    min_seqlet_len=DEFAULT_MIN_SEQLET_LEN,
    max_seqlet_len=DEFAULT_MAX_SEQLET_LEN,
    additional_flanks=DEFAULT_SEQLET_ADDITIONAL_FLANKS,
    clip_percentile=DEFAULT_SEQLET_CLIP_PERCENTILE,
):
    """-log10(p-value) of the best (lowest-p) tangermeme.seqlet.
    recursive_seqlets call overlapping each row's trimmed [start, end) span,
    or exactly 0.0 if no call overlaps at all. `contribs` must be the full
    (n_regions, region_width) projected contribution track from regions.npz,
    row-aligned with peak_row_index/peak_region_starts, same convention as
    every other score column here. Absolute value is passed to
    recursive_seqlets since it only identifies *positive* seqlets (per its
    own docstring) and contributions can be negative for repressive motifs.
    Explicitly upcast to float32 first: regions.npz stores contribs as
    float16 to save space, but recursive_seqlets' numba-jitted core has no
    float16 array type support (NotImplementedError at compile time) --
    found by hand running this against real data.

    Clips |contribs| to `clip_percentile` before the call. Found by hand on
    real data: recursive_seqlets' internal histogram bins the *entire*
    flattened input in one global range (`xmax, xmin = X.max(), X.min()`
    over all regions at once, not per-region), so a single extreme outlier
    value anywhere in the whole genome-wide array blows out bin_width and
    collapses every other region's real signal into a handful of near-zero
    bins -- reproduced directly on synthetic data: one contaminating outlier
    dropped calls on 100 otherwise-trivially-callable clean spikes from
    100/100 to 0/100, and clipping first restored calling to 101/100. This
    is why the real ENCSR220XSM run first came back with 100% of hits
    completely uncalled -- a global batching artifact, not biology.

    additional_flanks pads each called seqlet's returned start/end
    symmetrically *after* calling, per tangermeme's own Tutorial_A4_Seqlets:
    it doesn't affect which seqlets get called, the threshold, or the
    returned p-value/attribution -- only the boundaries used here for
    overlap testing. The recursive property requires every internal
    sub-span down to min_seqlet_len to also independently pass, which the
    tutorial documents as making call boundaries conservative (a real motif
    core with even a brief dip in one flanking position can fail to extend
    to its full width); additional_flanks is tangermeme's own sanctioned fix
    for exactly that, without loosening calling sensitivity the way raising
    threshold would.
    """
    contribs_abs = np.abs(contribs).astype(np.float32)
    clip_val = np.percentile(contribs_abs, clip_percentile)
    called = recursive_seqlets(
        np.clip(contribs_abs, 0, clip_val),
        threshold=threshold,
        min_seqlet_len=min_seqlet_len,
        max_seqlet_len=max_seqlet_len,
        additional_flanks=additional_flanks,
    ).rename(columns={"p-value": "pvalue"})

    by_row = {}
    for row in called.itertuples(index=False):
        by_row.setdefault(int(row.example_idx), []).append(
            (int(row.start), int(row.end), float(row.pvalue))
        )

    n = len(df)
    scores = np.zeros(n, dtype=np.float64)
    for i, row in enumerate(df.itertuples(index=False)):
        r = peak_row_index.get(int(row.peak_id))
        if r is None:
            continue
        local_start = int(row.start - peak_region_starts[r])
        local_end = int(row.end - peak_region_starts[r])
        best_p = None
        for s_start, s_end, p in by_row.get(r, []):
            if s_start < local_end and s_end > local_start:
                if best_p is None or p < best_p:
                    best_p = p
        if best_p is not None:
            scores[i] = -np.log10(max(best_p, 1e-300))
    return scores


def detect_low_confidence_cutoff(
    scores,
    n_bins=DEFAULT_N_BINS,
    smoothing_window=DEFAULT_SMOOTHING_WINDOW,
    min_rise_frac=DEFAULT_MIN_RISE_FRAC,
    min_bin_frac=DEFAULT_MIN_BIN_FRAC,
    min_total_hits=DEFAULT_MIN_TOTAL_HITS,
    log_scale=False,
):
    """Look for a genuine secondary (higher-score) mode past a trough
    following the primary mode of `scores`'s histogram.

    Returns a dict with the cutoff and diagnostic info if found, else None.
    Uses scipy's prominence-based peak finder rather than hand-rolled local-
    or global-extremum logic, which has two failure modes that are hard to
    dodge simultaneously: chasing the *first* local minimum latches onto
    tiny noise wiggles inside a noisy-but-still-descending bulk region, while
    chasing the *global* minimum of the tail gets fooled by the fact that any
    histogram's most extreme bins are sparse and decay toward ~0 anyway, so
    that edge can look like an even deeper "trough" than the real one
    between two modes. Prominence (how much a peak stands out above its
    surrounding valleys, not just its raw height) is robust to both.

    log_scale bins in log10-space instead of linear -- needed for a heavy
    right-skewed, effectively-unbounded column like hit_importance (e.g.
    median ~0.07 but max in the double digits): linear bins over that full
    range would compress nearly the entire real distribution into a handful
    of bins near zero, both hiding a genuine secondary mode and making any
    "peak" found in the sparse, spread-out tail untrustworthy. Non-positive
    scores can't be log-transformed and are excluded from detection (but
    would still end up below any cutoff found, since a cutoff is always
    positive here).

    Histogram range is padded past the data's own min/max on both ends,
    rather than binning over exactly (min, max) the way np.histogram
    defaults to -- scipy's find_peaks can never flag the first or last
    array element as a peak (no neighbor to compare against on the open
    side), so a genuine, real secondary mode sitting very close to a score
    column's own natural bound (e.g. a cosine similarity near 1.0, or a
    proximity score near 0) would otherwise collapse into that undetectable
    edge bin and silently vanish -- found by hand while validating
    hit_summit_proximity/hit_flank_similarity on synthetic data shaped
    exactly like this. One bin of padding isn't enough on its own:
    `smoothing_window`'s convolution spreads a sharp edge spike's mass back
    outward by its own radius, refilling the padding and erasing the
    prominence cliff that padding is supposed to create, so padding scales
    with smoothing_window's radius instead of a fixed one-bin margin.
    """
    n = len(scores)
    if n < min_total_hits:
        return None

    if log_scale:
        transformed = scores[scores > 0]
        if len(transformed) < min_total_hits:
            return None
        transformed = np.log10(transformed)
    else:
        transformed = scores

    def from_log(x):
        return float(10**x) if log_scale else float(x)

    lo, hi = transformed.min(), transformed.max()
    span = hi - lo
    bin_width = span / n_bins if span > 0 else 1.0
    pad = bin_width * (smoothing_window // 2 + 1)
    counts, edges = np.histogram(transformed, bins=n_bins, range=(lo - pad, hi + pad))
    counts = counts.astype(float)
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        counts = np.convolve(counts, kernel, mode="same")

    min_prominence = max(min_bin_frac * n, 1.0)
    peak_idxs, _ = find_peaks(counts, prominence=min_prominence)
    if len(peak_idxs) < 2:
        return None  # no distinct secondary mode at all

    primary_idx = peak_idxs[np.argmax(counts[peak_idxs])]
    later_peaks = peak_idxs[peak_idxs > primary_idx]
    if len(later_peaks) == 0:
        return None  # every other mode is at a *lower* score than the primary one
    peak_idx = later_peaks[np.argmax(counts[later_peaks])]

    trough_idx = primary_idx + int(np.argmin(counts[primary_idx : peak_idx + 1]))

    trough_count = max(counts[trough_idx], 1.0)
    peak_count = counts[peak_idx]
    if peak_count < min_bin_frac * n:
        return None  # secondary mode too small relative to this motif's total hits
    if peak_count < trough_count * min_rise_frac:
        return None  # not a large enough rise to trust over histogram noise

    return dict(
        cutoff=from_log(edges[trough_idx + 1]),
        primary_mode=from_log(edges[primary_idx]),
        trough=from_log(edges[trough_idx]),
        secondary_mode=from_log(edges[peak_idx]),
        trough_count=float(counts[trough_idx]),
        secondary_count=float(peak_count),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        default=None,
        help="override model directory, default derived from config",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="profile",
        choices=["profile", "count"],
        help="attribution/motif head the hits were called against (default: profile)",
    )
    parser.add_argument(
        "--min-trim-len",
        type=int,
        default=None,
        metavar="BP",
        help=(
            "must match the value hitcall/launch.py was run with, if any -- "
            "resolves the same per-experiment trim-coords-suffixed output "
            "directory rather than the plain {model_dir_name}_{head}/ one."
        ),
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default=DEFAULT_SCORE_COLUMN,
        help=(
            f"per-hit score column to check for bimodality (default: "
            f"{DEFAULT_SCORE_COLUMN}). Any existing Fi-NeMo hits.tsv column "
            f"works, plus three special values that aren't native columns "
            f"and are computed on the fly: {FLANK_SIMILARITY_COLUMN!r} "
            "(requires regions.npz and the .modisco.h5 -- filter_by_flank_"
            f"consistency.py's per-hit full-window CWM similarity), "
            f"{SUMMIT_PROXIMITY_COLUMN!r} (requires filtered_peaks.bed.gz -- "
            "diagnose_hit_summit_distance.py's -abs(distance to the real "
            f"PRO-cap TSS summit)), {SEQ_COMPLEXITY_COLUMN!r} (requires "
            "only regions.npz -- -(longest homopolymer-or-dinucleotide-"
            f"repeat run near the hit), and {SEQLET_CONFIDENCE_COLUMN!r} "
            "(requires only regions.npz -- -log10(p) of the best "
            "tangermeme.seqlet.recursive_seqlets call overlapping the hit, "
            "or 0.0 if none overlap; see module docstring for details on "
            "all of these"
        ),
    )
    parser.add_argument(
        "--modisco-h5",
        type=str,
        default=None,
        help=(
            f"override path to the .modisco.h5 file, default derived from "
            f"config -- only used when --score-column {FLANK_SIMILARITY_COLUMN}"
        ),
    )
    parser.add_argument(
        "--complexity-window",
        type=int,
        default=DEFAULT_COMPLEXITY_WINDOW,
        help=(
            "bp on each side of the hit center to scan for a repeat run -- "
            f"only used when --score-column {SEQ_COMPLEXITY_COLUMN} "
            f"(default: {DEFAULT_COMPLEXITY_WINDOW})"
        ),
    )
    parser.add_argument(
        "--seqlet-threshold",
        type=float,
        default=DEFAULT_SEQLET_THRESHOLD,
        help=(
            "p-value threshold for tangermeme.seqlet.recursive_seqlets to "
            f"call a seqlet -- only used when --score-column "
            f"{SEQLET_CONFIDENCE_COLUMN} (default: {DEFAULT_SEQLET_THRESHOLD}, "
            "the CLIPNET paper's validated value for calling TATA/Inr this "
            "way, not tangermeme's own default of 0.01)"
        ),
    )
    parser.add_argument(
        "--min-seqlet-len",
        type=int,
        default=DEFAULT_MIN_SEQLET_LEN,
        help=(
            "minimum recursive_seqlets call length (bp) -- only used when "
            f"--score-column {SEQLET_CONFIDENCE_COLUMN} (default: "
            f"{DEFAULT_MIN_SEQLET_LEN}, tangermeme's own default)"
        ),
    )
    parser.add_argument(
        "--max-seqlet-len",
        type=int,
        default=DEFAULT_MAX_SEQLET_LEN,
        help=(
            "maximum recursive_seqlets call length (bp) -- only used when "
            f"--score-column {SEQLET_CONFIDENCE_COLUMN} (default: "
            f"{DEFAULT_MAX_SEQLET_LEN}, tangermeme's own default)"
        ),
    )
    parser.add_argument(
        "--seqlet-additional-flanks",
        type=int,
        default=DEFAULT_SEQLET_ADDITIONAL_FLANKS,
        help=(
            "bp to symmetrically pad each called seqlet's start/end by "
            "after calling, before checking overlap with a hit -- only used "
            f"when --score-column {SEQLET_CONFIDENCE_COLUMN} (default: "
            f"{DEFAULT_SEQLET_ADDITIONAL_FLANKS}, tangermeme's own default). "
            "Doesn't affect calling sensitivity/threshold at all, unlike "
            "--seqlet-threshold -- only widens the boundary used for overlap "
            "testing here, to compensate for the recursive property's "
            "documented tendency toward conservative call boundaries"
        ),
    )
    parser.add_argument(
        "--seqlet-low-similarity-only",
        action="store_true",
        help=(
            "only apply the hit_seqlet_confidence corroboration floor to "
            "motifs already failing cwm_similarity QC, read from a prior "
            "report_bpnet.py run's report/motif_report.tsv -- only used "
            f"when --score-column {SEQLET_CONFIDENCE_COLUMN}. Motifs above "
            "the threshold are left completely untouched, not just skipped "
            "for having too few hits. Avoids the blanket ~50% drop across "
            "motifs that never had a cwm_similarity problem to begin with "
            "(see module docstring); requires report_bpnet.py to have "
            "already been run once against hits *before* this filter (e.g. "
            "with hits_confidence_filtered.tsv temporarily moved aside), "
            "since running it after would reflect the filtered hits instead "
            "of the original QC failures"
        ),
    )
    parser.add_argument(
        "--seqlet-similarity-threshold",
        type=float,
        default=0.9,
        help=(
            "cwm_similarity threshold defining 'already failing' when "
            "--seqlet-low-similarity-only is set (default: 0.9, matching "
            "report_bpnet.py's own --cwm-similarity-threshold default -- "
            "try a lower value like 0.85 to spare borderline motifs from "
            "this filter and only touch clearly-struggling ones)"
        ),
    )
    parser.add_argument(
        "--seqlet-clip-percentile",
        type=float,
        default=DEFAULT_SEQLET_CLIP_PERCENTILE,
        help=(
            "clip |contribs| to this percentile before calling "
            "recursive_seqlets -- only used when --score-column "
            f"{SEQLET_CONFIDENCE_COLUMN} (default: "
            f"{DEFAULT_SEQLET_CLIP_PERCENTILE}). Necessary because "
            "recursive_seqlets bins its *entire* input in one global range, "
            "so a single extreme outlier anywhere in the genome-wide array "
            "would otherwise collapse calling everywhere else; see "
            "compute_seqlet_confidence's docstring"
        ),
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help=(
            "bin --score-column in log10-space instead of linear -- use for "
            "heavy right-skewed, effectively-unbounded columns like "
            "hit_importance (bounded columns like hit_correlation/"
            "hit_similarity don't need this)"
        ),
    )
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument(
        "--min-rise-frac",
        type=float,
        default=DEFAULT_MIN_RISE_FRAC,
        help=(
            "required ratio of secondary-mode bin count to trough bin count "
            f"to treat the rise as real, not histogram noise (default: {DEFAULT_MIN_RISE_FRAC})"
        ),
    )
    parser.add_argument(
        "--min-bin-frac",
        type=float,
        default=DEFAULT_MIN_BIN_FRAC,
        help=(
            "minimum secondary-mode bin count, as a fraction of the motif's "
            f"total hit count, to trust it (default: {DEFAULT_MIN_BIN_FRAC}). "
            "This is a *single bin's* height, so a secondary mode spread "
            "across more bins (e.g. --log-scale on a wide/diffuse tail) "
            "dilutes below this much faster than a narrow one with the same "
            "total mass -- lower this (e.g. 0.001) if --log-scale finds "
            "nothing despite a visibly obvious secondary mode"
        ),
    )
    parser.add_argument(
        "--min-total-hits",
        type=int,
        default=DEFAULT_MIN_TOTAL_HITS,
        help=(
            "skip bimodality detection entirely for motifs with fewer than "
            f"this many hits (default: {DEFAULT_MIN_TOTAL_HITS}, too few for a "
            "reliable histogram)"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    model_dir_name = Path(args.model_dir).name if args.model_dir else args.experiment

    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    trim_coords = (
        modisco_dir
        / f"{args.experiment}_{args.head}_trim_coords_min{args.min_trim_len}bp.tsv"
        if args.min_trim_len is not None
        else None
    )
    suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, trim_coords)
    exp_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{args.head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir

    hits_path = resolve_hits_path(
        hits_dir,
        stages=["hits_dedensified.tsv", "hits_unique.tsv"],
        verbose=args.verbose,
    )
    if hits_path is None:
        print(f"Error: no hits found in {hits_dir}", file=sys.stderr)
        print("Run call_hits_bpnet.py first.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Reading hits from {hits_path}")

    hits = pd.read_csv(hits_path, sep="\t")
    original_columns = list(hits.columns)

    if args.score_column == FLANK_SIMILARITY_COLUMN:
        regions_npz = exp_dir / "regions.npz"
        modisco_h5 = (
            Path(args.modisco_h5) if args.modisco_h5
            else modisco_dir / f"{args.experiment}_{args.head}.modisco.h5"
        )
        for path, label, hint in [
            (regions_npz, "regions.npz", "Run call_hits_bpnet.py first."),
            (modisco_h5, "motif CWMs (.modisco.h5)", "Run MoDISco first."),
        ]:
            if not path.exists():
                print(f"Error: {label} not found: {path}", file=sys.stderr)
                print(hint, file=sys.stderr)
                sys.exit(1)

        if args.verbose:
            print(f"Computing {FLANK_SIMILARITY_COLUMN} from {regions_npz} and {modisco_h5}")

        sequences, contribs, peaks_df, _ = load_regions_npz(str(regions_npz))
        contribs = project_contribs(contribs, sequences)
        peak_row_index = build_peak_row_index(peaks_df)
        peak_region_starts = peaks_df["peak_region_start"].to_numpy()
        cwm_lookup, motif_width = build_cwm_lookup(modisco_h5)

        hits[FLANK_SIMILARITY_COLUMN] = compute_flank_similarity(
            hits, contribs, sequences, peak_row_index, peak_region_starts, cwm_lookup, motif_width
        )
        n_unscoreable = int(hits[FLANK_SIMILARITY_COLUMN].isna().sum())
        if args.verbose or n_unscoreable:
            print(
                f"{n_unscoreable}/{len(hits)} hits ({n_unscoreable / max(len(hits), 1):.1%}) "
                "could not be scored (untrimmed span outside the saved contribution "
                "track, e.g. near a peak edge) -- excluded from bimodality detection "
                "and never dropped by this score"
            )
    elif args.score_column == SUMMIT_PROXIMITY_COLUMN:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        if args.experiment not in config["experiments"]:
            print(f"Error: {args.experiment} not found in config", file=sys.stderr)
            sys.exit(1)
        filtered_peaks_path = REPO_ROOT / config["experiments"][args.experiment]["processed"]["filtered_peaks"]
        if not filtered_peaks_path.exists():
            print(f"Error: filtered_peaks not found: {filtered_peaks_path}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"Computing {SUMMIT_PROXIMITY_COLUMN} from {filtered_peaks_path}")

        filtered_peaks_df = load_filtered_peaks(filtered_peaks_path)
        has_any_summit = (
            filtered_peaks_df["summits_pos_list"].apply(len)
            + filtered_peaks_df["summits_neg_list"].apply(len)
        ) > 0
        mode = infer_coordinate_mode(filtered_peaks_df[has_any_summit], verbose=args.verbose)
        summit_lookup = build_summit_lookup(filtered_peaks_df[has_any_summit], mode)

        signed, matched = compute_distances(hits, summit_lookup)
        hits[SUMMIT_PROXIMITY_COLUMN] = np.where(matched, -np.abs(signed), np.nan)
        n_unmatched = int((~matched).sum())
        if args.verbose or n_unmatched:
            print(
                f"{n_unmatched}/{len(hits)} hits ({n_unmatched / max(len(hits), 1):.1%}) "
                "could not be matched to any peak with a real summit -- "
                "excluded from bimodality detection and never dropped by this score"
            )
    elif args.score_column == SEQ_COMPLEXITY_COLUMN:
        regions_npz = exp_dir / "regions.npz"
        if not regions_npz.exists():
            print(f"Error: regions.npz not found: {regions_npz}", file=sys.stderr)
            print("Run call_hits_bpnet.py first.", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"Computing {SEQ_COMPLEXITY_COLUMN} from {regions_npz}")

        sequences, _, peaks_df, _ = load_regions_npz(str(regions_npz))
        peak_row_index = build_peak_row_index(peaks_df)
        peak_region_starts = peaks_df["peak_region_start"].to_numpy()

        hits[SEQ_COMPLEXITY_COLUMN] = compute_seq_complexity(
            hits, sequences, peak_row_index, peak_region_starts, window=args.complexity_window
        )
        n_unscoreable = int(hits[SEQ_COMPLEXITY_COLUMN].isna().sum())
        if args.verbose or n_unscoreable:
            print(
                f"{n_unscoreable}/{len(hits)} hits ({n_unscoreable / max(len(hits), 1):.1%}) "
                f"could not be scored (--complexity-window {args.complexity_window}bp "
                "extends outside the saved region track, e.g. near a peak edge) -- "
                "excluded from bimodality detection and never dropped by this score"
            )
    elif args.score_column == SEQLET_CONFIDENCE_COLUMN:
        regions_npz = exp_dir / "regions.npz"
        if not regions_npz.exists():
            print(f"Error: regions.npz not found: {regions_npz}", file=sys.stderr)
            print("Run call_hits_bpnet.py first.", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(
                f"Computing {SEQLET_CONFIDENCE_COLUMN} from {regions_npz} "
                "(running tangermeme.seqlet.recursive_seqlets over every region -- "
                "more compute-intensive than the other score columns)"
            )

        sequences, contribs, peaks_df, _ = load_regions_npz(str(regions_npz))
        contribs = project_contribs(contribs, sequences)
        peak_row_index = build_peak_row_index(peaks_df)
        peak_region_starts = peaks_df["peak_region_start"].to_numpy()

        hits[SEQLET_CONFIDENCE_COLUMN] = compute_seqlet_confidence(
            hits,
            contribs,
            peak_row_index,
            peak_region_starts,
            threshold=args.seqlet_threshold,
            min_seqlet_len=args.min_seqlet_len,
            max_seqlet_len=args.max_seqlet_len,
            additional_flanks=args.seqlet_additional_flanks,
            clip_percentile=args.seqlet_clip_percentile,
        )
        n_uncorroborated = int((hits[SEQLET_CONFIDENCE_COLUMN] == 0.0).sum())
        if args.verbose or n_uncorroborated:
            print(
                f"{n_uncorroborated}/{len(hits)} hits ({n_uncorroborated / max(len(hits), 1):.1%}) "
                "have no overlapping recursive_seqlets call at all (scored 0.0 -- "
                "a real negative signal, not missing data, so these ARE included "
                "in and can be dropped by bimodality detection)"
            )
    elif args.score_column not in hits.columns:
        print(
            f"Error: --score-column {args.score_column!r} not found in "
            f"{hits_path} (columns: {list(hits.columns)})",
            file=sys.stderr,
        )
        sys.exit(1)

    keep_mask = np.ones(len(hits), dtype=bool)
    filtered_motifs = []
    unfiltered_motifs = []

    if args.score_column == SEQLET_CONFIDENCE_COLUMN:
        # Binary corroboration floor, not bimodality detection: real per-hit
        # data (K562 ENCSR220XSM TATA/TA-Inr, split by ground-truth signed
        # distance to the real TSS summit) showed recursive_seqlets
        # corroboration RATE differs ~2-3x between near-summit "core" hits
        # (~58-59%) and downstream-hump/far-background hits (~20-29%), while
        # the score's magnitude when corroborated (mean ~1.34-1.41) is
        # nearly identical across all three -- all the information is in
        # whether a call exists at all, not in how strong it is.
        # detect_low_confidence_cutoff's find_peaks-based trough-then-rise
        # search is the wrong tool for this shape (a spike at exactly 0.0
        # followed by a monotonically *decaying* tail, not a dip-then-
        # second-peak), so it reports "no secondary mode" even where real,
        # useful structure exists -- confirmed by hand on real data. This
        # is not a clean separator (~41% of real core hits would still be
        # dropped as false negatives; ~20-29% of spurious hits still survive
        # as corroborated by chance), and CLIPNET's own published methods
        # (--seqlet-threshold's default docstring) note recursive_seqlets
        # "struggled to consistently identify relatively low importance
        # profile motifs" independent of any contamination problem, so this
        # costs real recall on motifs like DPR/activator elements too.
        # Accepted deliberately, per explicit product decision: this needs
        # to scale identity-agnostically across the whole atlas, and
        # consistent recall on core promoter motifs (TATA/Inr) matters more
        # than recall on low-importance profile motifs here. --log-scale/
        # --n-bins/--smoothing-window/--min-rise-frac/--min-bin-frac are
        # ignored for this column since no bimodality search runs.
        #
        # --seqlet-low-similarity-only narrows that scaling tradeoff: instead
        # of applying the floor to every motif, restrict it to only those
        # already failing cwm_similarity QC (read from a prior
        # report_bpnet.py run's report/motif_report.tsv), leaving motifs
        # that were already fine completely untouched. Found necessary by
        # hand: the unrestricted floor dropped hits from 28/45 motifs
        # (0%-89% each), including motifs with no known contamination
        # problem, and pushed at least one borderline-passing motif
        # (cwm_similarity 0.895) under the 0.9 cutoff as a side effect.
        restricted_motifs = None
        if args.seqlet_low_similarity_only:
            motif_report_path = hits_dir / "report" / "motif_report.tsv"
            if not motif_report_path.exists():
                print(
                    f"Error: --seqlet-low-similarity-only needs {motif_report_path} "
                    "from a prior report_bpnet.py run (against hits from *before* "
                    "this filter, e.g. with hits_confidence_filtered.tsv temporarily "
                    "moved aside)",
                    file=sys.stderr,
                )
                sys.exit(1)
            motif_report = pd.read_csv(motif_report_path, sep="\t")
            restricted_motifs = set(
                motif_report.loc[
                    motif_report["cwm_similarity"] <= args.seqlet_similarity_threshold,
                    "motif_name",
                ]
            )
            if args.verbose:
                print(
                    f"Restricting {SEQLET_CONFIDENCE_COLUMN} to "
                    f"{len(restricted_motifs)} motif(s) with cwm_similarity <= "
                    f"{args.seqlet_similarity_threshold} (from {motif_report_path}): "
                    f"{sorted(restricted_motifs)}"
                )

        out_of_scope_motifs = []
        for motif_name, group in hits.groupby("motif_name", sort=False):
            if restricted_motifs is not None and motif_name not in restricted_motifs:
                out_of_scope_motifs.append(motif_name)
                continue
            if len(group) < args.min_total_hits:
                unfiltered_motifs.append(motif_name)
                continue
            below_floor = group.index[group[SEQLET_CONFIDENCE_COLUMN] <= 0]
            keep_mask[below_floor] = False
            filtered_motifs.append(
                dict(motif_name=motif_name, n_total=len(group), n_dropped=len(below_floor))
            )

        if args.verbose and out_of_scope_motifs:
            print(
                f"\n{len(out_of_scope_motifs)} motif(s) left completely untouched "
                f"(cwm_similarity above {args.seqlet_similarity_threshold}, out of "
                f"scope for --seqlet-low-similarity-only): {out_of_scope_motifs}"
            )
    else:
        for motif_name, group in hits.groupby("motif_name", sort=False):
            scores = group[args.score_column].to_numpy()
            scores = scores[~np.isnan(scores)]
            result = detect_low_confidence_cutoff(
                scores,
                n_bins=args.n_bins,
                smoothing_window=args.smoothing_window,
                min_rise_frac=args.min_rise_frac,
                min_bin_frac=args.min_bin_frac,
                min_total_hits=args.min_total_hits,
                log_scale=args.log_scale,
            )
            if result is None:
                unfiltered_motifs.append(motif_name)
                continue

            below_cutoff = group.index[group[args.score_column] < result["cutoff"]]
            keep_mask[below_cutoff] = False
            filtered_motifs.append(
                dict(
                    motif_name=motif_name,
                    n_total=len(group),
                    n_dropped=len(below_cutoff),
                    **result,
                )
            )

    # Drop any synthetic column this run added (hit_flank_similarity/
    # hit_summit_proximity/hit_seq_complexity/hit_seqlet_confidence aren't
    # part of Fi-NeMo's fixed hits.tsv schema) before writing output --
    # finemo's own downstream `report` subcommand hard-codes an expected
    # column count and errors on a mismatch (polars.exceptions.SchemaError:
    # "provided schema does not match number of columns in file"), found by
    # hand running report_bpnet.py against a hit_seqlet_confidence-filtered
    # hits_confidence_filtered.tsv.
    kept = hits[keep_mask][original_columns]

    if args.score_column == SEQLET_CONFIDENCE_COLUMN:
        if filtered_motifs:
            print(
                f"Applied a hard {SEQLET_CONFIDENCE_COLUMN} > 0 floor (binary "
                f"corroboration, no bimodality detection -- see module docstring) "
                f"to {len(filtered_motifs)} motif(s):"
            )
            for r in sorted(filtered_motifs, key=lambda r: -r["n_dropped"]):
                frac = r["n_dropped"] / r["n_total"] if r["n_total"] else 0.0
                print(
                    f"  {r['motif_name']}: dropped {r['n_dropped']}/{r['n_total']} "
                    f"hits ({frac:.1%})"
                )
        else:
            print("No motif had enough hits to apply the corroboration floor; nothing dropped.")

        if args.verbose and unfiltered_motifs:
            print(
                f"\n{len(unfiltered_motifs)} motif(s) left untouched (fewer than "
                f"{args.min_total_hits} hits): {unfiltered_motifs}"
            )
    else:
        if filtered_motifs:
            print(
                f"Detected a low-confidence mode (by {args.score_column}) in "
                f"{len(filtered_motifs)} motif(s):"
            )
            for r in sorted(filtered_motifs, key=lambda r: -r["n_dropped"]):
                print(
                    f"  {r['motif_name']}: cutoff={r['cutoff']:.3f} "
                    f"(trough={r['trough']:.3f}/{r['trough_count']:.0f} hits, "
                    f"secondary_mode={r['secondary_mode']:.3f}/{r['secondary_count']:.0f} hits) "
                    f"-- dropped {r['n_dropped']}/{r['n_total']} hits "
                    f"({r['n_dropped'] / r['n_total']:.1%})"
                )
        else:
            print(f"No motif showed a detectable low-confidence mode by {args.score_column}.")

        if args.verbose and unfiltered_motifs:
            print(
                f"\n{len(unfiltered_motifs)} motif(s) left untouched (no secondary "
                f"mode detected): {unfiltered_motifs}"
            )

    out_path = hits_dir / "hits_confidence_filtered.tsv"
    kept.to_csv(out_path, sep="\t", index=False)
    print(
        f"\nKept {len(kept)}/{len(hits)} hits "
        f"({len(hits) - len(kept)} dropped from {len(filtered_motifs)} motifs)"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
