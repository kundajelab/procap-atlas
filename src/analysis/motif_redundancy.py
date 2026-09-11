#!/usr/bin/env python3
"""How much of the MotifCompendium lexicon is the same motif counted twice?

Every count derived from the compendium -- lexicon size, rarefaction curves,
the number of tissue-restricted clusters -- is inflated if one real motif got
split across several clusters because `cluster_motifs.py --across-threshold`
failed to merge them. Unlike the other caveats in this directory, redundancy
biases in the *flattering* direction, so it is the one a referee is most likely
to find first.

JASPAR name collisions hint at the scale but cannot measure it: on the real
count-head compendium 306 clusters carry only 112 distinct JASPAR names (SP9
claimed by 31 clusters, TBP by 15), yet JASPAR annotation is a nearest-neighbour
lookup, so a bare GC-box and a GC-box with an ETS half-site can both best-match
SP9 while being genuinely different motifs. This script measures redundancy
directly, by comparing the cluster-average CWMs against each other.

Uses TOMTOM from `memelite` (the "tomtom-lite" reimplementation), which is a
Python API with no command-line entry point -- hence a script rather than a
shell command. Self-comparison of the compendium's own MEME export, diagonal
removed.

Redundancy is reported as *excess clusters*: the number of clusters that would
disappear if every near-duplicate group collapsed to one motif. Two things
about the real count-head compendium make how that is computed matter a lot.

**Motifs must be trimmed first.** MotifCompendium exports fixed-width CWM
windows -- all 944 count-head clusters are exactly 50bp -- while the
informative core is typically 6-15bp. Comparing untrimmed windows means TOMTOM
largely aligns low-information flanks, which resemble background and therefore
resemble each other, and `--min-overlap-frac` goes inert (35bp of a 50-vs-50
comparison is satisfied at almost any offset). Untrimmed, the sweep reported
77.6% excess at p <= 1e-6, which is flank similarity, not redundancy. Motifs
are therefore trimmed by information content before comparison, mirroring the
shape of Fi-NeMo's own CWM trim rule (outermost positions clearing
`threshold * max`, with a minimum-length floor) -- `--no-trim` reproduces the
old behaviour for comparison.

**Single linkage chains.** Connected components merge A~B~C even when A and C
are unrelated, and on this data one hub motif absorbs hundreds: the untrimmed
run put 142 clusters in one component at p <= 1e-12 and 919 of 944 at p <= 1e-2.
The sweep cannot fix this, so three merge criteria are reported side by side:

  mutual   only mutual best hits merge (A's best match is B and B's is A).
           Cannot chain at all; a conservative lower bound.
  complete complete-linkage clustering -- a group merges only if *every* pair
           within it passes. The best-behaved middle estimate.
  single   connected components, the chaining-prone upper bound, kept so the
           gap between it and `complete` shows how much chaining is happening.

A large single-vs-complete gap means the threshold is too loose for this data,
not that redundancy is high.

No single p-value threshold is defensible -- significance depends on motif
length and information content, and family members are genuinely similar
without being duplicates -- so the output is a sweep across thresholds, the
same treatment the abundance floor gets in plot_motif_rarefaction.py. Read the
shape: if excess is flat across several orders of magnitude, redundancy is well
determined; if it climbs steadily, the lexicon size is threshold-dependent and
should be quoted as a range.

One footgun worth knowing: TOMTOM scores columns against a background
estimated from the target set, so near-deterministic PWMs (a one-hot consensus
with epsilon elsewhere) make that background degenerate and can return p = 1.0
for two *identical* motifs. The compendium's cluster-average CWMs are soft
enough that this does not arise in practice, but it does mean synthetic
one-hot fixtures cannot be used to test the p-value path.

`--min-overlap-frac` additionally requires the best alignment to cover that
fraction of the shorter motif, which suppresses the case where a short motif
aligns significantly inside a longer unrelated one.

Outputs (in --out-dir):
  motif_redundancy_{head}_pairs.tsv       every passing pair at the loosest threshold
  motif_redundancy_{head}_summary.tsv     excess clusters per threshold
  motif_redundancy_{head}_components.tsv  cluster -> merged component, at --report-threshold
  motif_redundancy_{head}.{png,pdf}       excess vs threshold

Usage:
    python src/analysis/motif_redundancy.py --head count
    python src/analysis/motif_redundancy.py --head count --report-threshold 1e-6
    python src/analysis/motif_redundancy.py --head count --min-overlap-frac 0.8
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MC_DIR = REPO_ROOT / "motifcompendium" / "bpnet"
# Fi-NeMo's median trimmed motif width on real per-experiment CWMs at
# --cwm-trim-threshold 0.3, measured over 2.24M hits / 65 motifs of a real
# profile-head run: 14bp across motifs, 6bp weighted by hits, out of a
# 50bp window. The calibration target for trimming cluster averages.
FINEMO_MEDIAN_TRIM_BP = 14

DEFAULT_THRESHOLDS = (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2)
NAME_RE = re.compile(r"((?:pos|neg)_patterns\.\d+)")


def parse_motif_name(key: str, name_regex: re.Pattern = NAME_RE) -> str:
    """Recover a cluster identifier from a MEME metadata line.

    `memelite.io.read_meme` keys its dict on the whole metadata line, whose
    exact shape depends on how MotifCompendium's exporter wrote it, so the
    compendium-style identifier is extracted when present and the raw key kept
    otherwise -- never silently dropped.
    """
    match = name_regex.search(key)
    if match:
        return match.group(1)
    return key.replace("MOTIF", "").strip().split()[0] if key.strip() else key


def load_subset(path: Path) -> set[str]:
    """Motif names to restrict to: a bare list, or a TSV with a name column."""
    text = path.read_text().splitlines()
    if not text:
        return set()
    header = text[0].split("\t")
    for col in ("motif", "compendium_motif_name", "motif_name"):
        if col in header:
            idx = header.index(col)
            return {
                line.split("\t")[idx].strip()
                for line in text[1:] if line.strip()
            }
    return {line.split("\t")[0].strip() for line in text if line.strip()}


def load_motifs(meme_path: Path, name_regex: re.Pattern) -> tuple[list[str], list[np.ndarray]]:
    from memelite.io import read_meme

    motifs = read_meme(str(meme_path))
    names, pwms = [], []
    for key, pwm in motifs.items():
        names.append(parse_motif_name(key, name_regex))
        arr = np.asarray(pwm, dtype=np.float64)
        # memelite expects (alphabet, length); MEME files are (length, alphabet)
        # in the file itself, but read_meme already returns the torch layout.
        # Guard anyway: the alphabet axis is the short one for real motifs.
        if arr.ndim != 2:
            raise ValueError(f"motif {key!r} is not 2-D: shape {arr.shape}")
        if arr.shape[0] != 4 and arr.shape[1] == 4:
            arr = arr.T
        pwms.append(arr)
    return names, pwms


CWM_KEYS = (
    "contrib_scores", "contrib", "contribution_scores", "CWM", "cwm",
    "hypothetical_contribs",
)
PFM_KEYS = ("sequence", "PFM", "pfm", "ppm", "PPM", "probs")


def _as_alphabet_first(arr) -> np.ndarray | None:
    """Return (4, length) if `arr` looks like a motif matrix, else None."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        return None
    if a.shape[0] == 4 and a.shape[1] != 4:
        return a
    if a.shape[1] == 4 and a.shape[0] != 4:
        return a.T
    if a.shape == (4, 4):
        return a
    return None


def h5_tree(h5_path: Path, limit: int = 40) -> list[str]:
    """Flat listing of datasets in an h5, for diagnosing an unexpected layout."""
    import h5py

    found: list[str] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and len(found) < limit:
            found.append(f"{name}  shape={obj.shape} dtype={obj.dtype}")

    with h5py.File(h5_path, "r") as f:
        f.visititems(visit)
    return found


def _motif_name_from_path(path: str) -> str:
    """Turn an h5 group path into the `{posneg}_patterns.{cluster}` convention.

    Matches what link_hits_to_compendium.py writes into
    `compendium_motif_name`, so subsets and metadata joins line up. Falls back
    to a dotted version of the path so a motif is never silently dropped.
    """
    parts = [p for p in path.split("/") if p]
    for i, part in enumerate(parts):
        if part in ("pos_patterns", "neg_patterns"):
            rest = parts[i + 1:]
            if rest:
                suffix = rest[-1]
                if suffix.startswith("pattern_"):
                    suffix = suffix[len("pattern_"):]
                return f"{part}.{suffix}"
            return part
    return ".".join(parts)


def load_motifs_h5(h5_path: Path):
    """Load (names, PFMs, CWMs) from a cluster-average h5.

    Preferred over the MEME export because it retains contribution scores. A
    MEME file has only probabilities, and information content cannot separate a
    motif's core from its flanks on this data: PRO-cap peaks are GC-rich, so
    flanking positions carry real compositional information which clears any
    threshold relative to a soft cluster-average's modest maximum. Trimming has
    to use contribution magnitude, as Fi-NeMo's own `trim_motif` does.

    The layout is discovered, not assumed. MotifCompendium's exporter does not
    necessarily write the `pos_patterns/pattern_N/` hierarchy tfmodisco-lite
    does -- assuming it found zero motifs on the real file -- so this walks the
    whole h5 and treats any group holding a motif-shaped ((4, L) or (L, 4))
    dataset as a motif, matching dataset names against CWM_KEYS and PFM_KEYS.

    A group with contributions but no probability matrix falls back to
    row-normalized |contributions| as the comparison matrix: approximate, since
    TOMTOM needs probability-like columns, but the trim span is unaffected.
    """
    import h5py

    groups: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as f:

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            leaf = name.split("/")[-1]
            if leaf not in CWM_KEYS and leaf not in PFM_KEYS:
                return
            mat = _as_alphabet_first(obj[:])
            if mat is None:
                return
            parent = "/".join(name.split("/")[:-1]) or name
            groups.setdefault(parent, {})[leaf] = mat

        f.visititems(visit)

    names, pfms, cwms = [], [], []
    for path in sorted(groups):
        entry = groups[path]
        cwm = next((entry[k] for k in CWM_KEYS if k in entry), None)
        pfm = next((entry[k] for k in PFM_KEYS if k in entry), None)
        if cwm is None:
            continue
        if pfm is None:
            mag = np.abs(cwm)
            total = mag.sum(axis=0, keepdims=True)
            pfm = np.divide(mag, total, out=np.full_like(mag, 0.25), where=total > 0)
        names.append(_motif_name_from_path(path))
        pfms.append(pfm)
        cwms.append(cwm)
    return names, pfms, cwms


def trim_cwm(cwm: np.ndarray, threshold: float = 0.3, min_len: int = 6) -> tuple[int, int]:
    """Fi-NeMo's trim rule on a CWM: per-position summed |contribution|.

    Keeps the outermost positions clearing `threshold * max`, then widens
    symmetrically to `min_len` -- the same shape as
    `finemo.data_io.trim_motif` plus Kelly Cochran's ProCapNet min-length
    floor, replicated here so this script does not need the Linux-only
    `finemo` package.
    """
    mag = np.abs(np.asarray(cwm, dtype=np.float64)).sum(axis=0)
    width = len(mag)
    if width == 0 or mag.max() <= 0:
        return 0, width
    keep = np.flatnonzero(mag >= threshold * mag.max())
    start, end = int(keep[0]), int(keep[-1]) + 1
    while end - start < min_len and (start > 0 or end < width):
        if start > 0:
            start -= 1
        if end - start < min_len and end < width:
            end += 1
    return start, end


def trim_by_cwm(
    pfms: list[np.ndarray],
    cwms: list[np.ndarray],
    threshold: float,
    min_len: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Trim spans from the CWMs, apply them to the PFMs.

    TOMTOM needs probability-like columns, so the comparison runs on the PFM
    while the *span* comes from the contribution scores -- which is the only
    signal that actually marks where the motif is.
    """
    widths = np.array([m.shape[-1] for m in pfms])
    out = []
    for pfm, cwm in zip(pfms, cwms):
        start, end = trim_cwm(cwm, threshold, min_len)
        out.append(np.ascontiguousarray(pfm[:, start:end]))
    return out, widths


def trimming_ineffective(
    widths: np.ndarray, raw_widths: np.ndarray, frac: float = 0.8
) -> bool:
    """Did trimming fail to actually shrink these motifs?

    The signature of information-content trimming on cluster-average PFMs: the
    real count head went from 50bp to a median of 49bp, leaving the comparison
    flank-dominated. Extracted as a predicate so it is testable without having
    to assert on a subprocess's stderr.
    """
    if len(widths) == 0 or len(raw_widths) == 0:
        return False
    return bool(np.median(widths) > frac * np.median(raw_widths))


def information_content(pwm: np.ndarray) -> np.ndarray:
    """Per-position information content in bits, shape (length,).

    `pwm` is (alphabet, length) probabilities. IC = log2(A) + sum p log2 p,
    which is 0 for a uniform column and log2(4) = 2 for a fully determined one.
    """
    p = np.clip(np.asarray(pwm, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum(axis=0, keepdims=True)
    return np.log2(p.shape[0]) + (p * np.log2(p)).sum(axis=0)


def trim_pwm(
    pwm: np.ndarray, threshold: float = 0.3, min_len: int = 6
) -> tuple[int, int]:
    """Trim to the informative core; returns (start, end) as a half-open span.

    Mirrors the *shape* of Fi-NeMo's `trim_motif` rule -- keep the outermost
    positions whose per-position magnitude clears `threshold * max` -- but on
    information content rather than summed |contribution|, since a MEME export
    carries probabilities, not contributions. It is an analog, not the same
    function; use the cluster-average h5 with `finemo.data_io.trim_motif` if an
    exact match to hit-calling trimming is needed.

    Widened symmetrically to `min_len` (clamped to the motif width) for the same
    reason Kelly Cochran's ProCapNet run added a floor: short core-promoter
    elements can otherwise trim to 1-2bp, which no comparison can use.
    """
    ic = information_content(pwm)
    width = len(ic)
    if width == 0 or not np.isfinite(ic).any() or ic.max() <= 0:
        return 0, width
    keep = np.flatnonzero(ic >= threshold * ic.max())
    if keep.size == 0:
        return 0, width
    start, end = int(keep[0]), int(keep[-1]) + 1
    while end - start < min_len and (start > 0 or end < width):
        if start > 0:
            start -= 1
        if end - start < min_len and end < width:
            end += 1
    return start, end


def trim_motifs(
    pwms: list[np.ndarray], threshold: float, min_len: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """Trim every motif; returns (trimmed, original widths)."""
    widths = np.array([m.shape[-1] for m in pwms])
    out = []
    for m in pwms:
        start, end = trim_pwm(m, threshold, min_len)
        out.append(np.ascontiguousarray(m[:, start:end]))
    return out, widths


def degenerate_pvalues(p: np.ndarray) -> tuple[bool, float, int]:
    """Detect a collapsed TOMTOM p-value matrix.

    memelite estimates a background from the target set's columns, and for some
    input configurations that estimate degenerates and every off-diagonal
    p-value comes back as exactly 0.0 or 1.0 -- including p = 1.0 for two
    *identical* motifs. It is not a simple matter of having too few motifs:
    41 motifs of 14bp behave correctly while 81 of 10bp and 81 of 20bp both
    collapse. Nothing in the output reveals this unless it is checked, and
    every downstream number would be meaningless, so it is checked.

    Returns (is_degenerate, fraction_at_0_or_1, n_distinct_values).
    """
    off = p[np.isfinite(p)]
    if off.size == 0:
        return True, 1.0, 0
    extreme = float(np.mean((off == 0.0) | (off == 1.0)))
    distinct = int(np.unique(off).size)
    return (extreme > 0.9 or distinct < 10), extreme, distinct


def self_compare(pwms: list[np.ndarray], n_jobs: int) -> dict[str, np.ndarray]:
    """Run TOMTOM of every cluster against every other, diagonal masked out."""
    from memelite import tomtom

    p, scores, offsets, overlaps, strands = tomtom(pwms, pwms, n_jobs=n_jobs)
    p = np.asarray(p, dtype=np.float64).copy()
    np.fill_diagonal(p, np.inf)  # a motif is not its own duplicate

    bad, extreme, distinct = degenerate_pvalues(p)
    if bad:
        print(
            "WARNING: TOMTOM p-values look degenerate "
            f"({extreme:.0%} of off-diagonal values are exactly 0 or 1; "
            f"{distinct} distinct values across {len(pwms)} motifs). memelite "
            "estimates its background from the target columns and that "
            "estimate can collapse for particular input configurations -- when "
            "it does, identical motifs can score p = 1.0. Every redundancy "
            "figure below would be meaningless. Try a different "
            "--trim-threshold, or drop --subset so more motifs contribute to "
            "the background.",
            file=sys.stderr,
        )
    return {
        "p": p,
        "scores": np.asarray(scores),
        "offsets": np.asarray(offsets),
        "overlaps": np.asarray(overlaps),
        "strands": np.asarray(strands),
    }


def build_pairs(
    names: list[str],
    pwms: list[np.ndarray],
    res: dict[str, np.ndarray],
    p_threshold: float,
    min_overlap_frac: float,
) -> pd.DataFrame:
    """Upper-triangle pairs passing the p-value and overlap filters."""
    lengths = np.array([m.shape[-1] for m in pwms])
    p = res["p"]
    iu = np.triu_indices(len(names), k=1)
    # Symmetrize on the more conservative (larger) p-value: TOMTOM is not
    # symmetric, since the query's background distribution sets the scale.
    p_sym = np.maximum(p[iu], p.T[iu])
    overlap = res["overlaps"][iu]
    shorter = np.minimum(lengths[iu[0]], lengths[iu[1]])
    frac = np.divide(overlap, shorter, out=np.zeros_like(overlap, dtype=float),
                     where=shorter > 0)
    keep = (p_sym <= p_threshold) & (frac >= min_overlap_frac)
    return pd.DataFrame(
        {
            "motif_a": [names[i] for i in iu[0][keep]],
            "motif_b": [names[j] for j in iu[1][keep]],
            "p_value": p_sym[keep],
            "overlap": overlap[keep],
            "overlap_frac": np.round(frac[keep], 3),
            "len_a": lengths[iu[0][keep]],
            "len_b": lengths[iu[1][keep]],
            "strand": res["strands"][iu][keep],
        }
    ).sort_values("p_value")


def symmetric_p(
    pwms: list[np.ndarray], res: dict[str, np.ndarray], min_overlap_frac: float
) -> np.ndarray:
    """Full symmetrized p-value matrix, with failing-overlap pairs set to 1.

    TOMTOM is asymmetric (the query sets the background scale), so each pair is
    symmetrized on the larger -- more conservative -- of the two p-values.
    """
    lengths = np.array([m.shape[-1] for m in pwms])
    p = np.maximum(res["p"], res["p"].T)
    shorter = np.minimum.outer(lengths, lengths)
    frac = np.divide(
        res["overlaps"], shorter,
        out=np.zeros_like(res["overlaps"], dtype=float), where=shorter > 0,
    )
    p = np.where(np.maximum(frac, frac.T) >= min_overlap_frac, p, 1.0)
    np.fill_diagonal(p, 0.0)
    return p


def merge_mutual_best(names: list[str], p_sym: np.ndarray, threshold: float) -> dict[str, int]:
    """Merge only mutual best hits. Cannot chain, so a lower bound."""
    masked = p_sym.copy()
    np.fill_diagonal(masked, np.inf)
    best = masked.argmin(axis=1)
    rows, cols = [], []
    for i, j in enumerate(best):
        if best[j] == i and i < j and masked[i, j] <= threshold:
            rows.append(names[i])
            cols.append(names[j])
    return connected_components(names, pd.DataFrame({"motif_a": rows, "motif_b": cols}))


def merge_complete(names: list[str], p_sym: np.ndarray, threshold: float) -> dict[str, int]:
    """Complete-linkage clustering: a group merges only if every pair passes.

    This is what stops the chaining that makes connected components unusable
    here -- no cluster joins a group unless it is within threshold of every
    member already in it.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if len(names) < 2:
        return {n: 0 for n in names}
    d = squareform(np.clip((p_sym + p_sym.T) / 2, 0, 1), checks=False)
    labels = fcluster(linkage(d, method="complete"), t=threshold, criterion="distance")
    remap: dict[int, int] = {}
    return {n: remap.setdefault(int(l), len(remap)) for n, l in zip(names, labels)}


def connected_components(names: list[str], pairs: pd.DataFrame) -> dict[str, int]:
    """Union-find over passing pairs; component id per motif name."""
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["motif_a"], pairs["motif_b"]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    roots = {}
    out = {}
    for n in names:
        r = find(n)
        out[n] = roots.setdefault(r, len(roots))
    return out


MERGERS = ("mutual", "complete", "single")


def merge(
    criterion: str,
    names: list[str],
    pwms: list[np.ndarray],
    res: dict[str, np.ndarray],
    p_sym: np.ndarray,
    threshold: float,
    min_overlap_frac: float,
) -> dict[str, int]:
    if criterion == "mutual":
        return merge_mutual_best(names, p_sym, threshold)
    if criterion == "complete":
        return merge_complete(names, p_sym, threshold)
    if criterion == "single":
        return connected_components(
            names, build_pairs(names, pwms, res, threshold, min_overlap_frac)
        )
    raise ValueError(f"unknown merge criterion {criterion!r}")


def sweep(
    names: list[str],
    pwms: list[np.ndarray],
    res: dict[str, np.ndarray],
    thresholds,
    min_overlap_frac: float,
    jaspar: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Excess clusters per threshold under all three merge criteria.

    Reported side by side deliberately: `single` chains and `mutual` cannot, so
    the gap between them bounds how much of any apparent redundancy is
    transitive closure rather than real duplication.
    """
    p_sym = symmetric_p(pwms, res, min_overlap_frac)
    rows = []
    for t in thresholds:
        row = {
            "p_threshold": t,
            "n_clusters": len(names),
            "n_pairs": int(((p_sym <= t).sum() - len(names)) // 2),
        }
        for criterion in MERGERS:
            comps = merge(criterion, names, pwms, res, p_sym, t, min_overlap_frac)
            n_comp = len(set(comps.values()))
            sizes = pd.Series(list(comps.values())).value_counts()
            row[f"n_components_{criterion}"] = n_comp
            row[f"excess_{criterion}"] = len(names) - n_comp
            row[f"excess_frac_{criterion}"] = round((len(names) - n_comp) / len(names), 4)
            row[f"largest_{criterion}"] = int(sizes.iloc[0]) if len(sizes) else 0
            if jaspar is not None:
                frac, n_groups, _, chance = jaspar_agreement(comps, jaspar)
                fam, _, _, fam_chance = jaspar_agreement(comps, jaspar, family=True)
                row[f"jaspar_agree_{criterion}"] = (
                    round(frac, 3) if frac is not None else np.nan
                )
                row[f"jaspar_chance_{criterion}"] = round(chance, 4)
                row[f"jaspar_enrich_{criterion}"] = (
                    round(frac / chance, 1) if frac is not None and chance > 0 else np.nan
                )
                row[f"family_agree_{criterion}"] = (
                    round(fam, 3) if fam is not None else np.nan
                )
                row[f"family_enrich_{criterion}"] = (
                    round(fam / fam_chance, 1)
                    if fam is not None and fam_chance > 0 else np.nan
                )
                row[f"jaspar_groups_{criterion}"] = n_groups
        rows.append(row)
    return pd.DataFrame(rows)


def jaspar_family(name: str) -> str:
    """Collapse a JASPAR name to a rough TF family.

    Exact-name agreement understates real concordance because JASPAR itself
    contains near-identical motifs: SP1/SP2/SP9 or ETV4/ETV6/ETV7 are barely
    distinguishable as PWMs, so two clusters that genuinely represent the same
    motif can carry different best-hit labels. Stripping a trailing paralogue
    number (and taking the first partner of a dimer) tests whether a
    disagreement is within-family, which is benign, or across families, which
    is real over-merging.
    """
    x = str(name)
    if "::" in x:
        x = x.split("::")[0]
    x = re.sub(r"[0-9]+[A-Z]?$", "", x)
    return (x.upper().rstrip("-_") or str(name).upper())


def agreement_chance(groups: list[list[str]], pool: list[str]) -> float:
    """Fraction of merged groups expected to agree if merging were random.

    Without this the observed agreement is uninterpretable. Names are heavily
    skewed -- on the real count head 306 named clusters carry 112 names, with
    SP9 alone claiming 31 -- so some agreement arises by chance, but far less
    than intuition suggests: the measured baseline is 2.6% for exact names and
    4.1% for families, making a 52% observation a ~20x enrichment rather than
    the "low agreement" it superficially resembles.

    For a group of size k drawn at random from the name distribution,
    P(all identical) = sum_i p_i^k, averaged over the observed group sizes.
    """
    if not groups or not pool:
        return 0.0
    counts = pd.Series(pool).value_counts()
    p = (counts / counts.sum()).to_numpy()
    return float(np.mean([np.sum(p ** len(g)) for g in groups]))


def jaspar_agreement(
    comps: dict[str, int], jaspar: dict[str, str], family: bool = False
) -> tuple[float | None, int, int, float]:
    """Fraction of merged groups whose members share a JASPAR name/family.

    The only external check available on whether a merge is real -- but only
    against its chance baseline, which is also returned. Interpret the ratio,
    not the raw fraction.

    Returns (fraction, n_groups, n_clusters_in_them, chance); fraction is None
    when nothing merged, which is not the same as 0% agreement.
    """
    label = (lambda n: jaspar_family(n)) if family else (lambda n: n)
    by_comp: dict[int, list[str]] = {}
    pool: list[str] = []
    for motif, comp in comps.items():
        name = jaspar.get(motif)
        if isinstance(name, str) and name:
            by_comp.setdefault(comp, []).append(label(name))
            pool.append(label(name))
    merged = [v for v in by_comp.values() if len(v) > 1]
    if not merged:
        return None, 0, 0, 0.0
    consistent = sum(1 for v in merged if len(set(v)) == 1)
    return (
        consistent / len(merged), len(merged), sum(len(v) for v in merged),
        agreement_chance(merged, pool),
    )


def mutual_best_pairs(
    names: list[str], p_sym: np.ndarray, threshold: float
) -> pd.DataFrame:
    """The mutual-best-hit pairs themselves, for review.

    These are the pairs behind the `mutual` excess figure -- each cluster's
    single closest match, reciprocated. They cannot chain, so every row is a
    self-contained claim that two clusters are the same motif, which is what
    makes them the right thing to eyeball.
    """
    masked = p_sym.copy()
    np.fill_diagonal(masked, np.inf)
    best = masked.argmin(axis=1)
    rows = []
    for i, j in enumerate(best):
        if best[j] == i and i < j and masked[i, j] <= threshold:
            rows.append({"motif_a": names[i], "motif_b": names[j],
                         "p_value": float(masked[i, j])})
    # Keep the schema when empty: pd.DataFrame([]) has no columns at all, which
    # writes a zero-byte TSV that cannot be read back.
    return pd.DataFrame(rows, columns=["motif_a", "motif_b", "p_value"])


def annotate_pairs(
    pairs: pd.DataFrame,
    widths: dict[str, int],
    cluster_metadata: Path | None,
    logo_paths: Path | None,
    out_dir: Path,
) -> pd.DataFrame:
    """Attach JASPAR names, seqlet counts and logo paths to each pair.

    Ordered by the larger of the two clusters' seqlet counts, so the pairs that
    most affect the lexicon size come first rather than being buried among rare
    ones.
    """
    if pairs.empty:
        return pairs

    out = pairs.copy()
    out["trimmed_len_a"] = out["motif_a"].map(widths)
    out["trimmed_len_b"] = out["motif_b"].map(widths)

    def cluster_id(motif: str) -> int | None:
        tail = str(motif).rsplit(".", 1)[-1]
        return int(tail) if tail.isdigit() else None

    if cluster_metadata is not None and Path(cluster_metadata).exists():
        meta = pd.read_csv(cluster_metadata, sep="\t")
        if {"cluster_final", "posneg"} <= set(meta.columns):
            key = (
                meta["posneg"].astype(str) + "_patterns."
                + meta["cluster_final"].astype(int).astype(str)
            )
            for col, newname in (
                ("jaspar_name", "jaspar"), ("jaspar_score", "score"),
                ("total_seqlets", "seqlets"), ("n_experiments", "n_exp"),
            ):
                if col in meta.columns:
                    m = dict(zip(key, meta[col]))
                    out[f"{newname}_a"] = out["motif_a"].map(m)
                    out[f"{newname}_b"] = out["motif_b"].map(m)

    if {"jaspar_a", "jaspar_b"} <= set(out.columns):
        both = out["jaspar_a"].notna() & out["jaspar_b"].notna()
        out["name_agree"] = np.where(
            both, out["jaspar_a"].astype(str) == out["jaspar_b"].astype(str), None
        )
        out["family_agree"] = np.where(
            both,
            out["jaspar_a"].map(lambda v: jaspar_family(v) if isinstance(v, str) else v)
            == out["jaspar_b"].map(lambda v: jaspar_family(v) if isinstance(v, str) else v),
            None,
        )

    if logo_paths is not None and Path(logo_paths).exists():
        lp = pd.read_csv(logo_paths, sep="\t")
        if "cluster_final" in lp.columns and "logo_fwd_svg" in lp.columns:
            logo = dict(zip(lp["cluster_final"], lp["logo_fwd_svg"]))

            def resolve(motif: str) -> str | None:
                cid = cluster_id(motif)
                rel = logo.get(cid) if cid is not None else None
                if rel is None:
                    return None
                abs_path = (MC_DIR / rel).resolve()
                try:
                    return os.path.relpath(abs_path, out_dir.resolve())
                except ValueError:
                    return str(abs_path)

            out["logo_a"] = out["motif_a"].map(resolve)
            out["logo_b"] = out["motif_b"].map(resolve)

    if "seqlets_a" in out.columns:
        out["max_seqlets"] = out[["seqlets_a", "seqlets_b"]].max(axis=1)
        out = out.sort_values("max_seqlets", ascending=False)
    else:
        out = out.sort_values("p_value")
    return out.reset_index(drop=True)


def write_pairs_html(pairs: pd.DataFrame, path: Path, head: str, threshold: float) -> None:
    """Side-by-side logo pairs, so a merge can be judged by eye in seconds.

    Disagreeing pairs are listed first and highlighted: those are where
    over-merging would show, and they are the ones worth the reviewer's time.
    """
    if pairs.empty or "logo_a" not in pairs.columns:
        return
    rows = pairs.copy()
    if "name_agree" in rows.columns:
        rows["_order"] = rows["name_agree"].map(
            lambda v: 0 if v is False else (1 if v is None else 2)
        )
        rows = rows.sort_values(["_order", "max_seqlets"], ascending=[True, False])

    html = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:system-ui,sans-serif;margin:18px;font-size:13px}",
        "table{border-collapse:collapse}td,th{padding:5px 9px;",
        "border-bottom:1px solid #ddd;vertical-align:middle}",
        "img{height:58px}tr.dis{background:#fff3f3}tr.unk{background:#fafafa}",
        "code{font-size:11px;color:#444}</style></head><body>",
        f"<h2>Mutual-best-hit pairs — {head} head, p &le; {threshold:g}</h2>",
        f"<p>{len(rows)} pairs. Each row is one cluster and its reciprocated "
        "closest match, i.e. a self-contained claim that these two are the same "
        "motif. Pink rows disagree on JASPAR name and are listed first — that "
        "is where over-merging would be visible. A pair can still be a true "
        "duplicate while disagreeing, since JASPAR contains near-identical "
        "motifs (SP1/SP2/SP9).</p>",
        "<table><tr><th>p</th><th>A</th><th>logo A</th><th>B</th>"
        "<th>logo B</th><th>JASPAR</th><th>seqlets</th></tr>",
    ]
    for _, r in rows.iterrows():
        agree = r.get("name_agree")
        cls = "dis" if agree is False else ("unk" if agree is None else "")
        fam = r.get("family_agree")
        ja, jb = r.get("jaspar_a"), r.get("jaspar_b")
        label = f"{ja} / {jb}"
        if agree is True:
            label += " &#10003;"
        elif fam is True:
            label += " (same family)"
        sa, sb = r.get("seqlets_a"), r.get("seqlets_b")
        html.append(
            f"<tr class='{cls}'><td>{r['p_value']:.1e}</td>"
            f"<td><code>{r['motif_a']}</code><br>{r.get('trimmed_len_a','?')}bp</td>"
            f"<td>{'<img src=\'' + str(r['logo_a']) + '\'>' if r.get('logo_a') else '—'}</td>"
            f"<td><code>{r['motif_b']}</code><br>{r.get('trimmed_len_b','?')}bp</td>"
            f"<td>{'<img src=\'' + str(r['logo_b']) + '\'>' if r.get('logo_b') else '—'}</td>"
            f"<td>{label}</td><td>{sa} / {sb}</td></tr>"
        )
    html.append("</table></body></html>")
    path.write_text("\n".join(html))
    print(f"Saved {path}", file=sys.stderr)


def plot_sweep(summary: pd.DataFrame, head: str, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    style = {"mutual": ("#1b7837", "o-"), "complete": ("#404040", "s-"),
             "single": ("#b2182b", "^-")}
    for criterion, (color, marker) in style.items():
        col = f"excess_frac_{criterion}"
        if col in summary.columns:
            ax.plot(summary["p_threshold"], summary[col], marker, color=color,
                    lw=1.6, ms=4, label=criterion)
    ax.set_xscale("log")
    ax.set_xlabel("TOMTOM p-value threshold")
    ax.set_ylabel("Excess clusters (fraction of lexicon)")
    ax.set_ylim(0, 1.0)
    ax.legend(title="merge criterion", frameon=False, fontsize=7, title_fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"Compendium redundancy — {head} head\n"
        f"{int(summary['n_clusters'].iloc[0])} clusters",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}", file=sys.stderr)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--head", default="count", choices=["profile", "count"],
        help="which compendium to self-compare (default: count)",
    )
    parser.add_argument(
        "--meme", type=Path, default=None, metavar="PATH",
        help="override motifcompendium_{head}_cluster_averages.meme",
    )
    parser.add_argument(
        "--modisco-h5", type=Path, default=None, metavar="PATH",
        help="use motifcompendium_{head}_cluster_averages.h5 instead of the "
             "MEME export. Strongly preferred: it retains contribution scores, "
             "and only those can locate a motif's core -- information content "
             "on a PFM cannot, because GC-rich flanks carry real compositional "
             "information. Pass 'auto' to use the default path",
    )
    parser.add_argument(
        "--cluster-metadata", type=Path, default=None, metavar="PATH",
        help="join JASPAR names onto components, to test whether name "
             "collisions correspond to real CWM redundancy",
    )
    parser.add_argument(
        "--logo-paths", type=Path, default=None, metavar="PATH",
        help="motifcompendium_{head}_cluster_logo_paths.tsv, for embedding "
             "logos in the review HTML (default: auto-detect)",
    )
    parser.add_argument(
        "--p-thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
        metavar="P", help="thresholds to sweep (default: 1e-12 ... 1e-2)",
    )
    parser.add_argument(
        "--report-threshold", type=float, default=1e-6, metavar="P",
        help="threshold whose components and pairs are written out (default: 1e-6)",
    )
    parser.add_argument(
        "--trim-threshold", type=float, default=0.3, metavar="F",
        help="keep positions whose information content clears this fraction of "
             "the motif's max, mirroring Fi-NeMo's CWM trim rule (default: 0.3)",
    )
    parser.add_argument(
        "--min-trim-len", type=int, default=6, metavar="BP",
        help="widen any motif trimmed below this, symmetrically (default: 6)",
    )
    parser.add_argument(
        "--drop-untrimmable", action="store_true",
        help="exclude clusters whose contributions are so diffuse that trimming "
             "does not shrink them at all. These have no locatable core, so any "
             "comparison against them is meaningless and they act as chaining "
             "hubs; in a real Fi-NeMo run the equivalent motifs received 1-140 "
             "hits out of 2.2M",
    )
    parser.add_argument(
        "--no-trim", action="store_true",
        help="compare untrimmed windows. MotifCompendium exports fixed-width "
             "CWM windows (50bp on the real count head) whose informative core "
             "is 6-15bp, so this mostly aligns low-information flanks and "
             "massively overstates redundancy; kept only for comparison",
    )
    parser.add_argument(
        "--linkage", default="complete", choices=list(MERGERS),
        help="merge criterion whose components are written out; all three are "
             "always reported in the sweep (default: complete)",
    )
    parser.add_argument(
        "--subset", type=Path, default=None, metavar="PATH",
        help="restrict to the motif names in this file (one per line, or a TSV "
             "with a 'motif'/'compendium_motif_name' column). Redundancy is "
             "best measured over the clusters a claim actually rests on -- the "
             "prevalence-filtered set, or the tissue-restricted ones -- both "
             "because that is the inflated number and because far fewer motifs "
             "chain far less",
    )
    parser.add_argument(
        "--min-overlap-frac", type=float, default=0.7, metavar="F",
        help="require the best alignment to cover this fraction of the shorter "
             "motif, suppressing short-inside-long matches (default: 0.7)",
    )
    parser.add_argument(
        "--name-regex", default=NAME_RE.pattern, metavar="RE",
        help="regex extracting a cluster id from a MEME metadata line",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="TOMTOM parallelism (default: -1)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figures" / "motif_atlas",
        metavar="DIR", help="output directory (default: figures/motif_atlas/)",
    )
    args = parser.parse_args()

    h5_path = args.modisco_h5
    if h5_path is not None and str(h5_path) == "auto":
        h5_path = MC_DIR / f"motifcompendium_{args.head}_cluster_averages.h5"

    cwms = None
    if h5_path is not None:
        if not h5_path.exists():
            print(f"ERROR: h5 not found: {h5_path}", file=sys.stderr)
            sys.exit(1)
        names, pwms, cwms = load_motifs_h5(h5_path)
        if len(names) < 2:
            print(
                f"ERROR: found {len(names)} motif(s) in {h5_path}. Its layout "
                "is not what was expected; datasets present:",
                file=sys.stderr,
            )
            for line in h5_tree(h5_path):
                print(f"  {line}", file=sys.stderr)
            print(
                "\nExpected a group per motif holding one of "
                f"{CWM_KEYS} (contributions), optionally with one of "
                f"{PFM_KEYS} (probabilities).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"{args.head}: loaded {len(names)} clusters from {h5_path.name} "
            f"(e.g. {names[:2]})",
            file=sys.stderr,
        )
    else:
        meme_path = args.meme or (
            MC_DIR / f"motifcompendium_{args.head}_cluster_averages.meme"
        )
        if not meme_path.exists():
            print(f"ERROR: MEME file not found: {meme_path}", file=sys.stderr)
            print(
                "Run src/bpnet/motifcompendium/cluster_motifs.py --head "
                f"{args.head} first.",
                file=sys.stderr,
            )
            sys.exit(1)
        names, pwms = load_motifs(meme_path, re.compile(args.name_regex))
        print(
            "NOTE: comparing the MEME export. Information content cannot locate "
            "a motif core on this data (GC-rich flanks clear any IC threshold), "
            "so prefer --modisco-h5 auto, which trims on contribution scores.",
            file=sys.stderr,
        )
        if len(names) < 2:
            print(f"ERROR: only {len(names)} motif(s) in {meme_path}", file=sys.stderr)
            sys.exit(1)

    if args.subset is not None:
        wanted = load_subset(args.subset)
        keep = [i for i, n in enumerate(names) if n in wanted]
        missing = wanted - set(names)
        if missing:
            print(
                f"NOTE: {len(missing)} subset name(s) not found in the motif source, e.g. "
                f"{sorted(missing)[:3]}",
                file=sys.stderr,
            )
        if len(keep) < 2:
            print(f"ERROR: subset matched {len(keep)} motif(s)", file=sys.stderr)
            sys.exit(1)
        names = [names[i] for i in keep]
        pwms = [pwms[i] for i in keep]
        if cwms is not None:
            cwms = [cwms[i] for i in keep]
        print(f"{args.head}: restricted to {len(names)} clusters", file=sys.stderr)

    raw_widths = np.array([m.shape[-1] for m in pwms])
    if args.no_trim:
        print(
            f"{args.head}: {len(names)} clusters, UNTRIMMED width "
            f"{raw_widths.min()}-{raw_widths.max()}",
            file=sys.stderr,
        )
    else:
        if cwms is not None:
            pwms, raw_widths = trim_by_cwm(
                pwms, cwms, args.trim_threshold, args.min_trim_len
            )
            how = "contribution"
        else:
            pwms, raw_widths = trim_motifs(pwms, args.trim_threshold, args.min_trim_len)
            how = "information-content"
        widths = np.array([m.shape[-1] for m in pwms])
        q = np.percentile(widths, [25, 50, 75])
        n_full = int((widths >= raw_widths).sum())
        print(
            f"{args.head}: {len(names)} clusters, {how}-trimmed "
            f"{raw_widths.min()}-{raw_widths.max()}bp -> {widths.min()}-{widths.max()}bp "
            f"at threshold {args.trim_threshold:g}\n"
            f"  width quartiles {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} bp; "
            f"{n_full} cluster(s) did not shrink at all",
            file=sys.stderr,
        )
        # Cluster averages smear contribution into the flanks -- averaging
        # variably offset instances does that -- so the same relative threshold
        # keeps a wider span here than Fi-NeMo achieves on per-experiment CWMs.
        if q[1] > 2 * FINEMO_MEDIAN_TRIM_BP:
            print(
                f"  NOTE: median {q[1]:.0f}bp is well above the ~"
                f"{FINEMO_MEDIAN_TRIM_BP}bp Fi-NeMo reaches on per-experiment "
                "CWMs at this threshold, so these cores are still wide. Raise "
                "--trim-threshold until the median is comparable, or spurious "
                "merging will persist.",
                file=sys.stderr,
            )
        if how == "information-content" and trimming_ineffective(widths, raw_widths):
            print(
                "WARNING: information-content trimming barely shrank these "
                f"motifs (median {int(np.median(widths))} of "
                f"{int(np.median(raw_widths))}bp), so the comparison is still "
                "dominated by flanks and redundancy will be overstated. Rerun "
                "with --modisco-h5 auto.",
                file=sys.stderr,
            )
    widths = np.array([m.shape[-1] for m in pwms])

    if args.drop_untrimmable and not args.no_trim:
        keep = [i for i in range(len(names)) if widths[i] < raw_widths[i]]
        dropped = len(names) - len(keep)
        if len(keep) < 2:
            print(
                f"ERROR: only {len(keep)} cluster(s) trimmed at all; "
                "--drop-untrimmable would leave nothing to compare",
                file=sys.stderr,
            )
            sys.exit(1)
        names = [names[i] for i in keep]
        pwms = [pwms[i] for i in keep]
        raw_widths = raw_widths[keep]
        widths = widths[keep]
        print(
            f"{args.head}: dropped {dropped} untrimmable cluster(s) with no "
            f"locatable core, {len(names)} remain",
            file=sys.stderr,
        )

    jaspar_map: dict[str, str] = {}
    if args.cluster_metadata is not None and args.cluster_metadata.exists():
        _meta = pd.read_csv(args.cluster_metadata, sep="\t")
        if {"cluster_final", "posneg"} <= set(_meta.columns) and "jaspar_name" in _meta.columns:
            _key = (
                _meta["posneg"].astype(str) + "_patterns."
                + _meta["cluster_final"].astype(int).astype(str)
            )
            jaspar_map = {
                k: v for k, v in zip(_key, _meta["jaspar_name"])
                if isinstance(v, str) and v
            }
            overlap = len(set(names) & set(jaspar_map))
            print(
                f"{args.head}: {overlap}/{len(names)} clusters matched to a "
                "JASPAR name for the agreement check",
                file=sys.stderr,
            )

    res = self_compare(pwms, args.n_jobs)
    summary = sweep(
        names, pwms, res, args.p_thresholds, args.min_overlap_frac,
        jaspar=jaspar_map or None,
    )

    thresholds = sorted(set(args.p_thresholds) | {args.report_threshold})
    pairs = build_pairs(names, pwms, res, max(thresholds), args.min_overlap_frac)
    p_sym = symmetric_p(pwms, res, args.min_overlap_frac)
    comps = merge(
        args.linkage, names, pwms, res, p_sym, args.report_threshold,
        args.min_overlap_frac,
    )
    comp_df = pd.DataFrame(
        {"motif": names, "component": [comps[n] for n in names],
         "width": widths}
    ).sort_values(["component", "motif"])

    if args.cluster_metadata is not None and args.cluster_metadata.exists():
        meta = pd.read_csv(args.cluster_metadata, sep="\t")
        if {"cluster_final", "posneg"} <= set(meta.columns):
            key = (
                meta["posneg"].astype(str) + "_patterns."
                + meta["cluster_final"].astype(int).astype(str)
            )
            join = pd.DataFrame({"motif": key})
            for c in ("jaspar_name", "jaspar_score", "total_seqlets", "n_experiments"):
                if c in meta.columns:
                    join[c] = meta[c].values
            comp_df = comp_df.merge(join, on="motif", how="left")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"motif_redundancy_{args.head}"
    pairs.to_csv(stem.parent / f"{stem.name}_pairs.tsv", sep="\t", index=False)
    summary.to_csv(stem.parent / f"{stem.name}_summary.tsv", sep="\t", index=False)
    comp_df.to_csv(stem.parent / f"{stem.name}_components.tsv", sep="\t", index=False)
    for suffix in ("pairs", "summary", "components"):
        print(f"Saved {stem.parent / f'{stem.name}_{suffix}.tsv'}", file=sys.stderr)
    logo_paths = args.logo_paths
    if logo_paths is None:
        candidate = MC_DIR / f"motifcompendium_{args.head}_cluster_logo_paths.tsv"
        logo_paths = candidate if candidate.exists() else None

    width_map = {n: int(w) for n, w in zip(names, widths)}
    mutual = annotate_pairs(
        mutual_best_pairs(names, p_sym, args.report_threshold),
        width_map, args.cluster_metadata, logo_paths, args.out_dir,
    )
    mutual_path = stem.parent / f"{stem.name}_mutual_pairs.tsv"
    mutual.to_csv(mutual_path, sep="\t", index=False)
    print(f"Saved {mutual_path}  ({len(mutual)} pairs)", file=sys.stderr)
    write_pairs_html(
        mutual, stem.parent / f"{stem.name}_mutual_pairs.html",
        args.head, args.report_threshold,
    )
    if "name_agree" in mutual.columns and len(mutual):
        disagree = mutual[mutual["name_agree"] == False]  # noqa: E712
        print(
            f"  {len(disagree)} of {len(mutual)} pairs disagree on JASPAR name "
            "and are listed first in the HTML -- review those to judge whether "
            "the merges are real",
            file=sys.stderr,
        )

    plot_sweep(summary, args.head, stem)

    with pd.option_context("display.width", 200):
        print("\nRedundancy sweep:", file=sys.stderr)
        print(summary.to_string(index=False), file=sys.stderr)

    at = summary[summary["p_threshold"] == args.report_threshold]
    if not at.empty:
        row = at.iloc[0]
        print(f"\nAt p <= {args.report_threshold:g}:", file=sys.stderr)
        for criterion in MERGERS:
            print(
                f"  {criterion:<9} {int(row['n_clusters'])} -> "
                f"{int(row[f'n_components_{criterion}'])} clusters "
                f"({int(row[f'excess_{criterion}'])} excess, "
                f"{row[f'excess_frac_{criterion}']:.1%}); largest group "
                f"{int(row[f'largest_{criterion}'])}",
                file=sys.stderr,
            )
        gap = row["excess_frac_single"] - row["excess_frac_complete"]
        if gap > 0.2:
            print(
                f"\nWARNING: single-linkage exceeds complete-linkage by "
                f"{gap:.0%} of the lexicon, so connected components are chaining "
                "heavily at this threshold. Quote the complete or mutual "
                "figures, and treat single-linkage as an upper bound only.",
                file=sys.stderr,
            )

    if jaspar_map and not at.empty:
        row = at.iloc[0]
        print(
            f"\nJASPAR-name agreement within merged groups at "
            f"p <= {args.report_threshold:g}:",
            file=sys.stderr,
        )
        for criterion in MERGERS:
            col = f"jaspar_agree_{criterion}"
            if col not in row or pd.isna(row[col]):
                print(f"  {criterion:<9} (nothing merged)", file=sys.stderr)
                continue
            print(
                f"  {criterion:<9} name {row[col]:.0%} vs {row[f'jaspar_chance_{criterion}']:.1%} "
                f"chance ({row[f'jaspar_enrich_{criterion}']:.0f}x) | "
                f"family {row[f'family_agree_{criterion}']:.0%} "
                f"({row[f'family_enrich_{criterion}']:.0f}x) | "
                f"{int(row[f'jaspar_groups_{criterion}'])} groups",
                file=sys.stderr,
            )
        print(
            "\nRead the enrichment over chance, not the raw percentage. JASPAR\n"
            "names are heavily skewed, so some agreement is free -- but only a\n"
            "few percent, which makes even a ~50% observation a large\n"
            "enrichment and evidence the merges are real. Family-level\n"
            "agreement additionally absorbs JASPAR's own redundancy (SP1/SP2/SP9\n"
            "are near-identical PWMs), so a merge can be genuine while the two\n"
            "clusters carry different best-hit labels.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
