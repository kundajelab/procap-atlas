"""Call motif instances in BPNet attributions with Fi-NeMo.

Builds a peak coordinate file aligned with the saved OHE/attribution arrays,
converts them to Fi-NeMo's region format, and runs Fi-NeMo hit calling against
this experiment's own per-experiment MoDISco motif set
(modisco/bpnet/{experiment}_{head}.modisco.h5) -- the same scale of motif set
Kelly Cochran's ProCapNet run_finemo.py used, and much smaller than the
atlas-wide MotifCompendium compendium. Hits are therefore fast and free of
competitive-reconstruction pathologies from cell-type-specific motifs that
were never even discovered in this experiment's own attributions (Fi-NeMo's
joint sparse regression makes every motif in the h5 compete for the same
residual, so including atlas-wide motifs this experiment doesn't express just
adds noise and GPU cost). A hit's `motif_name` here (e.g. `pos_patterns.pattern_3`)
is only meaningful within this experiment; run
`src/bpnet/hitcall/link_hits_to_compendium.py` afterward to relabel hits with
the atlas-wide MotifCompendium cluster ID they belong to, for cross-experiment
comparability. Pass `--modisco-h5` explicitly to call hits against the shared
compendium h5 instead (`src/bpnet/motifcompendium/cluster_motifs.py` output),
reproducing the old atlas-wide-hit-calling behavior.

Default settings (region width, global lambda, CWM trim threshold) follow
Kelly Cochran's ProCapNet run_finemo.py:
https://github.com/kellycochran/procapnet_allscripts/blob/main/GENCODE/src/attributions_genomewide/run_finemo.py

peaks.narrowPeak/regions.npz (trim-independent) are cached in
hitcalls/bpnet/{model_dir_name}_{head}/ and reused across trim
configurations -- regions.npz is written atomically (temp file + rename) and
its zip validity is checked before reuse, so a job killed mid-write (SLURM
pre-emption/OOM/walltime) can't leave a corrupt cache file for a later run to
silently pick up. Non-default trimming (--cwm-trim-threshold,
--cwm-trim-thresholds, --cwm-trim-coords) instead moves finemo call-hits's
own output into a trim-suffixed subdirectory of that same directory (e.g.
{model_dir_name}_{head}/trimcoords-{file_stem}/) so rerunning with a
different trim configuration doesn't silently overwrite a previous run's
hits, and so launch.py's "already called" skip check can tell them apart.

Usage:
    python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM
    python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM --head count
    python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_gc0.1
    python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM --global-lambda 0.6
    python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM --modisco-h5 modisco/bpnet/ENCSR882DWM_profile.modisco.h5
    python src/bpnet/hitcall/call_hits_bpnet.py -e ENCSR882DWM --cwm-trim-thresholds hitcall/inr_trim_thresholds.tsv
"""

import argparse
import subprocess
import sys
import zipfile
from contextlib import ExitStack
from itertools import chain
from pathlib import Path

# Names of every stage of post-call-hits filtering, most- to least-processed.
# Shared by report_bpnet.py, filter_low_confidence_hits.py, and
# link_hits_to_compendium.py so they all resolve "the most-processed hits
# available" the same way.
HITS_FILE_STAGES = [
    "hits_filtered.tsv",
    "hits_confidence_filtered.tsv",
    "hits_dedensified.tsv",
    "hits_unique.tsv",
]

import pandas as pd
import yaml
from tangermeme.io import extract_loci

import compressed_io

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
FASTA = str(REPO_ROOT / "data" / "hg38.fa")
BLACKLIST = str(REPO_ROOT / "data" / "hg38.blacklist.bed.gz")
IN_WINDOW = 2114
DEFAULT_CWM_TRIM_THRESHOLD = 0.3


def trim_suffix(cwm_trim_threshold, cwm_trim_thresholds, cwm_trim_coords):
    """Build an output-dir suffix that distinguishes non-default trimming.

    Without this, every trim setting (--cwm-trim-threshold,
    --cwm-trim-thresholds, --cwm-trim-coords) wrote to the same
    hitcalls/bpnet/{model_dir_name}_{head}/ directory, so rerunning with a
    different trim configuration (e.g. compute_trim_floor.py's
    --cwm-trim-coords floor) silently overwrote the previous run's hits, and
    launch.py's "already called" skip check couldn't tell the two apart.
    """
    parts = []
    if cwm_trim_threshold != DEFAULT_CWM_TRIM_THRESHOLD:
        parts.append(f"trim{cwm_trim_threshold}")
    if cwm_trim_thresholds:
        parts.append(f"trimthresh-{Path(cwm_trim_thresholds).stem}")
    if cwm_trim_coords:
        parts.append(f"trimcoords-{Path(cwm_trim_coords).stem}")
    return ("_" + "_".join(parts)) if parts else ""


def resolve_experiment_paths(experiment, head, min_trim_len=None, model_dir=None):
    """Resolve (exp_dir, hits_dir, trim_coords, suffix) from --min-trim-len
    alone -- the block every hitcall/ script otherwise duplicated by hand.

    model_dir_name defaults to `experiment` when model_dir is None. exp_dir
    is regions.npz/peaks.narrowPeak's trim-independent home; hits_dir is its
    trim-suffixed subdirectory (or exp_dir itself for default trimming)
    where hits/report/ actually live. trim_coords is the modisco/bpnet
    trim-coords TSV path this --min-trim-len implies, or None -- existence
    isn't checked here; resolve_hits_path/compressed_io.exists() do that
    where it matters. Scripts driven by the full --cwm-trim-threshold/
    --cwm-trim-thresholds/--cwm-trim-coords flag surface directly (this
    module, report_bpnet.py) call trim_suffix() themselves instead, since
    --min-trim-len is just one convenience layered on top of that surface.
    """
    model_dir_name = Path(model_dir).name if model_dir else experiment
    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    trim_coords = (
        modisco_dir / f"{experiment}_{head}_trim_coords_min{min_trim_len}bp.tsv"
        if min_trim_len is not None
        else None
    )
    suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, trim_coords)
    exp_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir
    return exp_dir, hits_dir, trim_coords, suffix


def resolve_hits_path(hits_dir, stages=HITS_FILE_STAGES, verbose=False):
    """Find the most-processed hits file in `stages` (most- to
    least-processed order) that actually exists in `hits_dir`, treating a
    candidate as stale -- and falling through to the next, less-processed
    one -- if it's older than any less-processed file behind it.

    Without this, rerunning an earlier stage (e.g. filter_repeat_density.py
    with a different --cluster-window) after a later stage already ran
    (e.g. filter_low_confidence_hits.py) silently overwrites the earlier
    file in place, but the later, now-stale file still exists and still
    looks preferable by name alone -- report_bpnet.py and friends would keep
    reading it and never see the rerun's effect at all.
    """
    # Resolve each stage to whichever of its plain/.gz forms is present, so a
    # partially compressed tree still resolves. Staleness is then compared on
    # the files that actually exist, which is the same check as before.
    paths = [
        compressed_io.resolve(hits_dir / name, missing_ok=True) for name in stages
    ]
    for i, path in enumerate(paths):
        if path is None:
            continue
        stale_against = next(
            (later for later in paths[i + 1 :]
             if later is not None and later.stat().st_mtime > path.stat().st_mtime),
            None,
        )
        if stale_against is not None:
            if verbose:
                print(
                    f"WARNING: {path.name} is older than {stale_against.name} -- "
                    "treating it as stale and preferring an earlier-stage file instead.",
                    file=sys.stderr,
                )
            continue
        return path
    return None


def build_peaks_narrowpeak(peaks_path, chrom_splits, out_path):
    """Write a Fi-NeMo-compatible narrowPeak file whose rows line up 1:1 with
    the OHE/attribution arrays written by save_ohe.py/attribute_bpnet.py.

    Those scripts extract loci with tangermeme's extract_loci, which silently
    drops any peak that falls off a chromosome end or overlaps the blacklist,
    so the saved arrays can have fewer rows than the filtered_peaks file and in
    a different order than one might assume. This repeats the same filtering
    (with return_mask=True this time) and encodes each surviving peak's window
    midpoint as a synthetic narrowPeak summit (start=mid, summit=0), which
    Fi-NeMo reconstructs internally as `peak_start + summit`.
    """
    loci = pd.read_csv(
        peaks_path,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )
    all_chrom = list(chain.from_iterable(chrom_splits.values()))
    _, kept_mask = extract_loci(
        loci=loci,
        sequences=FASTA,
        chroms=all_chrom,
        in_window=IN_WINDOW,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        exclusion_lists=[BLACKLIST],
        return_mask=True,
    )
    kept = loci[kept_mask.numpy()].reset_index(drop=True)
    mid = kept["start"] + (kept["end"] - kept["start"]) // 2

    narrowpeak = pd.DataFrame(
        {
            "chrom": kept["chrom"],
            "start": mid,
            "end": mid + 1,
            "name": kept["chrom"]
            + ":"
            + kept["start"].astype(str)
            + "-"
            + kept["end"].astype(str),
            "score": 0,
            "strand": ".",
            "signal": 0.0,
            "pval": -1.0,
            "qval": -1.0,
            "summit": 0,
        }
    )
    compressed_io.write_bgzip(narrowpeak, out_path)
    return len(narrowpeak)


def run(cmd, verbose):
    if verbose:
        print(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def ensure_regions_npz(peaks_path, chrom_splits, ohe_path, attr_path, out_dir,
                        region_width, verbose=False):
    """Return out_dir/regions.npz, reusing it if present and valid, else
    rebuilding it (and its peaks.narrowPeak input) from the experiment's own
    filtered peaks + saved OHE/attribution arrays.

    Factored out of call_hits_bpnet.py's main() so extract_regions_bpnet.py
    can rebuild just this cache -- e.g. after regions.npz was deleted
    directly while hits.tsv and every downstream filter/report stage were
    left alone -- without also unconditionally re-running finemo call-hits,
    which has no skip-if-unchanged logic of its own and would overwrite
    those later stages for no reason.
    """
    peaks_narrowpeak = compressed_io.compressed_name(out_dir / "peaks.narrowPeak")
    regions_npz = out_dir / "regions.npz"
    if regions_npz.exists() and zipfile.is_zipfile(regions_npz):
        print(f"Reusing existing {regions_npz}")
        return regions_npz

    if regions_npz.exists():
        print(
            f"WARNING: existing {regions_npz} is not a valid .npz file "
            "(likely left behind by an interrupted run, e.g. a "
            "pre-empted/OOM-killed SLURM job) -- regenerating it.",
            file=sys.stderr,
        )
    n_peaks = build_peaks_narrowpeak(peaks_path, chrom_splits, peaks_narrowpeak)
    print(f"Wrote {n_peaks} peaks aligned to saved attributions: {peaks_narrowpeak}")
    # Write to a temporary path and rename into place only once finemo
    # finishes successfully, so a job killed mid-write (pre-emption, OOM,
    # walltime) can never leave a truncated/corrupt regions.npz at the
    # canonical cache path for a later run to silently "reuse".
    tmp_regions_npz = out_dir / "regions.tmp.npz"
    run(
        [
            "finemo",
            "extract-regions-modisco-fmt",
            "-s",
            str(ohe_path),
            "-a",
            str(attr_path),
            "-p",
            str(peaks_narrowpeak),
            "-o",
            str(tmp_regions_npz),
            "-w",
            str(region_width),
        ],
        verbose,
    )
    tmp_regions_npz.rename(regions_npz)
    return regions_npz


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
        help="attribution/motif head to call hits against (default: profile)",
    )
    parser.add_argument(
        "--modisco-h5",
        type=str,
        default=None,
        help=(
            "override the modisco-lite-format h5 of motif CWMs to call hits "
            "against; default is this experiment's own "
            "modisco/bpnet/{experiment}_{head}.modisco.h5. Pass the shared "
            "MotifCompendium cluster-average file instead "
            "(motifcompendium/bpnet/motifcompendium_{head}_cluster_averages.h5) "
            "to call hits atlas-wide against every experiment's motifs at "
            "once (slower, and prone to competing against motifs this "
            "experiment never expresses); prefer running "
            "link_hits_to_compendium.py after the default per-experiment run "
            "instead."
        ),
    )
    parser.add_argument(
        "--region-width",
        type=int,
        default=IN_WINDOW,
        help=(
            "width of the region fed to Fi-NeMo hit calling, centered on each "
            "peak (default: 2114, the model's full input window). Fi-NeMo's "
            "own default of 1000 would crop the saved attributions; Kelly "
            "Cochran's ProCapNet run used the full input window instead."
        ),
    )
    parser.add_argument(
        "--global-lambda",
        type=float,
        default=0.7,
        help="Fi-NeMo L1 sparsity weight (default: 0.7, both Fi-NeMo's and "
        "Kelly Cochran's ProCapNet default)",
    )
    parser.add_argument(
        "--cwm-trim-threshold",
        type=float,
        default=DEFAULT_CWM_TRIM_THRESHOLD,
        help="default motif trimming threshold (default: 0.3, Fi-NeMo/ProCapNet default)",
    )
    parser.add_argument(
        "--cwm-trim-thresholds",
        type=str,
        default=None,
        help=(
            "path to a Fi-NeMo -T/--cwm-trim-thresholds mapping file "
            "(motif_name<TAB>threshold, one per line) overriding "
            "--cwm-trim-threshold for specific motifs. Useful for very short "
            "core-promoter motifs (e.g. Initiator elements), which the default "
            "threshold can over-trim; Kelly Cochran's ProCapNet run patched "
            "Fi-NeMo's trimming with a minimum-length floor for this reason, "
            "but the current Fi-NeMo release has no such floor built in."
        ),
    )
    parser.add_argument(
        "--cwm-trim-coords",
        type=str,
        default=None,
        help=(
            "path to a Fi-NeMo -R/--cwm-trim-coords mapping file "
            "(motif_name<TAB>start<TAB>end, one per line) giving explicit "
            "trim coordinates for specific motifs, bypassing threshold-based "
            "trimming entirely for those motifs."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help=(
            "Fi-NeMo region batch size (default: 2000, Kelly Cochran's "
            "ProCapNet default, sized for a per-experiment MoDISco motif set "
            "of tens of motifs). If you override --modisco-h5 to point at "
            "the atlas-wide MotifCompendium compendium instead, lower this a "
            "lot (e.g. --batch-size 16) -- that much larger motif set makes "
            "GPU memory per batch far higher, and 2000/500/64 all reliably "
            "OOM'd on a 44GB GPU against it."
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="JIT-compile the Fi-NeMo optimizer (may not work on older GPUs)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = config["experiments"]
    if args.experiment not in experiments:
        print(f"Error: {args.experiment} not found in config", file=sys.stderr)
        sys.exit(1)
    exp = experiments[args.experiment]
    processed = exp.get("processed", {})
    peaks_path = str(REPO_ROOT / processed["filtered_peaks"])

    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = {int(k): v for k, v in yaml.safe_load(f)["folds"].items()}

    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else REPO_ROOT / "models" / "bpnet" / args.experiment
    )
    model_dir_name = model_dir.name

    attr_dir = REPO_ROOT / "attributions" / "bpnet"
    ohe_path = attr_dir / f"{args.experiment}_ohe.npz"
    attr_path = attr_dir / f"{model_dir_name}_{args.head}.npz"
    if args.modisco_h5:
        modisco_h5 = Path(args.modisco_h5)
    else:
        modisco_h5 = (
            REPO_ROOT
            / "modisco"
            / "bpnet"
            / f"{args.experiment}_{args.head}.modisco.h5"
        )

    for path, label in [
        (peaks_path, "filtered_peaks"),
        (ohe_path, "OHE sequences"),
        (attr_path, "attributions"),
        (modisco_h5, "motif CWMs"),
        (args.cwm_trim_thresholds, "cwm-trim-thresholds mapping"),
        (args.cwm_trim_coords, "cwm-trim-coords mapping"),
    ]:
        # compressed_io.exists() accepts either a path's plain or .gz form,
        # since compute_trim_floor.py's trim-coords/-thresholds mapping
        # files get compressed like any other .tsv in this tree.
        if path is not None and not compressed_io.exists(path):
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # peaks.narrowPeak/regions.npz depend only on the experiment/head/
    # region-width, not on trimming, so they live in the plain
    # {model_dir_name}_{head}/ directory and are reused across every trim
    # configuration for that (experiment, head) rather than being
    # regenerated (and duplicated) per trim setting. Only finemo call-hits's
    # own output -- which does depend on trimming -- moves into a
    # trim-suffixed subdirectory.
    out_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{args.head}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # bgzipped. `finemo extract-regions` reads this with polars.scan_csv,
    # which decompresses gzip transparently; bgzip is a valid gzip stream.
    regions_npz = ensure_regions_npz(
        peaks_path, chrom_splits, ohe_path, attr_path, out_dir,
        args.region_width, args.verbose,
    )

    suffix = trim_suffix(
        args.cwm_trim_threshold, args.cwm_trim_thresholds, args.cwm_trim_coords
    )
    call_hits_dir = out_dir / suffix.lstrip("_") if suffix else out_dir
    call_hits_dir.mkdir(parents=True, exist_ok=True)

    call_hits_cmd = [
        "finemo",
        "call-hits",
        "-r",
        str(regions_npz),
        "-m",
        str(modisco_h5),
        "-o",
        str(call_hits_dir),
        "-t",
        str(args.cwm_trim_threshold),
        "-l",
        str(args.global_lambda),
        "-b",
        str(args.batch_size),
    ]
    with ExitStack() as stack:
        if args.cwm_trim_thresholds:
            trim_thresholds = stack.enter_context(
                compressed_io.ensure_plain(args.cwm_trim_thresholds)
            )
            call_hits_cmd += ["-T", str(trim_thresholds)]
        if args.cwm_trim_coords:
            trim_coords = stack.enter_context(
                compressed_io.ensure_plain(args.cwm_trim_coords)
            )
            call_hits_cmd += ["-R", str(trim_coords)]
        if args.compile:
            call_hits_cmd.append("-J")
        run(call_hits_cmd, args.verbose)

    print(f"\nFi-NeMo hits saved to {call_hits_dir}")


if __name__ == "__main__":
    main()
