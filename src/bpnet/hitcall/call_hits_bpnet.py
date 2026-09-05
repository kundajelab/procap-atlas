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
configurations. Non-default trimming (--cwm-trim-threshold,
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
from itertools import chain
from pathlib import Path

import pandas as pd
import yaml
from tangermeme.io import extract_loci

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
    narrowpeak.to_csv(out_path, sep="\t", header=False, index=False)
    return len(narrowpeak)


def run(cmd, verbose):
    if verbose:
        print(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


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
        if path is not None and not Path(path).exists():
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

    peaks_narrowpeak = out_dir / "peaks.narrowPeak"
    regions_npz = out_dir / "regions.npz"
    if regions_npz.exists():
        print(f"Reusing existing {regions_npz}")
    else:
        n_peaks = build_peaks_narrowpeak(peaks_path, chrom_splits, peaks_narrowpeak)
        print(
            f"Wrote {n_peaks} peaks aligned to saved attributions: {peaks_narrowpeak}"
        )
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
                str(regions_npz),
                "-w",
                str(args.region_width),
            ],
            args.verbose,
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
    if args.cwm_trim_thresholds:
        call_hits_cmd += ["-T", args.cwm_trim_thresholds]
    if args.cwm_trim_coords:
        call_hits_cmd += ["-R", args.cwm_trim_coords]
    if args.compile:
        call_hits_cmd.append("-J")
    run(call_hits_cmd, args.verbose)

    print(f"\nFi-NeMo hits saved to {call_hits_dir}")


if __name__ == "__main__":
    main()
