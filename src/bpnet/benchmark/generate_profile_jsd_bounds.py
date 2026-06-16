#!/usr/bin/env python3
"""Generate replicate and average-profile BPNet profile JSD bounds.

For each experiment/fold this script computes:
  - replicate profile JSD: observed profile concordance between two replicate
    raw BigWig groups.
  - average-profile JSD: each held-out observed peak profile compared against
    the average observed profile across all peaks in the same test fold.

The primary outputs are:
  - configs/bpnet_replicates.yaml
  - performance_metrics/bpnet_bounds/{experiment}.json
  - performance_metrics/bpnet_bounds/procap-atlas_profile_jsd_bounds.tsv
  - performance_metrics/bpnet_bounds/per_locus/{experiment}.npz, if requested

Examples:
    python src/bpnet/benchmark/generate_profile_jsd_bounds.py --write-replicates-only
    python src/bpnet/benchmark/generate_profile_jsd_bounds.py -e ENCSR882DWM --save-per-locus
    python src/bpnet/benchmark/generate_profile_jsd_bounds.py --consolidate-only
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tangermeme.io import extract_loci

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
REPLICATE_CONFIG_PATH = REPO_ROOT / "configs" / "bpnet_replicates.yaml"
RAW_BIGWIG_DIR = REPO_ROOT / "data" / "raw" / "bigwigs"
FASTA = str(REPO_ROOT / "data" / "hg38.fa")
BLACKLIST = str(REPO_ROOT / "data" / "hg38.blacklist.bed.gz")
DEFAULT_OUT_DIR = REPO_ROOT / "performance_metrics" / "bpnet_bounds"
DEFAULT_TSV = DEFAULT_OUT_DIR / "procap-atlas_profile_jsd_bounds.tsv"
IGNORE_BASES = list("QWERYUIOPSDFHJKLZXVBNM")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--experiment",
        action="append",
        default=None,
        help="experiment accession ID; may be repeated (default: all experiments)",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--chrom-splits", type=Path, default=CHROM_SPLITS_PATH)
    parser.add_argument("--replicate-config", type=Path, default=REPLICATE_CONFIG_PATH)
    parser.add_argument("--raw-bigwig-dir", type=Path, default=RAW_BIGWIG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--in-window", type=int, default=2114)
    parser.add_argument("--out-window", type=int, default=1000)
    parser.add_argument(
        "--save-per-locus",
        action="store_true",
        help="write per-locus bound arrays to performance_metrics/bpnet_bounds/per_locus/",
    )
    parser.add_argument(
        "--write-replicates-only",
        action="store_true",
        help="write configs/bpnet_replicates.yaml and exit",
    )
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help="only consolidate existing JSON files into the TSV",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip experiments whose bounds JSON already exists",
    )
    return parser.parse_args()


def split_replicate_pairs(pl_bigwigs: list[str], mn_bigwigs: list[str]) -> dict:
    n_pairs = min(len(pl_bigwigs), len(mn_bigwigs))
    pairs = [
        {"pl_bigwig": pl_bigwigs[i], "mn_bigwig": mn_bigwigs[i]} for i in range(n_pairs)
    ]
    if n_pairs < 2:
        return {
            "status": "skipped",
            "reason": f"requires at least 2 paired plus/minus BigWigs, found {n_pairs}",
            "n_pairs": n_pairs,
            "pairs": pairs,
            "groups": {},
        }

    # Alternating split balances 4-replicate experiments while preserving the
    # natural 2-replicate split.
    group_a = pairs[::2]
    group_b = pairs[1::2]
    return {
        "status": "ok",
        "reason": "",
        "n_pairs": n_pairs,
        "pairs": pairs,
        "groups": {
            "a": group_a,
            "b": group_b,
        },
    }


def build_replicate_config(config: dict) -> dict:
    experiments = {}
    for exp_id, exp in config["experiments"].items():
        split = split_replicate_pairs(
            exp.get("pl_bigwigs", []), exp.get("mn_bigwigs", [])
        )
        experiments[exp_id] = {
            "biosample": exp.get("biosample", ""),
            **split,
        }
    return {
        "description": (
            "Replicate groups derived by pairing pl_bigwigs[i] with "
            "mn_bigwigs[i] from configs/experiment_config.yaml, then splitting "
            "paired raw BigWigs into alternating groups."
        ),
        "raw_bigwig_dir": str(RAW_BIGWIG_DIR.relative_to(REPO_ROOT)),
        "experiments": experiments,
    }


def write_replicate_config(config: dict, path: Path) -> dict:
    replicates = build_replicate_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(replicates, f, default_flow_style=False, sort_keys=False)
    n_ok = sum(1 for exp in replicates["experiments"].values() if exp["status"] == "ok")
    print(
        f"Wrote {len(replicates['experiments'])} experiments to {path} ({n_ok} usable)"
    )
    return replicates


def group_signal_filenames(group: list[dict]) -> tuple[list[str], list[str]]:
    return (
        [entry["pl_bigwig"] for entry in group],
        [entry["mn_bigwig"] for entry in group],
    )


def abs_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.abs(np.asarray(array, dtype=float))


def profile_from_channels(
    y_abs: np.ndarray, pl_slice: slice, mn_slice: slice
) -> np.ndarray:
    plus = y_abs[:, pl_slice, :].sum(axis=1)
    minus = y_abs[:, mn_slice, :].sum(axis=1)
    return np.stack([plus, minus], axis=1)


def normalize_profiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(values.shape[0], -1)
    totals = flat.sum(axis=1, keepdims=True)
    valid = (
        np.isfinite(flat).all(axis=1) & np.isfinite(totals[:, 0]) & (totals[:, 0] > 0)
    )
    normalized = np.zeros_like(flat, dtype=float)
    normalized[valid] = flat[valid] / totals[valid]
    return normalized, valid


def js_divergence_and_distance(
    p: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p_norm, p_valid = normalize_profiles(p)
    q_norm, q_valid = normalize_profiles(q)
    valid = p_valid & q_valid

    divergence = np.full(p_norm.shape[0], np.nan, dtype=float)
    distance = np.full(p_norm.shape[0], np.nan, dtype=float)
    if not valid.any():
        return divergence, distance

    p_valid_norm = p_norm[valid]
    q_valid_norm = q_norm[valid]
    m = 0.5 * (p_valid_norm + q_valid_norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_kl = np.where(
            p_valid_norm > 0, p_valid_norm * np.log(p_valid_norm / m), 0.0
        ).sum(axis=1)
        q_kl = np.where(
            q_valid_norm > 0, q_valid_norm * np.log(q_valid_norm / m), 0.0
        ).sum(axis=1)
    divergence[valid] = 0.5 * (p_kl + q_kl)
    distance[valid] = np.sqrt(divergence[valid])
    return divergence, distance


def average_profile_baseline(full_profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized, valid = normalize_profiles(full_profile)
    divergence = np.full(full_profile.shape[0], np.nan, dtype=float)
    distance = np.full(full_profile.shape[0], np.nan, dtype=float)
    if not valid.any():
        return divergence, distance

    average_flat = normalized[valid].mean(axis=0)
    average_flat = average_flat / average_flat.sum()
    average_profile = np.broadcast_to(
        average_flat.reshape(1, *full_profile.shape[1:]), full_profile.shape
    )
    divergence, distance = js_divergence_and_distance(full_profile, average_profile)
    return divergence, distance


def median_finite(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(np.median(values))


def extract_fold_profiles(
    loci: pd.DataFrame,
    chroms: list[str],
    group_a: list[dict],
    group_b: list[dict],
    full_pl_bigwig: str,
    full_mn_bigwig: str,
    raw_bigwig_dir: Path,
    in_window: int,
    out_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_pl, a_mn = group_signal_filenames(group_a)
    b_pl, b_mn = group_signal_filenames(group_b)

    signal_paths = (
        [str(raw_bigwig_dir / fn) for fn in a_pl]
        + [str(raw_bigwig_dir / fn) for fn in a_mn]
        + [str(raw_bigwig_dir / fn) for fn in b_pl]
        + [str(raw_bigwig_dir / fn) for fn in b_mn]
        + [full_pl_bigwig, full_mn_bigwig]
    )
    missing = [path for path in signal_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing signal files: {missing[:5]}")

    _, y = extract_loci(
        loci=loci,
        sequences=FASTA,
        chroms=chroms,
        signals=signal_paths,
        in_window=in_window,
        out_window=out_window,
        verbose=False,
        ignore=IGNORE_BASES,
        exclusion_lists=[BLACKLIST],
    )
    y_abs = abs_numpy(y)

    idx = 0
    a_pl_slice = slice(idx, idx + len(a_pl))
    idx += len(a_pl)
    a_mn_slice = slice(idx, idx + len(a_mn))
    idx += len(a_mn)
    b_pl_slice = slice(idx, idx + len(b_pl))
    idx += len(b_pl)
    b_mn_slice = slice(idx, idx + len(b_mn))
    idx += len(b_mn)
    full_pl_slice = slice(idx, idx + 1)
    idx += 1
    full_mn_slice = slice(idx, idx + 1)

    profile_a = profile_from_channels(y_abs, a_pl_slice, a_mn_slice)
    profile_b = profile_from_channels(y_abs, b_pl_slice, b_mn_slice)
    full_profile = profile_from_channels(y_abs, full_pl_slice, full_mn_slice)
    return profile_a, profile_b, full_profile


def load_loci(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )


def process_experiment(
    exp_id: str,
    exp: dict,
    replicate: dict,
    chrom_splits: dict[int, list[str]],
    raw_bigwig_dir: Path,
    out_dir: Path,
    in_window: int,
    out_window: int,
    save_per_locus: bool,
) -> dict:
    if replicate["status"] != "ok":
        raise RuntimeError(replicate["reason"])

    processed = exp.get("processed", {})
    peaks_path = REPO_ROOT / processed["filtered_peaks"]
    full_pl = str(REPO_ROOT / processed["pl_bigwig"])
    full_mn = str(REPO_ROOT / processed["mn_bigwig"])
    if not peaks_path.exists():
        raise FileNotFoundError(f"filtered peaks not found: {peaks_path}")
    if not Path(full_pl).exists() or not Path(full_mn).exists():
        raise FileNotFoundError(f"processed BigWigs not found: {full_pl}, {full_mn}")

    loci = load_loci(peaks_path)
    group_a = replicate["groups"]["a"]
    group_b = replicate["groups"]["b"]

    per_fold = {}
    per_locus_arrays = {}
    all_pr_div = []
    all_pr_dist = []
    all_avg_div = []
    all_avg_dist = []

    for fold in sorted(chrom_splits):
        profile_a, profile_b, full_profile = extract_fold_profiles(
            loci=loci,
            chroms=chrom_splits[fold],
            group_a=group_a,
            group_b=group_b,
            full_pl_bigwig=full_pl,
            full_mn_bigwig=full_mn,
            raw_bigwig_dir=raw_bigwig_dir,
            in_window=in_window,
            out_window=out_window,
        )

        pr_div, pr_dist = js_divergence_and_distance(profile_a, profile_b)
        avg_div, avg_dist = average_profile_baseline(full_profile)
        all_pr_div.append(pr_div)
        all_pr_dist.append(pr_dist)
        all_avg_div.append(avg_div)
        all_avg_dist.append(avg_dist)

        per_fold[str(fold)] = {
            "replicate_profile_jsd": median_finite(pr_div),
            "replicate_profile_js_distance": median_finite(pr_dist),
            "average_profile_jsd": median_finite(avg_div),
            "average_profile_js_distance": median_finite(avg_dist),
            "n_loci": int(np.isfinite(avg_div).sum()),
            "n_replicate_loci": int(np.isfinite(pr_div).sum()),
            "n_replicate_pairs": int(replicate["n_pairs"]),
        }

        if save_per_locus:
            per_locus_arrays[f"replicate_js_divergence_fold{fold}"] = pr_div
            per_locus_arrays[f"replicate_js_distance_fold{fold}"] = pr_dist
            per_locus_arrays[f"average_profile_js_divergence_fold{fold}"] = avg_div
            per_locus_arrays[f"average_profile_js_distance_fold{fold}"] = avg_dist

    all_pr_div = np.concatenate(all_pr_div)
    all_pr_dist = np.concatenate(all_pr_dist)
    all_avg_div = np.concatenate(all_avg_div)
    all_avg_dist = np.concatenate(all_avg_dist)

    result = {
        "experiment": exp_id,
        "biosample": exp.get("biosample", ""),
        "bounds_type": "profile_jsd",
        "scale": {
            "profile_jsd": "Jensen-Shannon divergence",
            "profile_js_distance": "sqrt(Jensen-Shannon divergence)",
        },
        "paths": {
            "filtered_peaks": str(peaks_path),
            "processed_pl_bigwig": full_pl,
            "processed_mn_bigwig": full_mn,
            "raw_bigwig_dir": str(raw_bigwig_dir),
        },
        "replicates": replicate,
        "per_fold": per_fold,
        "genome_wide": {
            "replicate_profile_jsd": median_finite(all_pr_div),
            "replicate_profile_js_distance": median_finite(all_pr_dist),
            "average_profile_jsd": median_finite(all_avg_div),
            "average_profile_js_distance": median_finite(all_avg_dist),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exp_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Wrote {out_path}")

    if save_per_locus:
        per_locus_dir = out_dir / "per_locus"
        per_locus_dir.mkdir(parents=True, exist_ok=True)
        npz_path = per_locus_dir / f"{exp_id}.npz"
        np.savez_compressed(npz_path, **per_locus_arrays)
        print(f"Wrote {npz_path}")

    return result


def fold_average(data: dict, metric: str) -> float | None:
    values = [
        fold.get(metric)
        for fold in data.get("per_fold", {}).values()
        if fold.get(metric) is not None
    ]
    if not values:
        return None
    return float(np.mean(values))


def consolidate_bounds(out_dir: Path, tsv_path: Path):
    rows = []
    for path in sorted(out_dir.glob("ENCSR*.json")):
        with open(path) as f:
            data = json.load(f)
        rows.append(
            {
                "experiment": data["experiment"],
                "biosample": data.get("biosample", ""),
                "n_folds": len(data.get("per_fold", {})),
                "replicate_profile_jsd_fold_average": fold_average(
                    data, "replicate_profile_jsd"
                ),
                "replicate_profile_js_distance_fold_average": fold_average(
                    data, "replicate_profile_js_distance"
                ),
                "average_profile_jsd_fold_average": fold_average(
                    data, "average_profile_jsd"
                ),
                "average_profile_js_distance_fold_average": fold_average(
                    data, "average_profile_js_distance"
                ),
                "replicate_profile_jsd_genome_wide": data["genome_wide"].get(
                    "replicate_profile_jsd"
                ),
                "replicate_profile_js_distance_genome_wide": data["genome_wide"].get(
                    "replicate_profile_js_distance"
                ),
                "average_profile_jsd_genome_wide": data["genome_wide"].get(
                    "average_profile_jsd"
                ),
                "average_profile_js_distance_genome_wide": data["genome_wide"].get(
                    "average_profile_js_distance"
                ),
            }
        )

    if not rows:
        print(f"No bounds JSON files found in {out_dir}", file=sys.stderr)
        return

    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {tsv_path}")


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.consolidate_only:
        consolidate_bounds(args.out_dir, args.tsv)
        return

    replicates = write_replicate_config(config, args.replicate_config)
    if args.write_replicates_only:
        return

    with open(args.chrom_splits) as f:
        data = yaml.safe_load(f)
    chrom_splits = {int(k): v for k, v in data["folds"].items()}
    experiments = config["experiments"]
    selected = args.experiment or sorted(experiments)

    for exp_id in selected:
        if exp_id not in experiments:
            print(f"WARNING: {exp_id} not found in config, skipping", file=sys.stderr)
            continue
        out_path = args.out_dir / f"{exp_id}.json"
        if args.skip_existing and out_path.exists():
            print(f"Skipping existing {out_path}")
            continue
        try:
            process_experiment(
                exp_id=exp_id,
                exp=experiments[exp_id],
                replicate=replicates["experiments"][exp_id],
                chrom_splits=chrom_splits,
                raw_bigwig_dir=args.raw_bigwig_dir,
                out_dir=args.out_dir,
                in_window=args.in_window,
                out_window=args.out_window,
                save_per_locus=args.save_per_locus,
            )
        except Exception as exc:
            print(f"ERROR: {exp_id}: {exc}", file=sys.stderr)

    consolidate_bounds(args.out_dir, args.tsv)


if __name__ == "__main__":
    main()
