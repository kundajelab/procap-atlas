#!/usr/bin/env python3
"""Upload TF-MoDISco and MotifCompendium results to Hugging Face.

Uploads modisco .h5 files and pre-tarred .modisco.tar report archives to
the procap-atlas-motifs dataset repo. Optionally uploads motifcompendium
outputs (cluster averages, metadata, pattern-to-cluster mappings, logos).

Expects .modisco.tar files to already exist alongside .modisco.h5 in
modisco/bpnet/. Tar the report directories first:

    cd modisco/bpnet && for d in *.modisco; do tar cf "$d.tar" "$d"; done

Run on Sherlock where these directories exist.

Usage:
    python src/bpnet/modisco/upload_modisco_hf.py --dry-run
    python src/bpnet/modisco/upload_modisco_hf.py --include modisco
    python src/bpnet/modisco/upload_modisco_hf.py --include modisco --include motifcompendium
    python src/bpnet/modisco/upload_modisco_hf.py --head count
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
DEFAULT_REPO_ID = "adamyhe/procap-atlas-motifs"
MODISCO_DIR = REPO_ROOT / "modisco" / "bpnet"
COMPENDIUM_DIR = REPO_ROOT / "motifcompendium" / "bpnet"
INCLUDES = ("modisco", "motifcompendium")
HEADS = ("profile", "count")


def load_experiments(config_path: Path) -> list[str]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return sorted(config["experiments"].keys())


def collect_modisco_uploads(
    experiments: list[str], heads: list[str],
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Collect modisco h5 and pre-tarred report archives for upload."""
    uploads = []
    missing = []

    for exp_id in experiments:
        for head in heads:
            h5_name = f"{exp_id}_{head}.modisco.h5"
            h5_path = MODISCO_DIR / h5_name
            if h5_path.exists():
                uploads.append((h5_path, f"modisco/{h5_name}"))
            else:
                missing.append(f"modisco/{h5_name}")

            tar_name = f"{exp_id}_{head}.modisco.tar"
            tar_path = MODISCO_DIR / tar_name
            if tar_path.exists():
                uploads.append((tar_path, f"modisco/{tar_name}"))
            else:
                missing.append(f"modisco/{tar_name}")

    return uploads, missing


def collect_compendium_uploads(
    heads: list[str],
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Collect motifcompendium outputs for upload."""
    uploads = []
    missing = []

    upload_patterns = [
        "cluster_averages.h5",
        "cluster_averages.meme",
        "pattern_to_cluster.tsv",
        "cluster_metadata.tsv",
        "cluster_logo_paths.tsv",
        "cluster_report.html",
        "cluster_summary.html",
    ]

    for head in heads:
        prefix = f"motifcompendium_{head}"
        for pattern in upload_patterns:
            fname = f"{prefix}_{pattern}"
            fpath = COMPENDIUM_DIR / fname
            if fpath.exists():
                uploads.append((fpath, f"motifcompendium/{fname}"))
            else:
                missing.append(f"motifcompendium/{fname}")

        logo_dir = COMPENDIUM_DIR / f"{prefix}_cluster_logos"
        if logo_dir.is_dir():
            for svg in sorted(logo_dir.rglob("*.svg")):
                rel = svg.relative_to(COMPENDIUM_DIR)
                uploads.append((svg, f"motifcompendium/{rel}"))

    return uploads, missing


def stage_uploads(uploads: list[tuple[Path, str]], staging_root: Path):
    """Symlink or copy files into a staging tree for upload_large_folder."""
    for local, dest in uploads:
        staged = staging_root / dest
        staged.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(local, staged)
        except OSError:
            shutil.copy2(local, staged)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--include",
        action="append",
        choices=INCLUDES,
        default=None,
        help="what to upload; repeatable (default: all)",
    )
    parser.add_argument(
        "--head",
        action="append",
        choices=HEADS,
        default=None,
        help="attribution head(s); repeatable (default: profile count)",
    )
    parser.add_argument(
        "-j", "--n-workers", type=int, default=10, help="upload workers (default: 10)"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    includes = args.include if args.include is not None else list(INCLUDES)
    heads = args.head if args.head is not None else list(HEADS)
    experiments = load_experiments(args.config)

    all_uploads = []
    all_missing = []

    if "modisco" in includes:
        uploads, missing = collect_modisco_uploads(experiments, heads)
        all_uploads.extend(uploads)
        all_missing.extend(missing)

    if "motifcompendium" in includes:
        uploads, missing = collect_compendium_uploads(heads)
        all_uploads.extend(uploads)
        all_missing.extend(missing)

    print(
        f"Found {len(all_uploads)} files to upload; "
        f"{len(all_missing)} expected files missing"
    )
    if all_missing:
        print(f"Missing examples (first 20 of {len(all_missing)}):")
        for m in all_missing[:20]:
            print(f"  {m}")

    if args.dry_run:
        print(f"\nDRY RUN — would upload to {args.repo_id}:")
        for local, dest in all_uploads[:50]:
            size_mb = local.stat().st_size / 1e6 if local.exists() else 0
            print(f"  {dest} ({size_mb:.1f} MB)")
        if len(all_uploads) > 50:
            print(f"  ... {len(all_uploads) - 50} more")
        total_mb = sum(
            f.stat().st_size for f, _ in all_uploads if f.exists()
        ) / 1e6
        print(f"\nTotal: {total_mb:.1f} MB across {len(all_uploads)} files")
    else:
        if not all_uploads:
            print("No files to upload")
            return

        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id, repo_type="dataset", exist_ok=True
        )

        with tempfile.TemporaryDirectory(
            prefix="modisco_hf_upload_"
        ) as upload_tmp:
            staging_root = Path(upload_tmp)
            stage_uploads(all_uploads, staging_root)
            print(
                f"Uploading {len(all_uploads)} files "
                f"using {args.n_workers} workers"
            )
            api.upload_large_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=str(staging_root),
                revision=args.revision,
                num_workers=args.n_workers,
            )
        print("Upload complete")


if __name__ == "__main__":
    main()
