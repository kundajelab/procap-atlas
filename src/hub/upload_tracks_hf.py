#!/usr/bin/env python3
"""Upload UCSC track assets to a Hugging Face dataset repo.

Usage:
    python src/hub/upload_tracks_hf.py --dry-run
    python src/hub/upload_tracks_hf.py --include observed --include attributions --include peaks
"""

import argparse
import os
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
DEFAULT_REPO_ID = "adamyhe/procap-atlas-tracks"
DEFAULT_REVISION = "main"
INCLUDES = ("observed", "attributions", "peaks")


def sanitize_name(s):
    return re.sub(r"[^\w-]", "_", s).strip("_")


def hf_resolve_url(repo_id: str, revision: str, path_in_repo: str) -> str:
    return (
        f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path_in_repo}"
    )


def validate_range_url(url: str):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1023"}, method="GET")
    with urllib.request.urlopen(req, timeout=30) as response:
        status = response.status
        accept_ranges = response.headers.get("Accept-Ranges", "")
        content_range = response.headers.get("Content-Range", "")
        if status not in (200, 206):
            raise RuntimeError(f"unexpected HTTP status {status}")
        if status != 206 and not accept_ranges:
            raise RuntimeError(
                "server did not return partial content or Accept-Ranges header"
            )
        return status, accept_ranges, content_range


def collect_uploads(experiments, includes, heads):
    uploads = []
    missing = []

    for exp_id, exp in experiments.items():
        processed = exp.get("processed", {})

        if "observed" in includes:
            for strand, key in [("pl", "pl_bigwig"), ("mn", "mn_bigwig")]:
                local = REPO_ROOT / processed.get(key, "")
                dest = f"observed/{exp_id}_{strand}.bigWig"
                if local.exists():
                    uploads.append((local, dest))
                else:
                    missing.append((local, dest))

        if "peaks" in includes:
            bed_path = processed.get("peaks", "")
            if bed_path:
                local_bed = REPO_ROOT / bed_path
                dest_bed = f"peaks/bed/{Path(bed_path).name}"
                if local_bed.exists():
                    uploads.append((local_bed, dest_bed))
                else:
                    missing.append((local_bed, dest_bed))

                biosample_clean = sanitize_name(exp.get("biosample", "unknown"))
                bb_name = f"{exp_id}_{biosample_clean}_peaks.bb"
                local_bb = local_bed.with_name(bb_name)
                dest_bb = f"peaks/bigbed/{bb_name}"
                if local_bb.exists():
                    uploads.append((local_bb, dest_bb))
                else:
                    missing.append((local_bb, dest_bb))

        if "attributions" in includes:
            for head in heads:
                local = (
                    REPO_ROOT
                    / "attributions"
                    / "bpnet"
                    / "bigwigs"
                    / f"{exp_id}_{head}.bigWig"
                )
                dest = f"attributions/bpnet/{exp_id}_{head}.bigWig"
                if local.exists():
                    uploads.append((local, dest))
                else:
                    missing.append((local, dest))

    return uploads, missing


def link_or_copy(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def stage_upload_tree(uploads, staging_root: Path):
    for local, dest in uploads:
        staged = staging_root / dest
        link_or_copy(local, staged)


def upload_large_folder(api, repo_id, revision, folder_path, num_workers):
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder_path),
        revision=revision,
        num_workers=num_workers,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--include",
        action="append",
        choices=INCLUDES,
        default=None,
        help="track class to upload; repeatable (default: all)",
    )
    parser.add_argument(
        "--head",
        action="append",
        choices=["profile", "count"],
        default=None,
        help="attribution head(s) to upload; repeatable (default: profile count)",
    )
    parser.add_argument(
        "-j",
        "--n-workers",
        type=int,
        default=10,
        help="number of upload_large_folder workers (default: 10)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-url",
        action="store_true",
        help="validate byte-range behavior for the first uploaded/planned URL",
    )
    args = parser.parse_args()

    includes = args.include if args.include is not None else list(INCLUDES)
    heads = args.head if args.head is not None else ["profile", "count"]

    with open(args.config) as f:
        config = yaml.safe_load(f)
    uploads, missing = collect_uploads(config["experiments"], includes, heads)

    print(
        f"Found {len(uploads)} files to upload; {len(missing)} expected files missing"
    )
    if missing:
        print("Missing examples:")
        for local, dest in missing[:10]:
            print(f"  {dest} <- {local}")

    if args.dry_run:
        print("UPLOAD LARGE FOLDER repo root:")
        for local, dest in uploads[:50]:
            print(f"  {local} -> {dest}")
        if len(uploads) > 50:
            print(f"  ... {len(uploads) - 50} more")
    else:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise ImportError("huggingface_hub is required for uploads") from e

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
        if not uploads:
            print("No files to upload")
        else:
            with tempfile.TemporaryDirectory(prefix="procap_atlas_hf_upload_") as tmp:
                staging_root = Path(tmp)
                stage_upload_tree(uploads, staging_root)
                print(
                    f"Uploading large folder with {len(uploads)} files "
                    f"using {args.n_workers} workers"
                )
                upload_large_folder(
                    api, args.repo_id, args.revision, staging_root, args.n_workers
                )

    if args.validate_url:
        if not uploads:
            print("No uploaded/planned files available for URL validation")
            return
        url = hf_resolve_url(args.repo_id, args.revision, uploads[0][1])
        status, accept_ranges, content_range = validate_range_url(url)
        print(
            f"Validated {url}: status={status}, "
            f"Accept-Ranges={accept_ranges or 'NA'}, Content-Range={content_range or 'NA'}"
        )


if __name__ == "__main__":
    main()
