#!/usr/bin/env python3
"""Find and remove/compress the large, unneeded files that
call_hits_bpnet.py and the post-hoc filtering pipeline
(launch_post_hoc_pipeline.py) leave behind under hitcalls/bpnet/.

Three categories, in increasing order of judgment call:

1. Always-safe deletes -- nothing in this codebase reads these by name:
   - hits.tsv: Fi-NeMo's own "complete hit data with all instances" dump.
     hits_unique.tsv (deduplicated by chr/start/motif_name/strand) is the
     canonical file every downstream script actually reads; hits.tsv is
     pure duplicate bloat, generally larger than hits_unique.tsv itself.
   - hits_flank_filtered.tsv / hits_seqlet_filtered.tsv: leftover stage
     names from the hit_flank_similarity / unscoped hit_seqlet_confidence
     experiments that got rejected while root-causing TATA/TA-Inr
     overcalling (see filter_low_confidence_hits.py's module docstring).
     The locked-in pipeline never writes these anymore. Worse than dead
     weight: HITS_FILE_STAGES (call_hits_bpnet.py) still lists them ahead
     of hits_confidence_filtered.tsv/hits_dedensified.tsv for staleness
     detection, so a leftover file here that happens to be *newer* than
     the real current output would make resolve_hits_path silently prefer
     the stale abandoned file instead.
   - regions.tmp.npz: call_hits_bpnet.py's atomic-write temp file for
     regions.npz, left behind only if a job died between writing it and
     renaming it into place. Only ever flagged if a valid regions.npz
     already sits next to it (otherwise it could be an in-progress write).

2. Compress in place (gzip): hits.bed. Same rows as hits_unique.tsv, just
   BED-formatted for genome-browser loading -- nothing in this codebase
   reads it back by exact filename, so gzipping it can't break any
   downstream script's file resolution the way gzipping hits_unique.tsv,
   hits_dedensified.tsv, hits_confidence_filtered.tsv, or hits_filtered.tsv
   would (those are all located by exact name in resolve_hits_path /
   launch_link.py).

3. Opt-in only (--include-abandoned-trim-dirs): call-hits output sitting
   directly in hitcalls/bpnet/{exp}_{head}/ (hits*.tsv, hits.bed,
   peaks_qc.tsv, motif_data.tsv, motif_cwms.npy, parameters.json, report/,
   comparison/) when a sibling trim*-prefixed subdirectory also exists for
   the same experiment/head. That sibling means call_hits_bpnet.py was
   rerun with --cwm-trim-coords/--min-trim-len (the min-length floor added
   to fix over-trimming of short core-promoter motifs like Inr), which is
   what every current launcher (launch_report.py, launch_low_confidence_hits.py,
   launch_link.py, ...) selects via --min-trim-len. The base-level files
   predate that switch. regions.npz/peaks.narrowPeak are NEVER included in
   this category even here -- they're trim-independent and reused by every
   trim configuration for that (experiment, head), including the current
   one (see call_hits_bpnet.py's own comment above its regions_npz cache
   check).

Never touched, under any flag: regions.npz, peaks.narrowPeak, hits_unique.tsv,
hits_dedensified.tsv, hits_confidence_filtered.tsv, hits_filtered.tsv,
hits_linked.tsv, and anything not matched by the categories above.

Defaults to a dry-run report (sizes per category, nothing modified).
Pass --execute to actually delete/compress category 1/2 files, and
additionally --include-abandoned-trim-dirs to also clean up category 3.
--min-age-hours (default 24) skips anything modified more recently than
that, as a guard against touching a still-running job's output.

Usage:
    python src/bpnet/hitcall/cleanup_hitcalls.py                     # dry-run report
    python src/bpnet/hitcall/cleanup_hitcalls.py --execute            # delete/compress categories 1-2
    python src/bpnet/hitcall/cleanup_hitcalls.py --execute --include-abandoned-trim-dirs
    python src/bpnet/hitcall/cleanup_hitcalls.py --min-age-hours 48 --execute
"""

import argparse
import gzip
import shutil
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
HITCALLS_BPNET_DIR = REPO_ROOT / "hitcalls" / "bpnet"

ALWAYS_SAFE_DELETE_NAMES = {
    "hits.tsv",
    "hits_flank_filtered.tsv",
    "hits_seqlet_filtered.tsv",
}
COMPRESS_NAMES = {"hits.bed"}
PROTECTED_BASE_DIR_NAMES = {"peaks.narrowPeak", "regions.npz", "regions.tmp.npz"}
NEVER_TOUCH_NAMES = {
    "hits_unique.tsv",
    "hits_dedensified.tsv",
    "hits_confidence_filtered.tsv",
    "hits_filtered.tsv",
    "hits_linked.tsv",
} | PROTECTED_BASE_DIR_NAMES


def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def is_old_enough(path, min_age_hours):
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours >= min_age_hours


def find_stale_regions_tmp(exp_head_dir, min_age_hours):
    tmp_path = exp_head_dir / "regions.tmp.npz"
    regions_path = exp_head_dir / "regions.npz"
    if not tmp_path.exists():
        return None
    if not (regions_path.exists() and zipfile.is_zipfile(regions_path)):
        # No valid regions.npz yet -- this could be an in-progress write,
        # never touch it.
        return None
    if not is_old_enough(tmp_path, min_age_hours):
        return None
    return tmp_path


def find_always_safe_deletes(exp_head_dir, min_age_hours):
    found = []
    for path in exp_head_dir.rglob("*"):
        if path.is_file() and path.name in ALWAYS_SAFE_DELETE_NAMES and is_old_enough(path, min_age_hours):
            found.append(path)
    stale_tmp = find_stale_regions_tmp(exp_head_dir, min_age_hours)
    if stale_tmp is not None:
        found.append(stale_tmp)
    return found


def find_compress_candidates(exp_head_dir, min_age_hours):
    found = []
    for path in exp_head_dir.rglob("*"):
        if path.is_file() and path.name in COMPRESS_NAMES and is_old_enough(path, min_age_hours):
            found.append(path)
    return found


def find_abandoned_trim_dirs(exp_head_dir, min_age_hours):
    """Files sitting directly in exp_head_dir (not in a nested trim*
    subdirectory) when a sibling trim*-prefixed subdirectory exists,
    excluding the trim-independent regions.npz/peaks.narrowPeak cache.
    """
    has_trim_sibling = any(
        child.is_dir() and child.name.startswith("trim") for child in exp_head_dir.iterdir()
    )
    if not has_trim_sibling:
        return []
    found = []
    for path in exp_head_dir.iterdir():
        if path.is_dir() and path.name.startswith("trim"):
            continue
        if path.name in PROTECTED_BASE_DIR_NAMES:
            continue
        if path.is_file():
            if is_old_enough(path, min_age_hours):
                found.append(path)
        elif path.is_dir():
            # report/, comparison/, or any other base-level directory --
            # age-gate on the directory's own mtime (crude but simple: a
            # directory only gets a newer mtime when a file is added or
            # removed from it directly, not on nested writes, but these
            # are only ever fully written once so that's fine here).
            if is_old_enough(path, min_age_hours):
                found.extend(f for f in path.rglob("*") if f.is_file())

    return found


def total_size(paths):
    return sum(p.stat().st_size for p in paths if p.exists())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="actually delete/compress category 1/2 files (default: dry-run report only)",
    )
    parser.add_argument(
        "--include-abandoned-trim-dirs", action="store_true",
        help=(
            "also delete category 3: base-level call-hits output superseded "
            "by a sibling trim*-prefixed subdirectory. Requires --execute to "
            "actually delete; without --execute, still reported."
        ),
    )
    parser.add_argument(
        "--min-age-hours", type=float, default=24.0,
        help="skip anything modified more recently than this many hours ago (default: 24)",
    )
    args = parser.parse_args()

    if not HITCALLS_BPNET_DIR.exists():
        print(f"Error: {HITCALLS_BPNET_DIR} not found", file=sys.stderr)
        sys.exit(1)

    exp_head_dirs = sorted(d for d in HITCALLS_BPNET_DIR.iterdir() if d.is_dir())

    always_safe, compress, abandoned = [], [], []
    for exp_head_dir in exp_head_dirs:
        always_safe.extend(find_always_safe_deletes(exp_head_dir, args.min_age_hours))
        compress.extend(find_compress_candidates(exp_head_dir, args.min_age_hours))
        abandoned.extend(find_abandoned_trim_dirs(exp_head_dir, args.min_age_hours))

    print(f"Scanned {len(exp_head_dirs)} experiment/head directories under {HITCALLS_BPNET_DIR}\n")

    print(f"[1] Always-safe deletes: {len(always_safe)} file(s), {human_bytes(total_size(always_safe))}")
    for p in always_safe:
        print(f"      {p.relative_to(REPO_ROOT)}  ({human_bytes(p.stat().st_size)})")

    print(f"\n[2] Compress candidates (gzip in place): {len(compress)} file(s), {human_bytes(total_size(compress))}")
    for p in compress:
        print(f"      {p.relative_to(REPO_ROOT)}  ({human_bytes(p.stat().st_size)})")

    label = "included" if args.include_abandoned_trim_dirs else "reported only, use --include-abandoned-trim-dirs to act on these"
    print(f"\n[3] Abandoned no-min-trim-len files ({label}): {len(abandoned)} file(s), {human_bytes(total_size(abandoned))}")
    for p in abandoned:
        print(f"      {p.relative_to(REPO_ROOT)}  ({human_bytes(p.stat().st_size)})")

    if not args.execute:
        print("\nDry-run only (pass --execute to actually delete/compress). Nothing was modified.")
        return

    for p in always_safe:
        p.unlink()
    print(f"\nDeleted {len(always_safe)} always-safe file(s).")

    for p in compress:
        gz_path = p.with_name(p.name + ".gz")
        with open(p, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        p.unlink()
    print(f"Compressed {len(compress)} file(s).")

    if args.include_abandoned_trim_dirs:
        for p in abandoned:
            if p.exists():
                p.unlink()
        # Clean up now-empty directories (report/, comparison/, ...) left
        # behind after their files were removed.
        for exp_head_dir in exp_head_dirs:
            for child in sorted(exp_head_dir.iterdir(), reverse=True):
                if child.is_dir() and not child.name.startswith("trim") and not any(child.rglob("*")):
                    child.rmdir()
        print(f"Deleted {len(abandoned)} abandoned no-min-trim-len file(s).")


if __name__ == "__main__":
    main()
