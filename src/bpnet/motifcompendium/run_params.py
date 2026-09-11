#!/usr/bin/env python3
"""Record the settings a MotifCompendium build ran with, next to its outputs,
and compare two builds' settings after the fact.

cluster_motifs.py's outputs carried no record of how they were produced. That
turned an ordinary question -- "was this compendium clustered at
--across-threshold 0.85 or 0.90?" -- into an unanswerable one, because the
default changed from 0.85 to 0.90 on 2026-08-21 (commit 31d4f24) and nothing
on disk distinguishes the two. The cluster metadata, the MEME export, the
cluster-average h5 and the HTML reports all look identical either way; the
only 0.85/0.90 strings in the reports are JASPAR match scores. File mtimes are
suggestive but not conclusive, since the threshold can be passed explicitly.

This matters beyond bookkeeping. The across-threshold sets how aggressively
variants of one motif merge, so it moves every count derived from the
compendium -- lexicon size, per-cluster prevalence, the number of
tissue-restricted clusters. Comparing a count-head build against a
profile-head build is only a comparison of *heads* if both used the same
thresholds and the same experiment set; otherwise head is confounded with
clustering settings and the comparison means nothing.

So every build now writes `motifcompendium_{head}_parameters.json` beside its
other outputs, recording the thresholds, the experiment selection, the
experiments that actually contributed a MoDISco h5, and the code revision.

Settings-only comparison is the default, because that is the question being
asked: timestamps and completion status differ between any two runs and would
bury the differences that matter.

Usage:
    python src/bpnet/motifcompendium/run_params.py \\
        motifcompendium/bpnet/motifcompendium_count_parameters.json
    python src/bpnet/motifcompendium/run_params.py \\
        motifcompendium/bpnet/motifcompendium_count_parameters.json \\
        motifcompendium/bpnet/motifcompendium_profile_parameters.json
    python src/bpnet/motifcompendium/run_params.py a.json b.json --all
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

PARAMS_TEMPLATE = "motifcompendium_{head}_parameters.json"

# The keys that change what the partition looks like. Everything else in the
# file is provenance, useful to read but not a reason to call two builds
# different.
SETTING_KEYS = (
    "within_threshold",
    "across_threshold",
    "min_reads",
    "blacklist",
    "n_experiments_with_modisco",
    "experiments_with_modisco",
    "jaspar_path",
    "motifcompendium_version",
)


def params_path(out_dir: Path, head: str) -> Path:
    """Where a head's parameter record lives, given its output directory."""
    return Path(out_dir) / PARAMS_TEMPLATE.format(head=head)


def git_commit(repo_root: Path | None = None) -> str | None:
    """Current HEAD, or None if git is unavailable or this is not a checkout.

    Never raises: a build must not die because provenance capture failed.
    """
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            cmd,
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def build_run_params(
    head: str,
    *,
    within_threshold: float,
    across_threshold: float,
    min_reads: float,
    blacklist,
    experiments_selected,
    experiments_with_modisco,
    out_dir,
    jaspar_path=None,
    motifcompendium_version: str | None = None,
    commit: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Assemble the record for one head's build.

    `experiments_selected` is what survived the experiment-level filters;
    `experiments_with_modisco` is what actually contributed a MoDISco h5 and
    therefore what the compendium was really built from. They differ whenever
    a selected experiment has no MoDISco output yet, and it is the second that
    answers "was this the 219-experiment build or the 198?" -- so both are
    recorded rather than just the filter settings that produced them.
    """
    selected = sorted(experiments_selected)
    with_modisco = sorted(experiments_with_modisco)
    started = (now or datetime.now(timezone.utc)).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "head": head,
        "within_threshold": within_threshold,
        "across_threshold": across_threshold,
        "min_reads": min_reads,
        "blacklist": sorted(blacklist),
        "n_experiments_selected": len(selected),
        "experiments_selected": selected,
        "n_experiments_with_modisco": len(with_modisco),
        "experiments_with_modisco": with_modisco,
        "out_dir": str(out_dir),
        "jaspar_path": str(jaspar_path) if jaspar_path is not None else None,
        "motifcompendium_version": motifcompendium_version,
        "commit": commit,
        "started_at": started,
        "completed_at": None,
        "completed": False,
    }


def mark_completed(params: dict, now: datetime | None = None) -> dict:
    """Copy of `params` marked complete, for rewriting once a build finishes.

    The record is written twice -- once before the expensive stages and once
    after -- so a build that died partway through leaves a file saying so
    rather than no file at all. A parameter record that only appeared on
    success would be indistinguishable from a crashed run's missing one, and
    the directory would look like a clean build that was never parameterized.
    """
    out = dict(params)
    out["completed"] = True
    out["completed_at"] = (now or datetime.now(timezone.utc)).isoformat()
    return out


def write_run_params(path: Path, params: dict) -> Path:
    """Write the record, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
    return path


def load_run_params(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compare_run_params(a: dict, b: dict, keys=SETTING_KEYS) -> dict:
    """Keys whose values differ between two records, as {key: (a, b)}.

    A key missing from one record counts as a difference reported against the
    sentinel "<absent>", so an old build compared against a new one shows
    which fields it never captured instead of silently matching.
    """
    diffs = {}
    for key in keys:
        left = a.get(key, "<absent>")
        right = b.get(key, "<absent>")
        if left != right:
            diffs[key] = (left, right)
    return diffs


def _summarize(value, limit: int = 6) -> str:
    """Render a value for the terminal, abbreviating long experiment lists."""
    if isinstance(value, list) and len(value) > limit:
        head = ", ".join(str(v) for v in value[:limit])
        return f"[{head}, ... {len(value)} total]"
    return str(value)


def format_comparison(
    a: dict, b: dict, label_a: str, label_b: str, keys=SETTING_KEYS
) -> str:
    """Human-readable diff, including the set arithmetic for experiment lists.

    For experiment lists a bare "these differ" is useless -- what is wanted is
    which experiments one build has and the other does not, which is exactly
    the 219-vs-198 question.
    """
    diffs = compare_run_params(a, b, keys=keys)
    if not diffs:
        return f"{label_a} and {label_b}: settings identical ({len(keys)} keys checked)"

    lines = [f"{label_a} vs {label_b}: {len(diffs)} setting(s) differ", ""]
    for key, (left, right) in diffs.items():
        lines.append(f"  {key}")
        lines.append(f"    {label_a}: {_summarize(left)}")
        lines.append(f"    {label_b}: {_summarize(right)}")
        if isinstance(left, list) and isinstance(right, list):
            only_a = sorted(set(left) - set(right))
            only_b = sorted(set(right) - set(left))
            if only_a:
                lines.append(f"    only in {label_a} ({len(only_a)}): {_summarize(only_a)}")
            if only_b:
                lines.append(f"    only in {label_b} ({len(only_b)}): {_summarize(only_b)}")
    return "\n".join(lines)


def format_record(params: dict, label: str) -> str:
    """Single-record summary, settings first."""
    lines = [f"{label}:"]
    for key in ("head", "within_threshold", "across_threshold", "min_reads"):
        lines.append(f"  {key}: {params.get(key, '<absent>')}")
    lines.append(
        f"  experiments: {params.get('n_experiments_with_modisco', '<absent>')} "
        f"with modisco of {params.get('n_experiments_selected', '<absent>')} selected"
    )
    for key in ("blacklist", "motifcompendium_version", "commit", "started_at"):
        lines.append(f"  {key}: {_summarize(params.get(key, '<absent>'))}")
    completed = params.get("completed", "<absent>")
    lines.append(f"  completed: {completed} ({params.get('completed_at')})")
    if completed is False:
        lines.append(
            "  WARNING: this build did not finish; its outputs may be partial"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("params", type=Path, nargs="+", metavar="PARAMS_JSON",
                        help="one record to summarize, or two to compare")
    parser.add_argument("--all", action="store_true",
                        help="compare every key, not just the settings that "
                             "change the partition")
    args = parser.parse_args()

    if len(args.params) > 2:
        parser.error("pass one record to summarize or two to compare")

    for path in args.params:
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            print(
                "Builds from before this record existed have no parameters "
                "file; check file mtimes against 2026-08-21 00:58 (commit "
                "31d4f24, when --across-threshold's default moved 0.85 -> "
                "0.90) and rebuild if it matters.",
                file=sys.stderr,
            )
            sys.exit(1)

    records = [load_run_params(p) for p in args.params]
    if len(records) == 1:
        print(format_record(records[0], str(args.params[0])))
        return

    keys = sorted(set(records[0]) | set(records[1])) if args.all else SETTING_KEYS
    print(format_comparison(
        records[0], records[1], str(args.params[0]), str(args.params[1]), keys=keys
    ))


if __name__ == "__main__":
    main()
