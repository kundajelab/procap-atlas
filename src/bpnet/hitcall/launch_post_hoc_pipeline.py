#!/usr/bin/env python3
"""Submit one SLURM job per (experiment, head) pair that runs the entire
post-hoc, CPU-only Fi-NeMo filtering/reporting pipeline sequentially, so
there's no need to submit each stage separately and wait for the previous
one to finish across the whole atlas before starting the next.

Consolidates launch_filter_repeat_density.py, launch_report.py (run
twice), and launch_low_confidence_hits.py into one job per experiment.
All three underlying scripts are already CPU-only (`#SBATCH -C NO_GPU` in
each of their own launchers), individually fast, and -- critically --
fully self-contained per experiment (each one only ever reads/writes that
one experiment's own hits.tsv/regions.npz/modisco.h5), so running them
sequentially inside a single job is simpler and more robust than SLURM
`--dependency` chaining across separate per-stage launchers, at the cost
of some parallelism (a slow experiment in one stage blocks that same
job's later stages, but not other experiments' jobs).

Deliberately excludes launch_link.py/link_hits_to_compendium.py: unlike
the four stages above, it depends on the atlas-wide MotifCompendium
cluster-average h5 (built separately by motifcompendium/cluster_motifs.py,
which aggregates motifs across *every* experiment in the atlas, not just
this one). That compendium has to be built/updated once, after all (or
enough) experiments' own MoDISco/hit-calling is done -- it isn't safe to
fold into each experiment's own independent, freely-parallel job, since an
experiment processed early could link against a stale or incomplete
compendium. Run launch_link.py as its own separate, later, atlas-scope
step once the compendium is up to date.

Order within each job, matching the real dependency chain established
while root-causing TATA/TA-Inr overcalling (see filter_low_confidence_hits.py's
module docstring and src/bpnet/README.md for the full investigation):

1. filter_repeat_density.py -- drops dense same-motif repeat clusters.
2. report_bpnet.py (baseline pass) -- REQUIRED before step 3, even though
   its own output gets overwritten by step 4: --seqlet-low-similarity-only
   (the default --low-confidence-args below) reads *this* pass's
   report/motif_report.tsv to decide which motifs are already failing
   cwm_similarity QC, so it must reflect hits from before the corroboration
   filter runs, not after.
3. filter_low_confidence_hits.py -- hit_seqlet_confidence, scoped to only
   the motifs step 2 flagged as failing (the locked-in configuration; see
   --low-confidence-args below to override).
4. report_bpnet.py (final pass) -- now prefers hits_confidence_filtered.tsv
   from step 3, and defaults to --cwm-similarity-threshold 0.8 (not 0.9) to
   retain the substantially-improved-but-not-quite-0.9 core-promoter
   motifs from step 3 instead of dropping them wholesale.

`set -e` in the generated sbatch script aborts the whole job if any stage
fails, rather than continuing on to a later stage against a broken/missing
input (e.g. running the corroboration filter's --seqlet-low-similarity-only
against a baseline report that never actually ran).

Jobs are submitted with --requeue, matching src/bpnet/fit/launch.py's
reasoning: the default --partition includes `owners`, which is preemptible
(`normal`/`akundaje`/`gpu` are not), and without --requeue a preempted job
just dies with no automatic resubmission. A requeued job re-runs this whole
script from
scratch rather than resuming mid-chain -- there's no per-stage skip logic
inside the job itself (see below), so this is safe: every stage overwrites
its own output deterministically from the same inputs, so redoing an
already-succeeded earlier stage on retry can't corrupt anything, just costs
some wasted recompute. check_post_hoc_pipeline_failures.py scans for jobs
that started but never reached this genuinely-complete state (including
non-preemption failures --requeue doesn't help with, e.g. a real bug or
bad data in one of the four scripts) without you having to check SLURM
job states or `.err` logs by hand -- text-scanning stderr for a generic
"did this fail" signal isn't reliable here the way it is for e.g.
modisco/relaunch_timeout.py's specific TIME_LIMIT text match, since this
pipeline's own dependencies (finemo, numpy, matplotlib) routinely print
non-fatal warnings to stderr even on a fully successful run.

Resource defaults are a rough sum across all four stages' own launcher
defaults (repeat-density 30min/16G/1cpu, report 2h/64G/4cpu x2,
low-confidence 30min/16G/1cpu) -- hit_seqlet_confidence specifically is
noted as more compute-intensive than filter_low_confidence_hits.py's other
--score-column options (a full, single-threaded tangermeme.seqlet.
recursive_seqlets pass per region, no GPU/parallelism support), so this is
a conservative starting budget; check actual wall-clock on a real
experiment and adjust --time if needed before running atlas-wide.

Jobs are skipped if hits_filtered.tsv already exists (fully done -- the
final report_bpnet.py pass's own output) or if hits_unique.tsv is missing
(run call_hits_bpnet.py/hitcall/launch.py first) -- unless --force is
given, which resubmits already-complete experiments too (e.g. after
changing --low-confidence-args/--report-args or the underlying scripts
and wanting to reprocess the whole atlas with the new settings; missing
hits_unique.tsv is still skipped even with --force, since there's nothing
to reprocess). Unlike the per-stage launchers, there's no per-stage skip
logic inside the job itself -- each
underlying script is safe to rerun and overwrites its own output
unconditionally, and the stages are fast enough that redundant
recomputation within an already-partially-done experiment isn't worth the
added complexity.

Usage:
    python src/bpnet/hitcall/launch_post_hoc_pipeline.py                    # submit all experiments, profile head
    python src/bpnet/hitcall/launch_post_hoc_pipeline.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/hitcall/launch_post_hoc_pipeline.py --head profile --head count
    python src/bpnet/hitcall/launch_post_hoc_pipeline.py --min-reads 20000000
    python src/bpnet/hitcall/launch_post_hoc_pipeline.py --min-trim-len 6  # match hitcall/launch.py's floor
    python src/bpnet/hitcall/launch_post_hoc_pipeline.py --low-confidence-args '--score-column hit_seqlet_confidence --seqlet-low-similarity-only --seqlet-similarity-threshold 0.85'
    python src/bpnet/hitcall/launch_post_hoc_pipeline.py --report-args '--cwm-similarity-threshold 0.85'
    python src/bpnet/hitcall/launch_post_hoc_pipeline.py --force  # reprocess every experiment, even already-complete ones
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, trim_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
HITCALL_DIR = REPO_ROOT / "src" / "bpnet" / "hitcall"
REPEAT_DENSITY_SCRIPT = HITCALL_DIR / "filter_repeat_density.py"
LOW_CONFIDENCE_SCRIPT = HITCALL_DIR / "filter_low_confidence_hits.py"
REPORT_SCRIPT = HITCALL_DIR / "report_bpnet.py"

DEFAULT_LOW_CONFIDENCE_ARGS = "--score-column hit_seqlet_confidence --seqlet-low-similarity-only"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
    parser.add_argument(
        "--head",
        type=str,
        action="append",
        choices=["profile", "count"],
        default=None,
        metavar="HEAD",
        help="attribution/motif head(s) to run; repeatable (default: profile)",
    )
    # SLURM resource flags -- see module docstring for how these defaults
    # were chosen (a sum across all four stages' own individual defaults).
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="8:00:00")
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "resubmit every experiment even if it already looks fully "
            "complete (e.g. after changing --low-confidence-args/"
            "--report-args/the underlying scripts themselves and wanting "
            "to reprocess the whole atlas with the new settings). Still "
            "skips experiments missing hits_unique.tsv -- there's nothing "
            "to reprocess for those regardless."
        ),
    )
    parser.add_argument(
        "--repeat-density-args",
        type=str,
        default="",
        help="extra arguments forwarded to filter_repeat_density.py (e.g. '--min-cluster-hits 4')",
    )
    parser.add_argument(
        "--low-confidence-args",
        type=str,
        default=DEFAULT_LOW_CONFIDENCE_ARGS,
        help=(
            "extra arguments forwarded to filter_low_confidence_hits.py "
            f"(default: {DEFAULT_LOW_CONFIDENCE_ARGS!r}, the locked-in "
            "configuration -- see that script's module docstring)"
        ),
    )
    parser.add_argument(
        "--report-args",
        type=str,
        default="",
        help=(
            "extra arguments forwarded to BOTH report_bpnet.py passes "
            "(e.g. '--cwm-similarity-threshold 0.85'); the same value is "
            "used for the baseline and final pass, since there's no reason "
            "for the QC threshold itself to differ between them"
        ),
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
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile"]

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))

    hitcalls_dir = REPO_ROOT / "hitcalls" / "bpnet"
    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    log_dir = REPO_ROOT / "logs" / "bpnet_hitcall_post_hoc_pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_missing = 0
    skipped_reads = 0
    for exp_id in experiments:
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        model_dir_name = exp_id

        for head in heads:
            cwm_trim_coords = (
                modisco_dir
                / f"{exp_id}_{head}_trim_coords_min{args.min_trim_len}bp.tsv"
                if args.min_trim_len is not None
                else None
            )
            suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, cwm_trim_coords)
            exp_dir = hitcalls_dir / f"{model_dir_name}_{head}"
            hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir

            hits_unique = hits_dir / "hits_unique.tsv"
            if not hits_unique.exists():
                skipped_missing += 1
                continue

            # hits_filtered.tsv is written by BOTH report_bpnet.py passes
            # (baseline and final) under the same filename, so its mere
            # existence doesn't prove the final pass (step 4) ran -- a job
            # that died right after the baseline pass would leave exactly
            # this file behind too. Require it to be at least as new as
            # hits_confidence_filtered.tsv (step 3's output, which can only
            # exist once step 3 has actually run) to confirm the intended
            # order (2 -> 3 -> 4) really completed, not just step 2 alone.
            hits_filtered = hits_dir / "hits_filtered.tsv"
            hits_confidence_filtered = hits_dir / "hits_confidence_filtered.tsv"
            already_done = (
                hits_filtered.exists()
                and hits_confidence_filtered.exists()
                and hits_filtered.stat().st_mtime >= hits_confidence_filtered.stat().st_mtime
            )
            if already_done and not args.force:
                skipped_done += 1
                continue

            job_name = f"bpnet_hitcall_post_hoc_pipeline_{exp_id}_{head}{suffix}"
            run_prefix = f"uv run --project {REPO_ROOT} --extra sherlock --frozen python"
            min_trim_flag = f" --min-trim-len {args.min_trim_len}" if args.min_trim_len is not None else ""
            cwm_trim_coords_flag = f" --cwm-trim-coords {cwm_trim_coords}" if cwm_trim_coords is not None else ""

            repeat_density_cmd = (
                f"{run_prefix} {REPEAT_DENSITY_SCRIPT} -e {exp_id} --head {head} -v"
                f"{min_trim_flag} {args.repeat_density_args}"
            )
            report_cmd = (
                f"{run_prefix} {REPORT_SCRIPT} -e {exp_id} --head {head} -v"
                f"{cwm_trim_coords_flag} {args.report_args}"
            )
            low_confidence_cmd = (
                f"{run_prefix} {LOW_CONFIDENCE_SCRIPT} -e {exp_id} --head {head} -v"
                f"{min_trim_flag} {args.low_confidence_args}"
            )

            sbatch_script = textwrap.dedent(f"""\
                #!/bin/bash -l
                #SBATCH --job-name={job_name}
                #SBATCH --ntasks=1
                #SBATCH --ntasks-per-node=1
                #SBATCH --nodes=1
                #SBATCH --cpus-per-task={args.cpus_per_task}
                #SBATCH --mem={args.mem}
                #SBATCH --partition={args.partition}
                #SBATCH --time={args.time}
                #SBATCH --output={log_dir}/{job_name}.out
                #SBATCH --error={log_dir}/{job_name}.err
                #SBATCH -C NO_GPU
                #SBATCH --requeue

                set -e

                ml biology
                ml htslib

                mamba activate "${{PROCAP_ATLAS_ENV:-procap-atlas}}"

                echo "=== [1/4] filter_repeat_density.py ==="
                {repeat_density_cmd}

                echo "=== [2/4] report_bpnet.py (baseline pass, for --seqlet-low-similarity-only scoping) ==="
                {report_cmd}

                echo "=== [3/4] filter_low_confidence_hits.py ==="
                {low_confidence_cmd}

                echo "=== [4/4] report_bpnet.py (final pass) ==="
                {report_cmd}
            """)

            if args.dry_run:
                print(f"--- {job_name} ---")
                print(sbatch_script)
                submitted += 1
                continue

            result = subprocess.run(
                ["sbatch"], input=sbatch_script, capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"{job_name}: {result.stdout.strip()}")
                submitted += 1
            else:
                print(
                    f"ERROR submitting {job_name}: {result.stderr.strip()}",
                    file=sys.stderr,
                )

    action = "Would submit" if args.dry_run else "Submitted"
    total = len(experiments) * len(heads)
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_reads} experiments "
        f"with <{args.min_reads:,} reads, skipped {skipped_missing} missing "
        f"hits_unique.tsv, skipped {skipped_done} already fully "
        f"pipelined ({total} total)"
    )


if __name__ == "__main__":
    main()
