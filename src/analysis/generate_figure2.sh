#!/bin/bash -l
#SBATCH --job-name=generate_figure2
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=normal,akundaje,owners
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayhe@stanford.edu
#SBATCH -C NO_GPU
#SBATCH --requeue

# Runs the entire Figure 2 pipeline (FIGURE2_HANDOFF.md's "Full run order",
# steps 1-5) as one batch job, so a dropped ssh connection or an
# interactive-session timeout can't kill a multi-hour run partway through.
# --requeue matters here because the default --partition includes
# preemptible `owners` -- every step below is safe to rerun from scratch
# (deterministic outputs, no partial-state corruption), so a requeue just
# repeats whatever step was interrupted.
#
# Almost everything here is mutually independent (different output files,
# no shared inputs) except two real dependencies: plot_figure2.py (step 4)
# needs step 2's rarefaction/concentration/count-exemplars outputs and step
# 3's profile-exemplars TSV, and plot_cross_celltype_figure.py needs
# cross_celltype_prediction.py's output. Everything else -- the QC summary,
# the three per-cluster diagnostic metaplots, the JASPAR-name-collapsed
# rarefaction/concentration pair, and compendium redundancy -- runs
# concurrently in the background, each logged to its own file under
# logs/generate_figure2_<job id>/ so interleaved stdout doesn't turn into
# an unreadable mess in the single slurm-%j.out.
#
# This is a plain mirror of the runbook, not a replacement for it -- keep
# the two in sync. Step 0 (atlas-wide housekeeping/cleanup_hitcalls.py) is
# deliberately not included here -- it's a one-off disk-cleanup pass, not
# part of "generate the figure," and is left to be run by hand.

set -euo pipefail

ml biology
ml htslib
ml ucsc-utils

mamba activate "${PROCAP_ATLAS_ENV:-procap-atlas}"

# sbatch copies this script into a spool directory and executes the copy,
# so ${BASH_SOURCE[0]} resolves under /var/spool/... at runtime -- deriving
# REPO_ROOT from it (as this script briefly did) silently resolves every
# relative path against /var/spool instead of the repo. SLURM_SUBMIT_DIR is
# set by Slurm to wherever `sbatch` was invoked from, which is the repo
# root here; fall back to BASH_SOURCE only for a plain `bash` test run
# outside SLURM, where it resolves correctly.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs/generate_figure2_${SLURM_JOB_ID:-$$}"
mkdir -p "$LOG_DIR"
echo "Per-step logs: $LOG_DIR"

pids=()
labels=()

# Backgrounds one command, tagged with a label for logging/error reporting.
# Not subject to `set -e` (bash never applies errexit to a backgrounded
# command) -- failures are instead collected explicitly by wait_all below.
launch() {
    local label="$1"; shift
    echo "+ [$label] (background) $*"
    ( uv run --project "$REPO_ROOT" --extra sherlock --frozen "$@" ) \
        > "$LOG_DIR/$label.log" 2>&1 &
    pids+=("$!")
    labels+=("$label")
}

# Waits for every job launch()ed since the last wait_all call, printing
# pass/fail per label, then aborts the script if anything failed -- the
# background-job equivalent of `set -e` for the parallel sections below.
wait_all() {
    local failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[${labels[$i]}] OK"
        else
            echo "[${labels[$i]}] FAILED -- see $LOG_DIR/${labels[$i]}.log" >&2
            failed=1
        fi
    done
    pids=()
    labels=()
    if [ "$failed" -ne 0 ]; then
        echo "One or more parallel steps failed; aborting." >&2
        exit 1
    fi
}

echo "=== Launching steps 1-3 and the step-5 pieces independent of step 4 ==="

# Step 1: atlas-wide QC summary
launch qc python src/bpnet/hitcall/consolidate_motif_reports.py --min-trim-len 6

# Step 2: Figure 2 main, count head (panels a-c)
launch rarefaction \
    python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2 --sweep
launch concentration_tissue \
    python src/analysis/motif_group_concentration.py --head count --group-level tissue --save-null-draws
launch concentration_biosample \
    python src/analysis/motif_group_concentration.py --head count --group-level biosample --save-null-draws
launch exemplars_count \
    python src/analysis/select_motif_exemplars.py --head count --max-groups 2 --per-group 3 --with-metaplots \
    --modisco-h5 motifcompendium/bpnet/motifcompendium_count_cluster_averages.h5 \
    --logo-paths motifcompendium/bpnet/motifcompendium_count_cluster_logo_paths.tsv \
    --logo-root motifcompendium/bpnet/

# Step 3: profile-head core-promoter band. cluster_final ids are namespaced
# separately per pos_patterns/neg_patterns group in the compendium h5 (a
# "cluster 4" only ever exists on one side) -- confirmed 4/21/24 are all
# pos_patterns-side. metaplot_motif.py's standalone CLI only writes a
# diagnostic PNG to figures/metaplots/ for visual inspection; plot_figure2.py
# calls collect_metaplot() directly in-process and never reads that PNG, so
# this loop has no dependency relationship with step 4 at all.
for cid in 4 21 24; do
    launch "metaplot_${cid}" python src/bpnet/hitcall/metaplot_motif.py --source compendium-seqlets \
        --head profile --compendium-motif-name "pos_patterns.${cid}" -v
done
launch exemplars_profile \
    python src/analysis/select_motif_exemplars.py --head profile --include-unmatched \
    --broad-groups 1 --top-ubiquitous 999

# Step 5 pieces with no relationship to step 4 at all
launch rarefaction_jaspar \
    python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2 \
    --collapse-by jaspar_name --sweep --out-dir figures/motif_atlas/collapsed
launch concentration_jaspar \
    python src/analysis/motif_group_concentration.py --head count --group-level tissue \
    --collapse-by jaspar_name --out-dir figures/motif_atlas/collapsed
launch redundancy \
    python src/analysis/motif_redundancy.py --head count --modisco-h5 auto \
    --trim-threshold 0.5 --drop-untrimmable --report-threshold 1e-6
launch cross_celltype_pred \
    python src/analysis/cross_celltype_prediction.py \
    --observed figures/count_correlation_all198/observed_counts.tsv \
    --predicted figures/count_correlation_all198/predicted_counts.tsv \
    --out-dir figures/cross_celltype_all198

wait_all
echo "=== Steps 1-3 and independent step-5 pieces done ==="

echo "=== Step 4 and the remaining step-5 piece (both now unblocked) ==="
launch figure2 \
    python src/analysis/plot_figure2.py --head count --modisco-h5 auto --with-metaplots \
    --n-restricted 14 \
    --profile-exemplars figures/motif_atlas/motif_exemplars_profile_ubiquitous.tsv \
    --profile-h5 motifcompendium/bpnet/motifcompendium_profile_cluster_averages.h5 \
    --profile-names configs/core_promoter_names.tsv --n-profile 3
launch cross_celltype_fig \
    python src/analysis/plot_cross_celltype_figure.py --in-dir figures/cross_celltype_all198

wait_all
echo "=== Done ==="
