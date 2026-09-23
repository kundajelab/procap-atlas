#!/bin/bash -l
#SBATCH --job-name=generate_figure2
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=normal,akundaje,owners
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayhe@stanford.edu
#SBATCH -C NO_GPU
#SBATCH --requeue

# Runs the entire Figure 2 pipeline (FIGURE2_HANDOFF.md's "Full run order",
# steps 1-5) as one sequential batch job, so a dropped ssh connection or an
# interactive-session timeout can't kill a multi-hour run partway through.
# --requeue matters here because the default --partition includes
# preemptible `owners` -- every step below is safe to rerun from scratch
# (deterministic outputs, no partial-state corruption), so a requeue just
# repeats whatever step was interrupted.
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

run() {
    echo "+ $*"
    uv run --project "$REPO_ROOT" --extra sherlock --frozen "$@"
}

echo "=== Step 1: atlas-wide QC summary ==="
run python src/bpnet/hitcall/consolidate_motif_reports.py --min-trim-len 6

echo "=== Step 2: Figure 2 main, count head (panels a-c) ==="
run python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2 --sweep

run python src/analysis/motif_group_concentration.py --head count --group-level tissue --save-null-draws
run python src/analysis/motif_group_concentration.py --head count --group-level biosample --save-null-draws

run python src/analysis/select_motif_exemplars.py --head count --max-groups 2 --per-group 3 --with-metaplots \
    --modisco-h5 motifcompendium/bpnet/motifcompendium_count_cluster_averages.h5 \
    --logo-paths motifcompendium/bpnet/motifcompendium_count_cluster_logo_paths.tsv \
    --logo-root motifcompendium/bpnet/

echo "=== Step 3: profile-head core-promoter band ==="
for cid in 4 21 24; do
    for posneg in pos neg; do
        run python src/bpnet/hitcall/metaplot_motif.py --source compendium-seqlets \
            --head profile --compendium-motif-name "${posneg}_patterns.${cid}" -v
    done
done

run python src/analysis/select_motif_exemplars.py --head profile --include-unmatched \
    --broad-groups 1 --top-ubiquitous 999

echo "=== Step 4: assemble Figure 2 ==="
run python src/analysis/plot_figure2.py --head count --modisco-h5 auto --with-metaplots \
    --n-restricted 14 \
    --profile-exemplars figures/motif_atlas/motif_exemplars_profile_ubiquitous.tsv \
    --profile-h5 motifcompendium/bpnet/motifcompendium_profile_cluster_averages.h5 \
    --profile-names configs/core_promoter_names.tsv --n-profile 3

echo "=== Step 5: Extended Data / Supplement ==="
run python src/analysis/plot_motif_rarefaction.py --head count --min-cluster-experiments 2 \
    --collapse-by jaspar_name --sweep --out-dir figures/motif_atlas/collapsed
run python src/analysis/motif_group_concentration.py --head count --group-level tissue \
    --collapse-by jaspar_name --out-dir figures/motif_atlas/collapsed

run python src/analysis/motif_redundancy.py --head count --modisco-h5 auto \
    --trim-threshold 0.5 --drop-untrimmable --report-threshold 1e-6

run python src/analysis/cross_celltype_prediction.py \
    --observed figures/count_correlation_all198/observed_counts.tsv \
    --predicted figures/count_correlation_all198/predicted_counts.tsv \
    --out-dir figures/cross_celltype_all198
run python src/analysis/plot_cross_celltype_figure.py --in-dir figures/cross_celltype_all198

echo "=== Done ==="
