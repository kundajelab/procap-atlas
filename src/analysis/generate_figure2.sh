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
# steps 0-4) as one sequential batch job, so a dropped ssh connection or an
# interactive-session timeout can't kill a multi-hour run partway through.
# --requeue matters here because the default --partition includes
# preemptible `owners` -- every step below is safe to rerun from scratch
# (deterministic outputs, no partial-state corruption), so a requeue just
# repeats whatever step was interrupted.
#
# This is a plain mirror of the runbook, not a replacement for it -- keep
# the two in sync. Extended Data/Supplement (FIGURE2_HANDOFF.md step 5) is
# deliberately not included: those are separate, largely independent
# figures (cross-cell-type prediction, JASPAR-name robustness) that don't
# need to block or be blocked by the main Figure 2 assembly.

set -euo pipefail

ml biology
ml htslib
ml ucsc-utils

mamba activate "${PROCAP_ATLAS_ENV:-procap-atlas}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

run() {
    echo "+ $*"
    uv run --project "$REPO_ROOT" --extra sherlock --frozen "$@"
}

echo "=== Step 0: atlas-wide housekeeping ==="
run python src/bpnet/hitcall/cleanup_hitcalls.py
run python src/bpnet/hitcall/cleanup_hitcalls.py --execute --include-abandoned-trim-dirs

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

echo "=== Done ==="
