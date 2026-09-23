#!/bin/bash -l
#SBATCH --job-name=assemble_figure2
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=normal,akundaje,owners
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayhe@stanford.edu
#SBATCH -C NO_GPU
#SBATCH --requeue

# Runs just FIGURE2_HANDOFF.md step 4 (final Figure 2 assembly) as one
# batch job -- everything upstream (steps 1-3, 5) is assumed already done.
# --requeue is safe: plot_figure2.py deterministically overwrites its own
# output every time, so a preemption/requeue just reruns it from scratch.

set -euo pipefail

# ml ucsc-utils is deliberately NOT loaded -- it pulls in a curl/openssl
# combination that conflicts with python/3.12.1's own openssl dependency
# and left the uv-managed venv's python3 unable to find
# libpython3.12.so.1.0 at runtime ("cannot open shared object file").
# plot_figure2.py never shells out to a UCSC binary anyway (pybigtools
# handles bigwig I/O in Python). python/3.12.1 is loaded explicitly rather
# than relying on whatever a given compute node's default module state
# happens to be -- the failure above reproduced with or without
# ucsc-utils, and only stopped once this was loaded explicitly.
ml biology
ml htslib
ml python/3.12.1

mamba activate "${PROCAP_ATLAS_ENV:-procap-atlas}"

# sbatch copies this script into a spool directory and executes the copy,
# so ${BASH_SOURCE[0]} resolves under /var/spool/... at runtime -- deriving
# REPO_ROOT from it silently resolves every relative path against
# /var/spool instead of the repo. SLURM_SUBMIT_DIR is set by Slurm to
# wherever `sbatch` was invoked from, which is the repo root here; fall
# back to BASH_SOURCE only for a plain `bash` test run outside SLURM.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# Fails fast and clearly if the venv's interpreter is broken, rather than
# a confusing failure partway through the real command below.
uv run --project "$REPO_ROOT" --extra sherlock --frozen python -c \
    "import sys; print('interpreter OK:', sys.version)"

uv run --project "$REPO_ROOT" --extra sherlock --frozen python src/analysis/plot_figure2.py \
    --head count --modisco-h5 auto --with-metaplots \
    --n-restricted 14 \
    --profile-exemplars figures/motif_atlas/motif_exemplars_profile_ubiquitous.tsv \
    --profile-h5 motifcompendium/bpnet/motifcompendium_profile_cluster_averages.h5 \
    --profile-names configs/core_promoter_names.tsv --n-profile 3

echo "=== Done ==="
