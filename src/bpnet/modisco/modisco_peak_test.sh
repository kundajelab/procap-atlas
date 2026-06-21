export NUMBA_NUM_THREADS=16
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
model=ENCSR261KBX
peak=ENCSR220XSM
head=count
time uv run --project "$REPO_ROOT" --frozen --extra bpnet modisco motifs \
    -s ${peak}_ohe.npz \
    -a ${model}_${peak}_${head}.npz \
    -o ${model}_${peak}_${head}.modisco.h5 \
    -n 1000000 -l 50 -w 1000 -v
time uv run --project "$REPO_ROOT" --frozen --extra bpnet modisco report \
    -i ${model}_${peak}_${head}.modisco.h5 \
    -o ${model}_${peak}_${head}.modisco \
    -m ../../data/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt
