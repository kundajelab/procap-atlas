#!/usr/bin/env python3
"""Submit a quick per-GPU-SKU SLURM job to check driver/CUDA compatibility
with the Cherimoya Apptainer image, without running a full training job.

Sherlock does not expose GPU driver or CUDA version as a queryable SLURM
node feature or constraint (only GPU_BRD/GPU_GEN/GPU_SKU/GPU_MEM/GPU_CC), so
there is no way to target a compatible node ahead of time. This submits one
short, minimal-resource job per requested SKU that runs `nvidia-smi` and then
tries to actually move a tensor onto the GPU inside the Apptainer image,
which is the real compatibility test (see
src/cherimoya/apptainer/README.md's "Running" section for background on the
--writable-tmpfs / CUDA driver-compat requirements).

Usage:
    python src/cherimoya/apptainer/check_gpu.py --dry-run
    python src/cherimoya/apptainer/check_gpu.py
    python src/cherimoya/apptainer/check_gpu.py --skus GPU_SKU:L40S GPU_SKU:A100_PCIE
"""

import argparse
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
APPTAINER_IMAGE = Path("/scratch/users/ayhe/apptainer/cherimoya.sif")

DEFAULT_SKUS = [
    "GPU_SKU:A100_PCIE",
    "GPU_SKU:A100_SXM4",
    "GPU_SKU:A40",
    "GPU_SKU:H100_SXM5",
    "GPU_SKU:H200_SXM5",
    "GPU_SKU:L40S",
]

CUDA_CHECK = (
    "import torch; "
    "print('torch:', torch.__version__); "
    "print('cuda available:', torch.cuda.is_available()); "
    "x = torch.zeros(1).cuda(); "
    "print('OK: moved tensor to', x.device)"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
    parser.add_argument(
        "--skus",
        nargs="+",
        default=DEFAULT_SKUS,
        help=f"GPU_SKU constraints to test, one job each (default: {DEFAULT_SKUS})",
    )
    parser.add_argument("--partition", type=str, default="akundaje,owners")
    parser.add_argument("--time", type=str, default="00:05:00")
    parser.add_argument(
        "--apptainer-image",
        type=Path,
        default=APPTAINER_IMAGE,
        help=f"Apptainer image to test (default: {APPTAINER_IMAGE})",
    )
    parser.add_argument(
        "--apptainer-bind",
        action="append",
        default=["/oak/stanford/groups/akundaje/ayhe", "/scratch/users/ayhe"],
        help=(
            "path to bind into the Apptainer container; may be repeated "
            "(default: /oak and /scratch)"
        ),
    )
    args = parser.parse_args()

    log_dir = REPO_ROOT / "logs" / "cherimoya_check_gpu"
    log_dir.mkdir(parents=True, exist_ok=True)

    bind_args = " ".join(f"--bind {shlex.quote(path)}" for path in args.apptainer_bind)

    for sku in args.skus:
        job_name = "cherimoya_check_gpu_" + sku.split(":")[-1]
        apptainer_cmd = (
            f'apptainer exec --nv --writable-tmpfs {bind_args} "$APPTAINER_IMAGE" '
            f'python -c {shlex.quote(CUDA_CHECK)}'
        )

        sbatch_script = textwrap.dedent(f"""\
            #!/bin/bash -l
            #SBATCH --job-name={job_name}
            #SBATCH --ntasks=1
            #SBATCH --ntasks-per-node=1
            #SBATCH --nodes=1
            #SBATCH --gpus=1
            #SBATCH -C {sku}
            #SBATCH --cpus-per-task=1
            #SBATCH --mem=4G
            #SBATCH --partition={args.partition}
            #SBATCH --time={args.time}
            #SBATCH --output={log_dir}/{job_name}.out
            #SBATCH --error={log_dir}/{job_name}.err

            APPTAINER_IMAGE={shlex.quote(str(args.apptainer_image))}

            nvidia-smi
            {apptainer_cmd}
        """)

        if args.dry_run:
            print(f"--- {job_name} ---")
            print(sbatch_script)
            continue

        result = subprocess.run(
            ["sbatch"], input=sbatch_script, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"{job_name}: {result.stdout.strip()}")
        else:
            print(
                f"ERROR submitting {job_name}: {result.stderr.strip()}",
                file=sys.stderr,
            )

    if not args.dry_run:
        print(
            f"\nCheck results in {log_dir}/*.err once each job finishes. "
            "'OK: moved tensor to cuda:0' means that SKU works; a driver "
            "version error means it doesn't."
        )


if __name__ == "__main__":
    main()
