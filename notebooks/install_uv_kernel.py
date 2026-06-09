"""Install an Open OnDemand-safe Jupyter kernel for the project uv environment."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager


def kernel_spec() -> dict:
    return {
        "argv": [
            "/usr/bin/env",
            "-u",
            "PYTHONPATH",
            str(Path(sys.executable).absolute()),
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "PRO-cap Atlas (uv)",
        "language": "python",
        "metadata": {"debugger": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the PRO-cap Atlas uv kernelspec."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the kernelspec without installing it.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    expected_environment = (repo_root / ".venv").resolve()
    active_environment = Path(sys.prefix).resolve()
    if active_environment != expected_environment:
        raise RuntimeError(
            "Run this installer through the project environment: "
            "uv run --group notebook python notebooks/install_uv_kernel.py"
        )

    spec = kernel_spec()
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return

    with tempfile.TemporaryDirectory() as temporary_directory:
        spec_directory = Path(temporary_directory) / "procap-atlas"
        spec_directory.mkdir()
        (spec_directory / "kernel.json").write_text(
            json.dumps(spec, indent=2) + "\n"
        )
        destination = KernelSpecManager().install_kernel_spec(
            str(spec_directory),
            kernel_name="procap-atlas",
            user=True,
            replace=True,
        )

    print(f"Installed PRO-cap Atlas (uv) kernel at {destination}")


if __name__ == "__main__":
    main()
