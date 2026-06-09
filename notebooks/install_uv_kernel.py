"""Install an Open OnDemand-safe Jupyter kernel for the project uv environment."""

import argparse
import json
import shlex
import sys
import tempfile
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager


def kernel_spec(launcher: Path) -> dict:
    return {
        "argv": [
            str(launcher),
            "{connection_file}",
        ],
        "display_name": "PRO-cap Atlas (uv)",
        "language": "python",
        "metadata": {"debugger": True},
    }


def launcher_script(repo_root: Path | None = None) -> str:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    executable = shlex.quote(str(Path(sys.executable).absolute()))
    working_directory = shlex.quote(str(repo_root.resolve()))
    return f"""#!/bin/bash
set -u

log_dir="${{XDG_STATE_HOME:-$HOME/.local/state}}/procap-atlas"
mkdir -p "$log_dir"
log_file="$log_dir/kernel.log"

{{
    printf '\\n=== %s ===\\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
    printf 'host: %s\\n' "$(hostname)"
    printf 'working directory: %s\\n' "$PWD"
    printf 'python: %s\\n' {executable}
    printf 'inherited PYTHONPATH: %s\\n' "${{PYTHONPATH-<unset>}}"
}} >>"$log_file" 2>&1

unset PYTHONPATH
export PYTHONNOUSERSITE=1
cd {working_directory}
exec {executable} -m ipykernel_launcher -f "$1" >>"$log_file" 2>&1
"""


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

    if args.dry_run:
        print(
            json.dumps(
                kernel_spec(Path("<installed-kernel>/launch_kernel.sh")),
                indent=2,
            )
        )
        print("\nlaunch_kernel.sh:\n")
        print(launcher_script(), end="")
        return

    with tempfile.TemporaryDirectory() as temporary_directory:
        spec_directory = Path(temporary_directory) / "procap-atlas"
        spec_directory.mkdir()
        launcher = spec_directory / "launch_kernel.sh"
        launcher.write_text(launcher_script())
        launcher.chmod(0o755)
        (spec_directory / "kernel.json").write_text(
            json.dumps(kernel_spec(launcher), indent=2) + "\n"
        )
        destination = KernelSpecManager().install_kernel_spec(
            str(spec_directory),
            kernel_name="procap-atlas",
            user=True,
            replace=True,
        )

    destination = Path(destination)
    installed_launcher = destination / "launch_kernel.sh"
    (destination / "kernel.json").write_text(
        json.dumps(kernel_spec(installed_launcher), indent=2) + "\n"
    )
    installed_launcher.chmod(0o755)
    print(f"Installed PRO-cap Atlas (uv) kernel at {destination}")
    print("Kernel startup log: ~/.local/state/procap-atlas/kernel.log")


if __name__ == "__main__":
    main()
