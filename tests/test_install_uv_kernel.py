from pathlib import Path

from notebooks.install_uv_kernel import kernel_spec, launcher_script


def test_kernel_uses_installed_launcher():
    launcher = Path("/tmp/procap-atlas/launch_kernel.sh")
    spec = kernel_spec(launcher)

    assert spec["argv"] == [str(launcher), "{connection_file}"]
    assert "env" not in spec


def test_launcher_isolates_python_and_logs_startup():
    script = launcher_script()

    assert "unset PYTHONPATH" in script
    assert "export PYTHONNOUSERSITE=1" in script
    assert ".venv/bin/python" in script
    assert "-m ipykernel_launcher -f \"$1\"" in script
    assert ".local/state" in script
    assert "kernel.log" in script
