from pathlib import Path

from notebooks.install_uv_kernel import kernel_spec


def test_kernel_ignores_inherited_pythonpath():
    spec = kernel_spec()
    executable = Path(spec["argv"][0])

    assert executable.parent.absolute() == Path(".venv/bin").absolute()
    assert executable.name.startswith("python")
    assert spec["argv"][1:3] == ["-E", "-m"]
    assert spec["argv"][-1] == "{connection_file}"
    assert "env" not in spec
