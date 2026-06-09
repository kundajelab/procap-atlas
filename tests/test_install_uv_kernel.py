from pathlib import Path

from notebooks.install_uv_kernel import kernel_spec


def test_kernel_ignores_inherited_pythonpath():
    spec = kernel_spec()
    executable = Path(spec["argv"][3])

    assert executable.parent.absolute() == Path(".venv/bin").absolute()
    assert executable.name.startswith("python")
    assert spec["argv"][:3] == ["/usr/bin/env", "-u", "PYTHONPATH"]
    assert spec["argv"][4] == "-m"
    assert spec["argv"][-1] == "{connection_file}"
    assert "env" not in spec
