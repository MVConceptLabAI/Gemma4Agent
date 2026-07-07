from __future__ import annotations

import importlib.util
import stat
import sys
import zipfile
from pathlib import Path


def _load_package_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "package_portable_benchmark.py"
    spec = importlib.util.spec_from_file_location("package_portable_benchmark", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_marks_shell_scripts_executable(tmp_path, monkeypatch):
    package_portable_benchmark = _load_package_module()
    output = tmp_path / "portable.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        ["package_portable_benchmark.py", "--output", str(output)],
    )

    assert package_portable_benchmark.main() == 0

    with zipfile.ZipFile(output) as zf:
        info = zf.getinfo("AiReceipes/scripts/wrap_each_test.sh")

    mode = (info.external_attr >> 16) & 0o777
    assert mode & stat.S_IXUSR
    assert mode & stat.S_IXGRP
    assert mode & stat.S_IXOTH
