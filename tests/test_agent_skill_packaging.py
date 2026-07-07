from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path


def _load_package_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "package_portable_benchmark.py"
    spec = importlib.util.spec_from_file_location("package_portable_benchmark_agent", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_zip_includes_agent_skill_and_mcp_files(tmp_path, monkeypatch):
    package_portable_benchmark = _load_package_module()
    output = tmp_path / "portable.zip"
    monkeypatch.setattr(sys, "argv", ["package_portable_benchmark.py", "--output", str(output)])

    assert package_portable_benchmark.main() == 0

    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        skill_text = zf.read("AiReceipes/skills/aireceipes-benchmark-router/SKILL.md").decode("utf-8")

    assert "AiReceipes/src/aireceipes/router.py" in names
    assert "AiReceipes/src/aireceipes/hardware.py" in names
    assert "AiReceipes/src/aireceipes/mcp_server.py" in names
    assert "AiReceipes/mcp/aireceipes-benchmark-router.json" in names
    assert "AiReceipes/skills/aireceipes-benchmark-router/SKILL.md" in names
    assert "launch_benchmark" in skill_text
    assert "plan_route" in skill_text
    assert "MCP" in skill_text
