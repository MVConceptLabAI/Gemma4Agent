import json

from aireceipes.cli import main


def test_cli_list_prints_recipe_table(tmp_path, capsys):
    recipe_dir = tmp_path / "recipes" / "inference"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "echo-smoke.toml").write_text(
        """
id = "inference.echo-smoke"
category = "inference"
name = "Echo smoke"
description = "Smoke test"
tags = ["smoke"]
[classification]
task = "chat-completions"
[runtime]
adapter = "echo"
[dataset]
prompts = ["hello"]
[kpis]
metrics = ["success_rate"]
""".strip(),
        encoding="utf-8",
    )

    exit_code = main(["list", "--recipes-dir", str(tmp_path / "recipes")])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "inference.echo-smoke" in out
    assert "inference" in out
    assert "smoke" in out


def test_cli_run_prints_metrics_path_and_writes_metrics(tmp_path, capsys):
    recipe_dir = tmp_path / "recipes" / "inference"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "echo-smoke.toml").write_text(
        """
id = "inference.echo-smoke"
category = "inference"
name = "Echo smoke"
description = "Smoke test"
tags = ["smoke"]
[classification]
task = "chat-completions"
[runtime]
adapter = "echo"
[dataset]
prompts = ["hello"]
[kpis]
metrics = ["success_rate", "latency_ms_avg"]
""".strip(),
        encoding="utf-8",
    )

    exit_code = main([
        "run",
        "inference.echo-smoke",
        "--recipes-dir",
        str(tmp_path / "recipes"),
        "--output-dir",
        str(tmp_path / "runs"),
    ])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "metrics.json" in out
    metrics_path = next((tmp_path / "runs").glob("*/metrics.json"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["kpis"]["success_rate"] == 1.0
