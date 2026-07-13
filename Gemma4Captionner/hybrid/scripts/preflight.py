from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


CHECKS: list[tuple[str, str, str]] = []


def _run(name: str, cmd: list[str], *, env: dict[str, str] | None = None) -> bool:
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    print(f"\n== {name}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=proc_env, text=True)
    status = "pass" if proc.returncode == 0 else "fail"
    CHECKS.append((name, status, f"exit={proc.returncode}"))
    return proc.returncode == 0


def _record(name: str, ok: bool, note: str) -> bool:
    CHECKS.append((name, "pass" if ok else "fail", note))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {note}")
    return ok


def _record_warn(name: str, ok: bool, note: str) -> bool:
    CHECKS.append((name, "pass" if ok else "warn", note))
    print(f"{'PASS' if ok else 'WARN'} {name}: {note}")
    return ok


def _docker_daemon_available() -> bool:
    proc = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        return _record("docker daemon", True, proc.stdout.strip())
    detail = (proc.stderr or proc.stdout).strip().splitlines()
    return _record_warn(
        "docker daemon",
        False,
        detail[0] if detail else "docker daemon unavailable",
    )


def _inspect_image(image: str) -> bool:
    proc = subprocess.run(
        ["docker", "inspect", image, "--format", "architecture={{.Architecture}} os={{.Os}} size={{.Size}}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return _record("docker image inspect", False, detail[0] if detail else "inspect failed")
    out = proc.stdout.strip()
    ok = "architecture=amd64" in out and "os=linux" in out
    return _record("docker image inspect", ok, out)


def _docker_contract_run(image: str) -> bool:
    input_dir = ROOT / "in" / "preflight"
    output_dir = ROOT / "out" / "docker_preflight"
    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "data" / "sample_tasks.json", input_dir / "tasks.json")

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{input_dir.resolve()}:/input:ro",
        "-v",
        f"{output_dir.resolve()}:/output",
        "-e",
        "PER_TASK_TIMEOUT_S=1",
        "-e",
        "FIREWORKS_API_KEY=",
        "-e",
        "GROQ_API_KEY=",
        "-e",
        "OPENROUTER_API_KEY=",
        image,
    ]
    t0 = time.perf_counter()
    ok = _run("docker degraded contract run", cmd)
    elapsed = time.perf_counter() - t0
    if not ok:
        return False

    results = output_dir / "results.json"
    ok = _check_results_contract(results)
    ok &= _run("docker degraded self-check", [PY, "eval/self_check.py", "--results", str(results.relative_to(ROOT))])
    _record("docker degraded elapsed", elapsed < 60, f"{elapsed:.2f}s")
    return ok


def _env_present(name: str) -> bool:
    return _record_warn(f"env {name}", bool(os.environ.get(name)), "set" if os.environ.get(name) else "missing")


def _check_no_secret_literals() -> bool:
    patterns = ("fw_", "gsk_", "sk-", "xoxb-", "ghp_")
    allowed_examples = ("fw_xxx",)
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part in rel for part in ("__pycache__", ".git", "out/")):
            continue
        if rel == ".env" or rel.startswith(".env."):
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".sh", ".json", ".jsonl", ""}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in patterns:
            if pat in text and not any(example in text for example in allowed_examples):
                offenders.append(f"{rel}:{pat}")
                break
    return _record("secret literal scan", not offenders, ", ".join(offenders[:10]) if offenders else "no obvious secret literals")


def _check_results_contract(path: Path) -> bool:
    if not path.exists():
        return _record("degraded results exists", False, f"missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}
    ok = isinstance(data, list) and data and all(
        isinstance(row.get("captions"), dict)
        and required <= set(row["captions"])
        and all(isinstance(v, str) and v.strip() for v in row["captions"].values())
        for row in data
    )
    return _record("degraded results contract", ok, f"{len(data) if isinstance(data, list) else 0} row(s)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Treat missing Docker/API keys as failures")
    parser.add_argument("--docker-build", action="store_true", help="Attempt docker buildx build when daemon is available")
    parser.add_argument("--docker-run", action="store_true", help="Run built image against mounted sample tasks in degraded mode")
    args = parser.parse_args()

    ok = True
    ok &= _run(
        "python compile",
        [
            PY,
            "-m",
            "py_compile",
            "app/main.py",
            "app/pipeline.py",
            "app/models.py",
            "app/prompts.py",
            "app/audio.py",
            "app/frames.py",
            "app/cache.py",
            "eval/local_judge.py",
            "eval/grounding_audit.py",
            "eval/quality_audit.py",
            "eval/self_check.py",
            "scripts/contract_test.py",
            "scripts/quality_gate.py",
            "scripts/mock_run.py",
            "finetune/build_dataset_v2.py",
            "finetune/generate_scenes.py",
            "finetune/train_gemma_lora.py",
            "finetune/augment_dataset.py",
        ],
    )
    ok &= _run("contract test", [PY, "scripts/contract_test.py"])
    ok &= _run("mock run", [PY, "scripts/mock_run.py", "--tasks", "data/sample_tasks.json", "--out", "out/mock_results.json"])
    ok &= _run("mock self-check", [PY, "eval/self_check.py", "--results", "out/mock_results.json"])
    ok &= _run("mock quality audit", [PY, "eval/quality_audit.py", "--results", "out/mock_results.json"])
    ok &= _run("lora dry-run", [PY, "finetune/train_gemma_lora.py", "--dataset", "finetune/dataset_v2.jsonl", "--dry-run"])

    degraded_out = "out/preflight_degraded_results.json"
    ok &= _run(
        "degraded app run",
        [PY, "-m", "app.main"],
        env={
            "INPUT_PATH": "data/sample_tasks.json",
            "OUTPUT_PATH": degraded_out,
            "PER_TASK_TIMEOUT_S": "1",
            "FIREWORKS_API_KEY": "",
            "GROQ_API_KEY": "",
            "OPENROUTER_API_KEY": "",
        },
    )
    ok &= _check_results_contract(ROOT / degraded_out)
    ok &= _run("degraded self-check", [PY, "eval/self_check.py", "--results", degraded_out])
    ok &= _run("degraded quality audit", [PY, "eval/quality_audit.py", "--results", degraded_out])
    ok &= _check_no_secret_literals()

    docker_ok = _docker_daemon_available()
    fw_ok = _env_present("FIREWORKS_API_KEY")
    groq_ok = _env_present("GROQ_API_KEY")

    if args.docker_build and docker_ok:
        image = os.environ.get("IMAGE", "track2-captioner:dev")
        ok &= _run("docker build linux/amd64", ["docker", "buildx", "build", "--platform", "linux/amd64", "--tag", image, "--load", "."])
        ok &= _inspect_image(image)
    elif args.docker_build:
        ok = False

    if args.docker_run and docker_ok:
        image = os.environ.get("IMAGE", "track2-captioner:dev")
        ok &= _inspect_image(image)
        ok &= _docker_contract_run(image)
    elif args.docker_run:
        ok = False

    if args.strict:
        ok &= docker_ok and fw_ok

    print("\n== Summary")
    for name, status, note in CHECKS:
        print(f"{status.upper():4} {name} - {note}")

    if not groq_ok:
        print("NOTE GROQ_API_KEY is optional; audio transcription will be skipped.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
