from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time


def _run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, errors="replace", stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def _windows_pids(port: str) -> list[str]:
    output = _run(["netstat", "-ano"])
    pattern = re.compile(rf"(?:127\.0\.0\.1|0\.0\.0\.0|\[?::1\]?):{re.escape(port)}\s+\S+\s+LISTENING\s+(\d+)", re.IGNORECASE)
    return sorted(set(pattern.findall(output)))


def _unix_pids(port: str) -> list[str]:
    if shutil.which("lsof"):
        output = _run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"])
        return sorted({line.strip() for line in output.splitlines() if line.strip().isdigit()})
    if shutil.which("fuser"):
        output = _run(["fuser", f"{port}/tcp"])
        return sorted({part for part in output.split() if part.isdigit()})
    return []


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: kill_port.py <port>", file=sys.stderr)
        return 2
    port = sys.argv[1]
    is_windows = os.name == "nt"
    pids = _windows_pids(port) if is_windows else _unix_pids(port)
    if not pids:
        print(f"no listener on port {port}")
        return 0

    for pid in pids:
        print(f"killing PID {pid} on port {port}")
        if is_windows:
            subprocess.run(["taskkill", "/F", "/PID", pid], check=False)
        else:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                continue
    if not is_windows:
        time.sleep(1)
        for pid in _unix_pids(port):
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
