from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .hardware import detect_hardware, launch_hardware_benchmark
from .router import build_route_plan


def get_tool_manifest() -> dict[str, Any]:
    return {
        "name": "aireceipes-benchmark-router",
        "version": "0.1.0",
        "tools": [
            {
                "name": "aireceipes.detect_hardware",
                "description": "Detect host CPU/GPU information before selecting benchmark routes.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "aireceipes.launch_benchmark",
                "description": "Launch or dry-run the AiReceipes Gemma benchmark matrix for the current hardware.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "hardware": {"type": "string", "default": "auto"},
                        "model": {"type": "string", "default": "gemma-4-12b"},
                        "workflow": {"type": "string", "default": "agentic"},
                        "output_dir": {"type": "string", "default": "runs"},
                        "run_id": {"type": ["string", "null"], "default": None},
                        "dry_run": {"type": "boolean", "default": False},
                        "optimize_for": {"type": "string", "enum": ["balanced", "fastest", "cheapest", "quality"], "default": "balanced"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "aireceipes.plan_route",
                "description": "Read one or more comparison.json files and produce an agent route policy for a workflow.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "comparison": {"type": "array", "items": {"type": "string"}},
                        "workflow": {"type": "string", "default": "agentic"},
                        "strategy": {"type": "string", "enum": ["balanced", "fastest", "cheapest", "quality"], "default": "balanced"},
                        "min_accuracy": {"type": "number", "default": 0.0},
                    },
                    "required": ["comparison"],
                    "additionalProperties": False,
                },
            },
        ],
    }


def handle_tool_call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    args = dict(arguments or {})
    try:
        if name == "aireceipes.detect_hardware":
            return {"ok": True, "content": detect_hardware()}
        if name == "aireceipes.launch_benchmark":
            content = launch_hardware_benchmark(
                hardware=str(args.get("hardware", "auto")),
                model=str(args.get("model", "gemma-4-12b")),
                workflow=str(args.get("workflow", "agentic")),
                output_dir=Path(str(args.get("output_dir", "runs"))),
                run_id=args.get("run_id"),
                dry_run=bool(args.get("dry_run", False)),
                optimize_for=str(args.get("optimize_for", "balanced")),
            )
            return {"ok": True, "content": content}
        if name == "aireceipes.plan_route":
            comparisons = args.get("comparison")
            if isinstance(comparisons, str):
                comparisons = [comparisons]
            if not isinstance(comparisons, list) or not comparisons:
                raise ValueError("comparison must be a non-empty string list")
            content = build_route_plan(
                [Path(str(item)) for item in comparisons],
                workflow=str(args.get("workflow", "agentic")),
                strategy=str(args.get("strategy", "balanced")),
                min_accuracy=float(args.get("min_accuracy", 0.0)),
            )
            return {"ok": True, "content": content}
        raise ValueError(f"unknown tool: {name}")
    except Exception as exc:
        return {"ok": False, "error": str(exc), "content": None}


def _serve_stdio() -> int:
    """Tiny JSONL MCP-compatible shim for local agent harnesses.

    This is intentionally dependency-free: each stdin line is a JSON object with either
    {"method":"tools/list"} or {"method":"tools/call","params":{"name":"...","arguments":{...}}}.
    It returns one JSON object per line so external routers can bridge it into full MCP.
    """
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            method = request.get("method")
            if method in {"tools/list", "list_tools"}:
                response = {"ok": True, "result": get_tool_manifest()}
            elif method in {"tools/call", "call_tool"}:
                params = request.get("params") if isinstance(request.get("params"), dict) else {}
                response = handle_tool_call(str(params.get("name")), params.get("arguments") if isinstance(params.get("arguments"), dict) else {})
            else:
                response = {"ok": False, "error": f"unknown method: {method}"}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if argv and argv[0] == "manifest":
        print(json.dumps(get_tool_manifest(), indent=2, sort_keys=True))
        return 0
    return _serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
