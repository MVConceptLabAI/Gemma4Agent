---
name: aireceipes-benchmark-router
description: Launch AiReceipes hardware benchmarks via MCP and build route policies that choose the fastest/cheapest acceptable model for a workflow.
---

# AiReceipes benchmark router

Use this skill when an agent needs to:

- detect the local hardware before testing inference backends;
- launch an AiReceipes Gemma/DiffusionGemma benchmark matrix;
- read one or more `comparison.json` benchmark outputs;
- choose the fastest, cheapest, highest-quality, or balanced model route for a workflow such as `agentic`, `chat`, `code`, or `translation`.

## MCP tools

The portable package includes `mcp/aireceipes-benchmark-router.json`, which points an agent harness at:

```bash
python -m aireceipes.mcp_server
```

Available tools:

1. `aireceipes.detect_hardware`
   - Input: `{}`
   - Output: platform, CPU, GPU list, and probe status.

2. `aireceipes.launch_benchmark`
   - Input example:
     ```json
     {
       "hardware": "auto",
       "model": "gemma-4-12b",
       "workflow": "agentic",
       "output_dir": "runs",
       "run_id": "agent-auto",
       "dry_run": true,
       "optimize_for": "balanced"
     }
     ```
   - Use `dry_run: true` first to inspect the generated `scripts/run_gemma_matrix.py` command.
   - Use `dry_run: false` only when the local machine has the required model files and servers configured.

3. `aireceipes.plan_route`
   - Input example:
     ```json
     {
       "comparison": ["runs/20260707T092022Z-gemma-variant-bakeoff/comparison.json"],
       "workflow": "agentic",
       "strategy": "balanced",
       "min_accuracy": 0.8
     }
     ```
   - Output includes `selected`, ordered `candidates`, rejected candidates, and a simple `route_policy` block an agent router can consume.

## CLI equivalents

Build a route policy from benchmark results:

```bash
aireceipes route \
  --comparison runs/latest/comparison.json \
  --workflow agentic \
  --strategy balanced \
  --min-accuracy 0.8 \
  --output runs/latest/route-plan.json
```

Strategies:

- `fastest`: prioritize `effective_tokens_per_sec` after the quality floor.
- `cheapest`: prioritize `routing.cost_per_1m_output_tokens_usd` after the quality floor.
- `quality`: prioritize accuracy/case success.
- `balanced`: prefer acceptable quality, good speed, and low cost.

## Agent workflow

1. Call `aireceipes.detect_hardware`.
2. Call `aireceipes.launch_benchmark` with `dry_run: true`.
3. If the command and paths are valid, run with `dry_run: false` or run `scripts/run_gemma_matrix.py` directly.
4. Pass the resulting `comparison.json` to `aireceipes.plan_route`.
5. Use `route_policy[0].route_to` as the primary local model for that workflow and fall back through `route_policy[0].fallback`.

## Notes

- Route quality is read from the requested vertical if present, otherwise from global KPIs.
- Candidates below `min_accuracy` are rejected even if they are faster or cheaper.
- Local backends default to zero marginal token cost unless a model row contains `routing.cost_per_1m_output_tokens_usd`.
