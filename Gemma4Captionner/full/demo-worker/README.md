# Gemma 4 SceneGate public demo

The Cloudflare Worker serves the public page and proxies small JSON job requests
to the GPU-only Flask API. It never exposes JupyterLab, its token, or provider
keys. Captioning is asynchronous: the Worker creates a job and polls it, which
avoids keeping an edge request open during GPU inference.

## 1. Start the GPU API

On the GPU, run the Gemma 4 Full checkout and set secrets in that shell only:

```bash
export DEMO_ORIGIN_TOKEN='generate-a-long-random-value'
export OPENROUTER_API_KEY='your-provider-key'
export GROQ_API_KEY='optional-fallback-key'
export FIREWORKS_API_KEY='optional-fallback-key'
chmod +x scripts/start_cloud_demo.sh
./scripts/start_cloud_demo.sh
```

The instance platform must expose port `8799` through its instance proxy. Do
not publish the Jupyter `/lab` endpoint.

## 2. Configure Worker secrets

From this directory, deploy only after the GPU endpoint responds to the
authenticated `/health` endpoint. Use Wrangler's interactive secret prompt;
do not put values in `wrangler.jsonc` or Git:

```bash
wrangler secret put GPU_ORIGIN_URL
wrangler secret put GPU_ORIGIN_TOKEN
wrangler secret put DEMO_ACCESS_TOKEN
wrangler deploy
```

- `GPU_ORIGIN_URL`: the GPU instance proxy base URL for port 8799. Keep it a
  Worker secret if it includes an instance credential.
- `GPU_ORIGIN_TOKEN`: exactly matches `DEMO_ORIGIN_TOKEN` on the GPU.
- `DEMO_ACCESS_TOKEN`: a separate access code entered by approved demo users.

The deployed `workers.dev` URL is the public demo address. The Worker accepts
only an HTTPS video URL; uploads are intentionally excluded to avoid edge body
limits and uncontrolled transfer cost.
