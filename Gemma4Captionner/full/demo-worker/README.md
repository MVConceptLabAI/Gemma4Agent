# Optional Gemma 4 Captioning GPU Worker demo

> Optional deployment adapter. The V18 hackathon container and the static
> `docs/demo.html` page do not require this Worker.

The Cloudflare Worker serves the public page and proxies job requests to the
GPU-only Flask API. It never exposes JupyterLab, its token, or provider keys.
Captioning is asynchronous: the Worker creates a job and polls it, which avoids
keeping an edge request open during GPU inference.

Demo users can either paste a public HTTPS video URL or upload one MP4, MOV,
WebM, or MKV file up to 80 MB. Uploads stream through the Worker instead of
being buffered there and are removed from the GPU after captioning completes.

## 1. Start the GPU API

On the GPU, run the Gemma 4 Full checkout and set secrets in that shell only:

```bash
export DEMO_ORIGIN_TOKEN='generate-a-long-random-value'
export OPENROUTER_API_KEY='your-provider-key'
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

The deployed `workers.dev` URL is the public interactive demo address. Set
`DEMO_UPLOAD_MAX_BYTES` on the GPU only when a lower upload limit is required;
it must stay at or below the Cloudflare account request-body allowance.
