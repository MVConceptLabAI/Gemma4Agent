const encoder = new TextEncoder();

function responseJson(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

async function sameSecret(supplied, expected) {
  if (!supplied || !expected) return false;
  const left = encoder.encode(supplied);
  const right = encoder.encode(expected);
  if (left.byteLength !== right.byteLength) return false;
  return crypto.subtle.timingSafeEqual(left, right);
}

function page() {
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gemma 4 SceneGate</title>
  <style>body{margin:0;background:#071119;color:#edf4f8;font-family:Inter,system-ui,sans-serif}.wrap{max-width:850px;margin:0 auto;padding:64px 24px}.tag{color:#57e0c2;letter-spacing:2px;font-size:12px;font-weight:800;text-transform:uppercase}h1{font-size:clamp(38px,7vw,72px);line-height:1;margin:16px 0}p{color:#aebdca;font-size:18px;line-height:1.5}.card{background:#0e202b;border:1px solid #234151;border-radius:16px;padding:22px;margin-top:28px}input,button{box-sizing:border-box;width:100%;border-radius:10px;padding:14px;font:inherit;margin-top:12px}input{background:#071119;border:1px solid #355363;color:#edf4f8}button{background:#57e0c2;border:0;color:#071119;font-weight:800;cursor:pointer}button:disabled{opacity:.55;cursor:wait}.caption{border-left:3px solid #57e0c2;margin-top:16px;padding:10px 14px}.style{color:#82aaff;font-size:12px;font-weight:800;letter-spacing:1px;text-transform:uppercase}.error{color:#ff9292}</style>
  <main class="wrap"><div class="tag">Gemma 4 SceneGate</div><h1>Grounded video captions.</h1><p>Submit a public HTTPS video URL. The GPU creates four factual captions: formal, sarcastic, humorous tech and humorous non-tech.</p><section class="card"><label>Demo access code<input id="code" type="password" autocomplete="current-password"></label><label>Public HTTPS video URL<input id="url" type="url" placeholder="https://.../clip.mp4"></label><button id="go">Create captions</button><div id="out" aria-live="polite"></div></section></main>
  <script>const $=id=>document.getElementById(id);const out=$('out');const go=$('go');const headers=()=>({'content-type':'application/json','x-demo-access-token':$('code').value});const sleep=ms=>new Promise(r=>setTimeout(r,ms));function show(m,c=''){out.innerHTML='<p class="'+c+'">'+m+'</p>'}async function poll(id){for(;;){await sleep(2500);const r=await fetch('/api/jobs/'+id,{headers:{'x-demo-access-token':$('code').value}});const d=await r.json();if(!r.ok)throw new Error(d.error||'status failed');if(d.status==='complete'){out.innerHTML=Object.entries(d.captions).map(([s,c])=>'<div class="caption"><div class="style">'+s+'</div><div>'+c+'</div></div>').join('');return}if(d.status==='failed')throw new Error(d.error);show('GPU status: '+d.status+'...')}}go.onclick=async()=>{go.disabled=true;show('Queueing the GPU job...');try{const r=await fetch('/api/jobs',{method:'POST',headers:headers(),body:JSON.stringify({video_url:$('url').value})});const d=await r.json();if(!r.ok)throw new Error(d.error||'request failed');await poll(d.job_id)}catch(e){show(e.message,'error')}finally{go.disabled=false}};</script></body></html>`;
}

function upstreamUrl(origin, suffix) {
  return `${origin.replace(/\/$/, "")}${suffix}`;
}

async function proxy(request, env, suffix, init = {}) {
  if (!env.GPU_ORIGIN_URL || !env.GPU_ORIGIN_TOKEN) {
    return responseJson({ error: "demo backend is not configured" }, 503);
  }
  try {
    const upstream = await fetch(upstreamUrl(env.GPU_ORIGIN_URL, suffix), {
      ...init,
      headers: {
        "content-type": "application/json",
        "x-demo-origin-token": env.GPU_ORIGIN_TOKEN,
      },
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") || "application/json; charset=utf-8", "cache-control": "no-store" },
    });
  } catch (error) {
    console.log(JSON.stringify({ event: "gpu_proxy_failure", message: String(error) }));
    return responseJson({ error: "GPU service is unavailable; try again shortly" }, 503);
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/") {
      return new Response(page(), { headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store", "content-security-policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'" } });
    }
    if (!url.pathname.startsWith("/api/")) return responseJson({ error: "not found" }, 404);
    const supplied = request.headers.get("x-demo-access-token");
    if (!(await sameSecret(supplied, env.DEMO_ACCESS_TOKEN))) return responseJson({ error: "unauthorized" }, 401);
    if (request.method === "POST" && url.pathname === "/api/jobs") {
      const size = Number(request.headers.get("content-length") || "0");
      if (size > 12 * 1024) return responseJson({ error: "request is too large" }, 413);
      const body = await request.text();
      return proxy(request, env, "/jobs", { method: "POST", body });
    }
    const match = url.pathname.match(/^\/api\/jobs\/([a-f0-9]{36})$/);
    if (request.method === "GET" && match) return proxy(request, env, `/jobs/${match[1]}`);
    return responseJson({ error: "not found" }, 404);
  },
};
