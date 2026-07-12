const encoder = new TextEncoder();
const MAX_UPLOAD_BYTES = 80 * 1024 * 1024;

function responseJson(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
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
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gemma 4 SceneGate demo</title>
  <style>
    :root{color-scheme:dark}body{margin:0;background:#071119;color:#edf4f8;font-family:Inter,system-ui,sans-serif}.wrap{max-width:900px;margin:0 auto;padding:52px 24px}.tag{color:#57e0c2;letter-spacing:2px;font-size:12px;font-weight:800;text-transform:uppercase}h1{font-size:clamp(38px,7vw,72px);line-height:1;margin:16px 0}p{color:#aebdca;font-size:17px;line-height:1.5}.card{background:#0e202b;border:1px solid #234151;border-radius:16px;padding:22px;margin-top:28px}label{display:block;margin-top:16px;color:#aebdca;font-size:14px}input,button{box-sizing:border-box;width:100%;border-radius:10px;padding:14px;font:inherit;margin-top:8px}input{background:#071119;border:1px solid #355363;color:#edf4f8}button{background:#57e0c2;border:0;color:#071119;font-weight:800;cursor:pointer;margin-top:22px}button:disabled{opacity:.55;cursor:wait}.or{text-align:center;color:#6f8998;margin:20px 0 0}.caption{border-left:3px solid #57e0c2;margin-top:16px;padding:10px 14px;background:#0a1821}.style{color:#82aaff;font-size:12px;font-weight:800;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px}.error{color:#ff9292}.hint{font-size:13px;color:#7f98a7}.preview{display:none;max-width:100%;max-height:330px;margin-top:16px;border-radius:12px;background:#000}.status{min-height:22px;margin-top:20px}
  </style>
  <main class="wrap"><div class="tag">Gemma 4 SceneGate</div><h1>Grounded video captions.</h1><p>Upload a short video or provide a public HTTPS URL. The GPU returns the same scene in four styles: formal, sarcastic, humorous tech and humorous non-tech.</p><section class="card"><label>Demo access code<input id="code" type="password" autocomplete="current-password" required></label><label>Public HTTPS video URL <span class="hint">(optional)</span><input id="url" type="url" placeholder="https://.../clip.mp4"></label><div class="or">or</div><label>Upload a video <span class="hint">(MP4, MOV, WebM, MKV; max 80 MB)</span><input id="file" type="file" accept="video/mp4,video/quicktime,video/webm,video/x-matroska,.mp4,.mov,.webm,.mkv"></label><video id="preview" class="preview" controls muted></video><button id="go">Create four captions</button><div id="out" class="status" aria-live="polite"></div></section></main>
  <script>
    const $=id=>document.getElementById(id), out=$('out'), go=$('go'), file=$('file'), url=$('url'), preview=$('preview');
    const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
    const token=()=>$('code').value;
    function status(message,error=false){out.replaceChildren();const p=document.createElement('p');p.textContent=message;if(error)p.className='error';out.append(p)}
    function captions(values){out.replaceChildren();for(const [style,caption] of Object.entries(values)){const card=document.createElement('div'), name=document.createElement('div'), text=document.createElement('div');card.className='caption';name.className='style';name.textContent=style;text.textContent=caption;card.append(name,text);out.append(card)}}
    file.addEventListener('change',()=>{const selected=file.files[0];if(!selected){preview.removeAttribute('src');preview.style.display='none';return}url.value='';preview.src=URL.createObjectURL(selected);preview.style.display='block'});
    url.addEventListener('input',()=>{if(url.value)file.value=''});
    async function poll(id){for(;;){await sleep(2500);const r=await fetch('/api/jobs/'+id,{headers:{'x-demo-access-token':token()}});const d=await r.json();if(!r.ok)throw new Error(d.error||'status failed');if(d.status==='complete'){captions(d.captions);return}if(d.status==='failed')throw new Error(d.error||'captioning failed');status('GPU status: '+d.status+'...')}}
    go.addEventListener('click',async()=>{const selected=file.files[0], source=url.value.trim();if(!token()){status('Enter the demo access code.',true);return}if(!selected&&!source){status('Upload a video or provide a public HTTPS URL.',true);return}if(selected&&source){status('Choose one source: upload or URL.',true);return}go.disabled=true;status(selected?'Uploading video to the GPU queue...':'Queueing the GPU job...');try{let r;if(selected){if(selected.size>80*1024*1024)throw new Error('Video exceeds the 80 MB demo limit.');const form=new FormData();form.append('video',selected,selected.name);r=await fetch('/api/uploads',{method:'POST',headers:{'x-demo-access-token':token()},body:form})}else{r=await fetch('/api/jobs',{method:'POST',headers:{'content-type':'application/json','x-demo-access-token':token()},body:JSON.stringify({video_url:source})})}const d=await r.json();if(!r.ok)throw new Error(d.error||'request failed');await poll(d.job_id)}catch(error){status(error instanceof Error?error.message:'request failed',true)}finally{go.disabled=false}});
  </script></body></html>`;
}

function upstreamUrl(origin, suffix) {
  return `${origin.replace(/\/$/, "")}${suffix}`;
}

async function proxy(env, suffix, init) {
  if (!env.GPU_ORIGIN_URL || !env.GPU_ORIGIN_TOKEN) return responseJson({ error: "demo backend is not configured" }, 503);
  try {
    const upstream = await fetch(upstreamUrl(env.GPU_ORIGIN_URL, suffix), init);
    return new Response(upstream.body, { status: upstream.status, headers: { "content-type": upstream.headers.get("content-type") || "application/json; charset=utf-8", "cache-control": "no-store" } });
  } catch (error) {
    console.log(JSON.stringify({ event: "gpu_proxy_failure", message: String(error) }));
    return responseJson({ error: "GPU service is unavailable; try again shortly" }, 503);
  }
}

function upstreamHeaders(env, contentType) {
  const headers = new Headers({ "x-demo-origin-token": env.GPU_ORIGIN_TOKEN });
  if (contentType) headers.set("content-type", contentType);
  return headers;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/") return new Response(page(), { headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store", "content-security-policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; media-src blob:; base-uri 'none'; frame-ancestors 'none'" } });
    if (!url.pathname.startsWith("/api/")) return responseJson({ error: "not found" }, 404);
    if (!(await sameSecret(request.headers.get("x-demo-access-token"), env.DEMO_ACCESS_TOKEN))) return responseJson({ error: "unauthorized" }, 401);
    if (request.method === "POST" && url.pathname === "/api/jobs") {
      const size = Number(request.headers.get("content-length") || "0");
      if (size > 12 * 1024) return responseJson({ error: "request is too large" }, 413);
      const body = await request.text();
      return proxy(env, "/jobs", { method: "POST", body, headers: upstreamHeaders(env, "application/json") });
    }
    if (request.method === "POST" && url.pathname === "/api/uploads") {
      const size = Number(request.headers.get("content-length") || "0");
      if (size > MAX_UPLOAD_BYTES) return responseJson({ error: "video exceeds the 80 MB demo limit" }, 413);
      const contentType = request.headers.get("content-type");
      if (!contentType?.startsWith("multipart/form-data")) return responseJson({ error: "upload must be multipart form data" }, 400);
      return proxy(env, "/uploads", { method: "POST", body: request.body, headers: upstreamHeaders(env, contentType) });
    }
    const match = url.pathname.match(/^\/api\/jobs\/([a-f0-9]{36})$/);
    if (request.method === "GET" && match) return proxy(env, `/jobs/${match[1]}`, { headers: upstreamHeaders(env) });
    return responseJson({ error: "not found" }, 404);
  },
};
