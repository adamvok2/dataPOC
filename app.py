from fastapi.responses import HTMLResponse

import os, json, tempfile, logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --------- ZERO-ENV DEFAULTS (local, writable, ephemeral on redeploy) ---------
DB_PATH   = "data/spss.duckdb"
META_PATH = "data/metadata.json"
TABLE     = "data"
BATCH     = 50_000  # small chunk = safer on tiny RAM

# ensure folders exist; never crash on boot
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(META_PATH).parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SPSS POC (no env vars)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for POC; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- in-memory OpenRouter client (set from the UI) ----
OPENROUTER = {"client": None, "model": "openai/gpt-4o-mini", "error": None}

def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")

def con():
    import duckdb
    return duckdb.connect(DB_PATH)

def load_meta():
    try:
        if not os.path.exists(META_PATH): return {}
        with open(META_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return {}

def save_meta(m: dict):
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Failed to write metadata: {e}")

# ---------- config endpoint: paste key from the page ----------
class SetConfigIn(BaseModel):
    openrouter_api_key: str
    model: str | None = None

@app.post("/config")
def set_config(cfg: SetConfigIn):
    OPENROUTER["error"] = None
    try:
        # import only when needed so startup never dies
        from openai import OpenAI  # requires openai>=1.0.0
        model = cfg.model or OPENROUTER["model"]
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=cfg.openrouter_api_key.strip())
        OPENROUTER["client"] = client
        OPENROUTER["model"] = model
        return {"ok": True, "model": model}
    except Exception as e:
        OPENROUTER["client"] = None
        OPENROUTER["error"] = f"OpenRouter init failed: {e}"
        return JSONResponse({"ok": False, "error": OPENROUTER["error"]}, status_code=400)

# ---------- endpoints ----------
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    import duckdb, pyreadstat, pandas as pd  # lazy imports
    # save upload to temp
    import tempfile as _tf
    with _tf.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
        tmp.write(await file.read())
        sav_path = tmp.name

    # 0-row read to get metadata
    _, meta = pyreadstat.read_sav(sav_path, row_limit=0)
    col_names = [sanitize(c) for c in meta.column_names]
    name_map  = dict(zip(meta.column_names, col_names))

    c = con()
    c.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} AS SELECT 1 AS _init LIMIT 0;")
    c.execute(f"DELETE FROM {TABLE};")

    # chunked load to avoid RAM spikes
    offset = 0; total = 0
    while True:
        df, _ = pyreadstat.read_sav(sav_path, row_offset=offset, row_limit=BATCH, apply_value_formats=False)
        if df.empty: break
        df = df.rename(columns=name_map)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string")
        if total == 0:
            c.register("chunk", df); c.execute(f"CREATE OR REPLACE TABLE {TABLE} AS SELECT * FROM chunk"); c.unregister("chunk")
        else:
            c.register("chunk", df); c.execute(f"INSERT INTO {TABLE} SELECT * FROM chunk"); c.unregister("chunk")
        n = len(df); total += n; offset += n
        if n < BATCH: break

    value_labels = {name_map.get(k, k): v for k, v in meta.variable_value_labels.items()}
    col_labels   = {name_map.get(k, k): v for k, v in meta.column_names_to_labels.items()}
    save_meta({"rows": total, "columns": col_names, "column_labels": col_labels,
               "value_labels": value_labels, "source_filename": file.filename})

    try: os.remove(sav_path)
    except Exception: pass

    return {"status": "ok", "rows": total, "columns": col_names}

@app.get("/schema")
def schema():
    c = con()
    info = c.execute(f"PRAGMA table_info('{TABLE}')").fetchall()
    meta = load_meta()
    cols = [{"name": i[1], "type": i[2], "label": meta.get("column_labels", {}).get(i[1], "")} for i in info]
    return {"table": TABLE, "columns": cols, "rows": meta.get("rows"), "source": meta.get("source_filename")}

def _ensure_select(sql: str) -> str:
    s = sql.strip().lower()
    if not s.startswith("select"): raise ValueError("SELECT only")
    if " limit " not in s: sql = sql.strip() + " LIMIT 1000"
    return sql

class SqlIn(BaseModel):
    sql: str

@app.post("/sql")
def sql(q: SqlIn):
    try:
        df = con().execute(_ensure_select(q.sql)).fetchdf()
        return {"rows": json.loads(df.to_json(orient="records"))}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# ---- “ask the dataset” via OpenRouter tool-calling ----
class AskIn(BaseModel):
    question: str

def t_schema(): return schema()

def t_value_counts(column: str, top: int = 20):
    import duckdb
    col = duckdb.escape_identifier(column)
    df = con().execute(
        f"SELECT {col} AS value, COUNT(*) AS count "
        f"FROM {TABLE} GROUP BY 1 ORDER BY 2 DESC NULLS LAST LIMIT {int(top)}"
    ).fetchdf()
    return {"column": column, "rows": json.loads(df.to_json(orient="records"))}

def t_describe():
    import pandas as pd, duckdb
    df0 = con().execute(f"SELECT * FROM {TABLE} LIMIT 0").fetchdf()
    num = [c for c in df0.columns if pd.api.types.is_numeric_dtype(df0[c])]
    if not num: return {"columns": [], "summary_row": {}}
    expr = ", ".join([f"min({duckdb.escape_identifier(c)}) as {c}_min, max({duckdb.escape_identifier(c)}) as {c}_max, avg({duckdb.escape_identifier(c)}) as {c}_mean" for c in num])
    df = con().execute(f"SELECT {expr} FROM {TABLE}").fetchdf()
    return {"columns": num, "summary_row": json.loads(df.to_json(orient="records"))[0]}

TOOLS = [
  {"type":"function","function":{"name":"get_schema","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"value_counts","parameters":{"type":"object","properties":{"column":{"type":"string"},"top":{"type":"integer"}},"required":["column"]}}},
  {"type":"function","function":{"name":"describe","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"run_sql","parameters":{"type":"object","properties":{"sql":{"type":"string"}},"required":["sql"]}}}
]

NAME_TO_TOOL = {
  "get_schema": lambda **kw: t_schema(),
  "value_counts": lambda **kw: t_value_counts(**kw),
  "describe":    lambda **kw: t_describe(),
  "run_sql":     lambda **kw: {"rows": json.loads(con().execute(_ensure_select(kw["sql"])).fetchdf().to_json(orient="records"))}
}

@app.post("/ask")
def ask(body: AskIn):
    if not OPENROUTER["client"]:
        return JSONResponse({"error":"OpenRouter key not set. Use the 'Set OpenRouter key' box on this page."}, status_code=400)

    msgs = [
        {"role":"system","content":"You are a data analyst. Use tools first, be concise, SELECT-only."},
        {"role":"user","content": body.question}
    ]

    client = OPENROUTER["client"]; model = OPENROUTER["model"]
    r = client.chat.completions.create(
        model=model, messages=msgs, tools=TOOLS, tool_choice="auto", temperature=0.1
    )
    m = r.choices[0].message

    tool_calls = []
    if getattr(m, "tool_calls", None):
        import json as _json
        for tc in m.tool_calls:
            name = tc.function.name
            args = _json.loads(tc.function.arguments or "{}")
            fn = NAME_TO_TOOL.get(name)
            try: result = fn(**args) if fn else {"error":"unknown tool"}
            except Exception as e: result = {"error": str(e)}
            tool_calls.append({"tool": name, "args": args, "result": result})
            msgs.append({"role":"tool","tool_call_id": tc.id, "name": name, "content": json.dumps(result)})
        r2 = client.chat.completions.create(model=model, messages=msgs, temperature=0.1)
        m = r2.choices[0].message

    return {"answer": m.content, "tools": tool_calls}

# ---------- tiny UI (root) ----------
@app.get("/", response_class=HTMLResponse)
def ui():
    return f"""
<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SPSS POC</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:32px auto;padding:0 16px}}
.card{{border:1px solid #ddd;border-radius:10px;padding:16px;margin:16px 0}}
button{{padding:8px 14px;border-radius:8px;border:1px solid #222;background:#111;color:#fff;cursor:pointer}}
button:disabled{{opacity:.6;cursor:default}}
pre{{background:#f7f7f7;padding:12px;border-radius:8px;overflow:auto}}
input[type=file],textarea,input[type=text]{{width:100%}}
small{{color:#666}}
label b{{display:inline-block;min-width:180px}}
</style>
</head><body>
<h1>SPSS POC</h1>
<p><small>DB: {DB_PATH} | META: {META_PATH} | BATCH_ROWS: {BATCH}</small></p>

<div class="card">
  <h2>0) Set OpenRouter key (stored in memory only)</h2>
  <label><b>API Key</b></label><input id="key" type="text" placeholder="sk-or-..." />
  <label><b>Model</b></label><input id="model" type="text" value="openai/gpt-4o-mini" />
  <br/><br/><button id="setKeyBtn">Set OpenRouter key</button>
  <div id="cfgOut"></div>
</div>

<div class="card">
  <h2>1) Upload .sav</h2>
  <input id="file" type="file" accept=".sav" />
  <p><small>Start small. This is a POC.</small></p>
  <button id="ingestBtn">Ingest</button>
  <div id="ingestOut"></div>
</div>

<div class="card">
  <h2>2) Schema</h2>
  <button id="schemaBtn">Refresh schema</button>
  <pre id="schemaPre"></pre>
</div>

<div class="card">
  <h2>3) Ask the dataset</h2>
  <textarea id="q" rows="3">Show top 10 values of Gender with counts.</textarea><br/><br/>
  <button id="askBtn">Ask</button>
  <h3>Answer</h3>
  <div id="answer" style="white-space:pre-wrap"></div>
  <h3>Tool calls</h3>
  <pre id="toolsPre"></pre>
</div>

<script>
const base = location.origin;
const el = (id) => document.getElementById(id);
const jfmt = (x) => { try { return JSON.stringify(x, null, 2); } catch(e){ return String(x); } };

el('setKeyBtn').onclick = async () => {
  const key = el('key').value.trim();
  const model = el('model').value.trim() || 'openai/gpt-4o-mini';
  if(!key){ alert('Enter your sk-or-... key'); return; }
  el('setKeyBtn').disabled = true;
  el('cfgOut').textContent = 'Setting...';
  try {
    const r = await fetch(base + '/config', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ openrouter_api_key: key, model })
    });
    const txt = await r.text(); let j={}; try{ j=JSON.parse(txt) }catch{}
    if(!r.ok){ el('cfgOut').textContent = j.error ? ('Error: ' + j.error) : ('HTTP '+r.status+' '+txt); }
    else { el('cfgOut').textContent = 'OK. Model: ' + (j.model || model); }
  } catch(e){ el('cfgOut').textContent = String(e); }
  finally { el('setKeyBtn').disabled = false; }
};

el('ingestBtn').onclick = async () => {
  const f = el('file').files[0];
  if(!f){ alert('Pick a .sav file first'); return; }
  el('ingestBtn').disabled = true;
  el('ingestOut').textContent = 'Uploading...';
  try {
    const fd = new FormData(); fd.append('file', f);
    const r = await fetch(base + '/ingest', { method:'POST', body: fd });
    const text = await r.text(); el('ingestOut').innerHTML = '<pre>'+text+'</pre>';
  } catch (e) { el('ingestOut').textContent = String(e); }
  finally { el('ingestBtn').disabled = false; }
};

el('schemaBtn').onclick = async () => {
  el('schemaPre').textContent = 'Loading...';
  try { const r = await fetch(base + '/schema'); const j = await r.json(); el('schemaPre').textContent = jfmt(j); }
  catch(e){ el('schemaPre').textContent = String(e); }
};

el('askBtn').onclick = async () => {
  const q = el('q').value.trim();
  if(!q){ alert('Type a question'); return; }
  el('askBtn').disabled = true;
  el('answer').textContent = 'Thinking...'; el('toolsPre').textContent = '';
  try {
    const r = await fetch(base + '/ask', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ question: q }) });
    const txt = await r.text(); let j={}; try{ j=JSON.parse(txt) }catch{}
    if (!r.ok) { el('answer').textContent = j.error ? ('Error: ' + j.error) : ('HTTP '+r.status+' '+txt); return; }
    el('answer').textContent = j.answer || ''; el('toolsPre').textContent = jfmt(j.tools || []);
  } catch(e) { el('answer').textContent = String(e); }
  finally { el('askBtn').disabled = false; }
};
</script>
</body></html>
"""
