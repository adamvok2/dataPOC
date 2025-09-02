from fastapi.responses import HTMLResponse

import os, json, tempfile, logging, traceback
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
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

# ---- helpers ----
def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")

def qi(name: str) -> str:
    # quote a SQL identifier for DuckDB: "foo""bar" style
    return '"' + str(name).replace('"', '""') + '"'

def duckdb_literal(s: str) -> str:
    # quote a SQL string literal
    return "'" + str(s).replace("'", "''") + "'"

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

# Register value-label UDF once per connection
def register_label_udf(c):
    # Pull labels from metadata once per connection
    meta = load_meta()
    vlabels = meta.get("value_labels", {})  # {"DB2": {"1":"Male","2":"Female"}, ...}

    # Normalize mapping keys to strings for robust matching
    norm = {}
    for col, mapping in vlabels.items():
        norm[str(col)] = {str(k): v for k, v in mapping.items()}
    vlabels = norm

    def _label(col_name, val):
        # Return the original value if no mapping; otherwise look up by string key
        mapping = vlabels.get(str(col_name))
        if not mapping:
            return val
        if val is None:
            return val
        key = str(val)
        return mapping.get(key, mapping.get(val, val))

    # IMPORTANT: don’t pass parameter/return type hints (they break on some versions)
    # Always replace; don’t swallow failures.
    c.create_function("label", _label, replace=True)


def con():
    import duckdb
    c = duckdb.connect(DB_PATH)
    register_label_udf(c)
    return c

# pick a canonical weight column by creating a view data_w with 'w'
DEFAULT_WEIGHT_COLUMNS = ["w", "weight", "case_weight", "CaseWeight", "Weight"]

def ensure_weight_view():
    c = con()
    cols = [r[1] for r in c.execute(f"PRAGMA table_info({qi(TABLE)})").fetchall()]
    chosen = next((nm for nm in DEFAULT_WEIGHT_COLUMNS if nm in cols and nm != "w"), None)
    if chosen:
        c.execute(f'CREATE OR REPLACE VIEW {qi("data_w")} AS SELECT *, COALESCE({qi(chosen)}, 1.0) AS w FROM {qi(TABLE)}')
    else:
        # either already has w, or no weight column -> default 1.0
        if "w" in cols:
            c.execute(f'CREATE OR REPLACE VIEW {qi("data_w")} AS SELECT * FROM {qi(TABLE)}')
        else:
            c.execute(f'CREATE OR REPLACE VIEW {qi("data_w")} AS SELECT *, 1.0 AS w FROM {qi(TABLE)}')

# ---------- config endpoint: paste key from the page ----------
class SetConfigIn(BaseModel):
    openrouter_api_key: str
    model: str | None = None

@app.post("/config")
def set_config(cfg: SetConfigIn):
    OPENROUTER['error'] = None
    try:
        from openai import OpenAI  # openai>=1.x
        import httpx

        model = cfg.model or OPENROUTER['model']

        # Build our own HTTP client so the OpenAI SDK never tries to pass `proxies=...`
        http_client = httpx.Client(timeout=60.0)  # respects HTTP(S)_PROXY by default

        client = OpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key=cfg.openrouter_api_key.strip(),
            http_client=http_client,
            # Optional OpenRouter headers (set your domain/title if you want analytics):
            default_headers={
                'HTTP-Referer': 'https://example.com',
                'X-Title': 'SPSS POC',
            },
        )

        OPENROUTER['client'] = client
        OPENROUTER['model'] = model
        return {'ok': True, 'model': model}
    except Exception as e:
        OPENROUTER['client'] = None
        OPENROUTER['error'] = f'OpenRouter init failed: {e}'
        return JSONResponse({'ok': False, 'error': OPENROUTER['error']}, status_code=400)

# ---------- endpoints ----------
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    import duckdb, pyreadstat, pandas as pd  # lazy imports
    # save upload to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
        tmp.write(await file.read())
        sav_path = tmp.name

    # 0-row read to get metadata
    _, meta = pyreadstat.read_sav(sav_path, row_limit=0)
    col_names = [sanitize(c) for c in meta.column_names]
    name_map  = dict(zip(meta.column_names, col_names))

    c = con()
    c.execute(f"CREATE TABLE IF NOT EXISTS {qi(TABLE)} AS SELECT 1 AS _init LIMIT 0;")
    c.execute(f"DELETE FROM {qi(TABLE)};")

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
            c.register("chunk", df); c.execute(f"CREATE OR REPLACE TABLE {qi(TABLE)} AS SELECT * FROM chunk"); c.unregister("chunk")
        else:
            c.register("chunk", df); c.execute(f"INSERT INTO {qi(TABLE)} SELECT * FROM chunk"); c.unregister("chunk")
        n = len(df); total += n; offset += n
        if n < BATCH: break

    value_labels = {name_map.get(k, k): v for k, v in meta.variable_value_labels.items()}
    col_labels   = {name_map.get(k, k): v for k, v in meta.column_names_to_labels.items()}
    save_meta({"rows": total, "columns": col_names, "column_labels": col_labels,
               "value_labels": value_labels, "source_filename": file.filename})

    try: os.remove(sav_path)
    except Exception: pass

    # ensure canonical weight view exists for this dataset
    ensure_weight_view()

    return {"status": "ok", "rows": total, "columns": col_names}

@app.get("/schema")
def schema():
    c = con()
    info = c.execute(f"PRAGMA table_info({qi(TABLE)})").fetchall()
    meta = load_meta()
    cols = [{"name": i[1], "type": i[2], "label": meta.get("column_labels", {}).get(i[1], "")} for i in info]
    return {"table": TABLE, "columns": cols, "rows": meta.get("rows"), "source": meta.get("source_filename")}

def _ensure_select(sql: str) -> str:
    # allow WITH ... SELECT
    s = sql.strip().lower()
    if not (s.startswith("select") or s.startswith("with")):
        raise ValueError("SELECT/WITH only")
    # block anything dangerous-ish
    banned = [";", " drop ", " create ", " alter ", " insert ", " update ", " delete ", " attach ", " copy ", " pragma "]
    if any(b in s for b in banned):
        raise ValueError("Only a single SELECT/WITH query is allowed")
    # add LIMIT if user forgot and query isn't a naked CTE block
    if " limit " not in s:
        # naive check: if it ends with ')', it might still be a SELECT (...) subquery; we still add LIMIT
        sql = sql.strip() + " LIMIT 1000"
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

# ---- “ask the dataset” via OpenRouter tool-calling (SQL-first) ----
class AskIn(BaseModel):
    question: str

def t_schema(): return schema()

TOOLS = [
  {"type":"function","function":{"name":"run_sql","parameters":{"type":"object","properties":{"sql":{"type":"string"}},"required":["sql"]}}},
  {"type":"function","function":{"name":"get_schema","parameters":{"type":"object","properties":{}}}}
]

def _run_sql_impl(sql: str):
    c = con()
    try:
        df = c.execute(_ensure_select(sql)).fetchdf()
    except Exception as e:
        msg = str(e)
        # If the function is missing (e.g., brand-new process), register and retry once
        if "Scalar Function with name label does not exist" in msg:
            register_label_udf(c)
            df = c.execute(_ensure_select(sql)).fetchdf()
        else:
            raise
    return {"rows": json.loads(df.to_json(orient="records"))}

NAME_TO_TOOL["run_sql"] = lambda **kw: _run_sql_impl(kw["sql"])


SYSTEM_PROMPT = """You are a senior market-research data analyst.
- Always use the run_sql tool to answer, composing a single SELECT (WITH allowed).
- Default table is data; prefer the view data_w which exposes a canonical weight column w.
- You may use label('<column_name>', <column>) to convert coded values using SPSS value labels.
- For percentages: return 100 * SUM(CASE WHEN <condition> THEN w ELSE 0 END) / NULLIF(SUM(w),0) AS pct, and also return SUM(w) AS base_w.
- For weighted averages: SUM(<metric> * w) / NULLIF(SUM(w),0) AS weighted_avg, include base_w = SUM(w).
- Use CTEs for base filters, recodes (CASE), trims (PERCENT_RANK), top-k, etc.
- Quote identifiers that contain spaces or special characters with double quotes.
- Never write non-SELECT statements; no PRAGMA/DDL/DML. Keep results concise."""

@app.post("/ask")
def ask(body: AskIn, request: Request):
    if not OPENROUTER["client"]:
        return JSONResponse({"error":"OpenRouter key not set. Use the 'Set OpenRouter key' box on this page."}, status_code=400)

    try:
        msgs = [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": body.question}
        ]

        client = OPENROUTER["client"]; model = OPENROUTER["model"]

        # 1) First turn: let the model decide (should choose run_sql)
        r = client.chat.completions.create(
            model=model,
            messages=msgs,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=800,
        )

        m = r.choices[0].message
        raw_tool_calls = getattr(m, "tool_calls", None) or (m.get("tool_calls") if isinstance(m, dict) else None)
        tool_results = []

        if raw_tool_calls:
            import json as _json

            norm_tool_calls = []
            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None) if hasattr(tc, "function") else (tc.get("function"))
                name = getattr(fn, "name", None) if hasattr(fn, "name") else (fn.get("name") if fn else None)
                arguments = getattr(fn, "arguments", None) if hasattr(fn, "arguments") else (fn.get("arguments") if fn else None)
                tc_id = getattr(tc, "id", None) if hasattr(tc, "id") else tc.get("id")

                norm_tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments or "{}",
                    },
                })

            # Append the assistant turn with tool_calls
            msgs.append({"role": "assistant", "content": None, "tool_calls": norm_tool_calls})

            # Execute tools and append results
            for tc in norm_tool_calls:
                name = tc["function"]["name"]
                args_raw = tc["function"]["arguments"]
                try:
                    args = json.loads(args_raw or "{}")
                except Exception:
                    args = {}

                fn = NAME_TO_TOOL.get(name)
                try:
                    result = fn(**args) if fn else {"error": "unknown tool: " + str(name)}
                except Exception as e:
                    result = {"error": str(e)}

                tool_results.append({"tool": name, "args": args, "result": result})

                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": json.dumps(result)
                })

            # 2) Second turn: model sees tool outputs and answers
            r2 = client.chat.completions.create(
                model=model,
                messages=msgs,
                tools=TOOLS,
                temperature=0.1,
                max_tokens=800,
            )
            m = r2.choices[0].message

        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
        return {"answer": content, "tools": tool_results}

    except Exception as e:
        logging.error("ASK endpoint failed: %s\n%s", str(e), traceback.format_exc())
        return JSONResponse({"error": f"/ask failed: {e}"}, status_code=500)

# ---------- tiny UI (root) ----------
@app.get("/", response_class=HTMLResponse)
def ui():
    # keep the dynamic bit small, avoid f-strings over JS/CSS with braces
    stats = f'<p><small>DB: {DB_PATH} | META: {META_PATH} | BATCH_ROWS: {BATCH}</small></p>'

    return """
<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SPSS POC</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:32px auto;padding:0 16px}
.card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:16px 0}
button{padding:8px 14px;border-radius:8px;border:1px solid #222;background:#111;color:#fff;cursor:pointer}
button:disabled{opacity:.6;cursor:default}
pre{background:#f7f7f7;padding:12px;border-radius:8px;overflow:auto}
input[type=file],textarea,input[type=text]{width:100%}
small{color:#666}
label b{display:inline-block;min-width:180px}
</style>
</head><body>
<h1>SPSS POC</h1>
""" + stats + """
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
  <textarea id="q" rows="3">Show % of Q1=1 among UK adults by Gender, include base sizes. Use data_w and label().</textarea><br/><br/>
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
