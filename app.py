# app.py
from fastapi.responses import HTMLResponse
import os, json, tempfile, logging, re, traceback
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# =========================
# HARD-CODED OPENROUTER AUTH
# =========================
HARDCODED_MODEL = "openai/gpt-4o-2024-11-20"
HARDCODED_KEY   = "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"

def get_openrouter_client():
    # Fresh client per request; avoids any accidental in-memory overrides.
    from openai import OpenAI  # openai>=1.x
    import httpx
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=HARDCODED_KEY.strip(),
        http_client=httpx.Client(timeout=60.0),
        default_headers={
            "HTTP-Referer": "https://example.com",
            "X-Title": "SPSS POC",
        },
    )

# --------- ZERO-ENV DEFAULTS (local, writable, ephemeral on redeploy) ---------
DB_PATH   = "data/spss.duckdb"
META_PATH = "data/metadata.json"
TABLE     = "data"
BATCH     = 50_000  # small chunk = safer on tiny RAM

# ensure folders exist; never crash on boot
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(META_PATH).parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SPSS POC (hardcoded auth)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # POC: open; lock down for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")

def qi(name: str) -> str:
    # quote an identifier for DuckDB
    return '"' + str(name).replace('"', '""') + '"'

def con():
    import duckdb
    c = duckdb.connect(DB_PATH)
    # Register a UDF to resolve SPSS value labels:
    def _label(col_name, value):
        meta = load_meta()
        vl = meta.get("value_labels", {}).get(str(col_name), {})
        # keys may be ints or strings depending on pyreadstat; try multiple lookups
        if value is None:
            return None
        if value in vl:
            return vl[value]
        s = str(value)
        if s in vl:
            return vl[s]
        try:
            f = float(value)
            if f in vl:
                return vl[f]
            i = int(f)
            if i in vl:
                return vl[i]
        except Exception:
            pass
        return value  # fallback to raw code
    try:
        c.create_function("label", _label)
    except Exception:
        # already registered on this connection is fine
        pass
    return c

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

# ---------- ingest .sav ----------
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    import duckdb, pyreadstat, pandas as pd  # lazy imports
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
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
        df, _ = pyreadstat.read_sav(
            sav_path, row_offset=offset, row_limit=BATCH, apply_value_formats=False
        )
        if df.empty:
            break
        df = df.rename(columns=name_map)
        # normalize strings for DuckDB
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string")
        if total == 0:
            c.register("chunk", df)
            c.execute(f"CREATE OR REPLACE TABLE {TABLE} AS SELECT * FROM chunk")
            c.unregister("chunk")
        else:
            c.register("chunk", df)
            c.execute(f"INSERT INTO {TABLE} SELECT * FROM chunk")
            c.unregister("chunk")
        n = len(df); total += n; offset += n
        if n < BATCH:
            break

    value_labels = {name_map.get(k, k): v for k, v in meta.variable_value_labels.items()}
    col_labels   = {name_map.get(k, k): v for k, v in meta.column_names_to_labels.items()}
    save_meta({
        "rows": total,
        "columns": col_names,
        "column_labels": col_labels,
        "value_labels": value_labels,
        "source_filename": file.filename
    })

    try: os.remove(sav_path)
    except Exception: pass

    return {"status": "ok", "rows": total, "columns": col_names}

# ---------- schema ----------
@app.get("/schema")
def schema():
    c = con()
    info = c.execute(f"PRAGMA table_info('{TABLE}')").fetchall()
    meta = load_meta()
    cols = [{"name": i[1], "type": i[2], "label": meta.get("column_labels", {}).get(i[1], "")} for i in info]
    return {"table": TABLE, "columns": cols, "rows": meta.get("rows"), "source": meta.get("source_filename")}

# ---------- SQL endpoint (SELECT-only, supports label() UDF) ----------
def _ensure_select(sql: str) -> str:
    s = sql.strip().lower()
    if not s.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    if " limit " not in s and not s.endswith(" limit"):
        sql = sql.strip() + " LIMIT 1000"
    return sql

class SqlIn(BaseModel):
    sql: str

def _run_sql_impl(sql: str):
    try:
        sql = _ensure_select(sql)
        df = con().execute(sql).fetchdf()
        return {"rows": json.loads(df.to_json(orient="records"))}
    except Exception as e:
        return {"error": str(e)}

@app.post("/sql")
def sql(q: SqlIn):
    out = _run_sql_impl(q.sql)
    if "error" in out:
        return JSONResponse(out, status_code=400)
    return out

# ---------- Tools for LLM ----------
class AskIn(BaseModel):
    question: str

def t_schema():
    return schema()

def t_value_counts(column: str, top: int = 20):
    c = con()
    colq = qi(column)
    tblq = qi(TABLE)
    # Use the registered label() UDF to turn codes into human labels when available
    df = c.execute(
        f"""
        SELECT
          {colq} AS value,
          label('{column}', {colq}) AS label,
          COUNT(*) AS count
        FROM {tblq}
        GROUP BY 1,2
        ORDER BY 3 DESC NULLS LAST
        LIMIT {int(top)}
        """
    ).fetchdf()
    return {"column": column, "rows": json.loads(df.to_json(orient="records"))}

def t_describe():
    c = con()
    info = c.execute(f"PRAGMA table_info({qi(TABLE)})").fetchdf()

    numeric_prefixes = (
        "TINYINT","SMALLINT","INTEGER","BIGINT","HUGEINT",
        "UTINYINT","USMALLINT","UINTEGER","UBIGINT",
        "FLOAT","REAL","DOUBLE","DECIMAL"
    )
    num_cols = [row["name"] for _, row in info.iterrows()
                if str(row["type"]).upper().startswith(numeric_prefixes)]
    if not num_cols:
        return {"columns": [], "summary_row": {}}

    expr_parts = []
    for c_name in num_cols:
        idq = qi(c_name)
        expr_parts.append(f"min({idq}) as {c_name}_min")
        expr_parts.append(f"max({idq}) as {c_name}_max")
        expr_parts.append(f"avg({idq}) as {c_name}_mean")

    expr = ", ".join(expr_parts)
    df = c.execute(f"SELECT {expr} FROM {qi(TABLE)}").fetchdf()
    return {"columns": num_cols, "summary_row": json.loads(df.to_json(orient='records'))[0]}

TOOLS = [
  {"type":"function","function":{"name":"get_schema","description":"Get table, columns (with types & labels) and row count.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"value_counts","description":"Top-N counts of a categorical column, returning raw value, human label (via SPSS metadata), and count.","parameters":{"type":"object","properties":{"column":{"type":"string"},"top":{"type":"integer","default":20}},"required":["column"]}} },
  {"type":"function","function":{"name":"describe","description":"Summaries (min/max/mean) for numeric columns.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"run_sql","description":"Run a SELECT-only SQL query against the dataset. You can use label('<col>', <col>) in SELECTs for human labels.","parameters":{"type":"object","properties":{"sql":{"type":"string"}},"required":["sql"]}}}
]

NAME_TO_TOOL = {
  "get_schema": lambda **kw: t_schema(),
  "value_counts": lambda **kw: t_value_counts(**kw),
  "describe":    lambda **kw: t_describe(),
  "run_sql":     lambda **kw: _run_sql_impl(kw["sql"])
}

SYSTEM_PROMPT = """
You are a data analyst working over one DuckDB table named 'data'.
- ALWAYS start with a tool call to inspect schema or compute results.
- Prefer value_counts(column) for categorical summaries.
- Use run_sql for custom queries. Only SELECTs are allowed.
- You may use label('<col>', <col>) in SQL to render SPSS value labels.
- Be concise and include small result snippets.
"""

@app.post("/ask")
def ask(body: AskIn, request: Request):
    client = get_openrouter_client()
    model = HARDCODED_MODEL

    try:
        msgs = [
            {"role":"system","content": SYSTEM_PROMPT.strip()},
            {"role":"user","content": body.question}
        ]

        # 1) Let the model decide tool calls
        r = client.chat.completions.create(
            model=model,
            messages=msgs,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=800,
        )

        # OpenAI SDK may return objects or dict-like; normalize
        m = r.choices[0].message
        raw_tool_calls = getattr(m, "tool_calls", None) or (m.get("tool_calls") if isinstance(m, dict) else None)

        tool_results = []

        if raw_tool_calls:
            import json as _json

            # Append assistant turn containing tool_calls (required before tool messages)
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

            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": norm_tool_calls,
            })

            # Execute tools and append tool results
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
                tools=TOOLS,          # keep tools available
                temperature=0.1,
                max_tokens=800,
            )
            m = r2.choices[0].message

        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
        return {"answer": content, "tools": tool_results}

    except Exception as e:
        logging.error("ASK endpoint failed: %s\n%s", str(e), traceback.format_exc())
        # If OpenRouter returned an HTTP error, it will be inside the exception text
        return JSONResponse({"error": f"/ask failed: {e}"}, status_code=500)

# ---------- simple OpenRouter ping/debug ----------
@app.get("/openrouter/ping")
def openrouter_ping():
    import httpx
    try:
        r = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {HARDCODED_KEY.strip()}"},
            timeout=20.0,
        )
        mask = HARDCODED_KEY.strip()
        masked = (mask[:10] + "..." + mask[-6:]) if len(mask) > 20 else "short-key"
        body = r.text
        if len(body) > 4000:  # keep it short
            body = body[:4000] + "…"
        return {"sent_key_preview": masked, "status_code": r.status_code, "ok": r.status_code == 200, "body": body}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------- tiny UI (root) ----------
@app.get("/", response_class=HTMLResponse)
def ui():
    stats = '<p><small>DB: ' + DB_PATH + ' | META: ' + META_PATH + ' | BATCH_ROWS: ' + str(BATCH) + '</small></p>'
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
.code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
</style>
</head><body>
<h1>SPSS POC</h1>
""" + stats + """
<div class="card">
  <h2>Auth</h2>
  <p>Model and API key are <b>hard-coded</b> in the backend for this POC.</p>
  <p class="code">Model: openai/gpt-4o-2024-11-20</p>
  <p>Debug your key from the container: <a href="/openrouter/ping" target="_blank">/openrouter/ping</a></p>
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
  <textarea id="q" rows="3">Show top 10 values of DB2 with labels and counts.</textarea><br/><br/>
  <button id="askBtn">Ask</button>
  <h3>Answer</h3>
  <div id="answer" style="white-space:pre-wrap"></div>
  <h3>Tool calls</h3>
  <pre id="toolsPre"></pre>
  <p><b>Tip:</b> In custom SQL you can use <span class="code">label('DB2', DB2)</span> to render SPSS value labels.</p>
</div>

<script>
const base = location.origin;
const el = (id) => document.getElementById(id);
const jfmt = (x) => { try { return JSON.stringify(x, null, 2); } catch(e){ return String(x); } };

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
