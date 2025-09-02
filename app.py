from fastapi.responses import HTMLResponse

import os, json, tempfile, logging, math
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ------------------- storage paths & constants -------------------
DB_PATH   = "data/spss.duckdb"
META_PATH = "data/metadata.json"
TABLE     = "data"
BATCH     = 50_000

Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(META_PATH).parent.mkdir(parents=True, exist_ok=True)

# ------------------- app -------------------
app = FastAPI(title="SPSS POC")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # POC; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- OpenRouter (hardcoded) -------------------
# You asked to hardcode these:
HARDCODED_MODEL = "openai/gpt-4o-2024-11-20"
HARDCODED_KEY   = "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"

OPENROUTER = {"client": None, "model": HARDCODED_MODEL, "error": None}

def init_openrouter_client():
    try:
        from openai import OpenAI  # openai>=1.x
        import httpx
        http_client = httpx.Client(timeout=60.0)  # respects env proxies automatically
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=HARDCODED_KEY,
            http_client=http_client,
            default_headers={
                "HTTP-Referer": "https://example.com",
                "X-Title": "SPSS POC",
            },
        )
        return client
    except Exception as e:
        logging.exception("Failed to init OpenRouter: %s", e)
        return None

OPENROUTER["client"] = init_openrouter_client()

# ------------------- utilities -------------------
def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")

def qi(name: str) -> str:
    # DuckDB identifier quoting
    return '"' + str(name).replace('"', '""') + '"'

def load_meta():
    try:
        if not os.path.exists(META_PATH):
            return {}
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_meta(m: dict):
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error("Failed to write metadata: %s", e)

def _duck_label(colname, x):
    """Python UDF used inside DuckDB SQL: label('DB2', DB2) -> value label text."""
    meta = load_meta()
    vmap = meta.get("value_labels", {}).get(colname)
    if not vmap:
        return x
    return vmap.get(str(x), x)

def con():
    import duckdb
    c = duckdb.connect(DB_PATH)
    # register UDF on each connection
    try:
        c.create_function("label", _duck_label)
    except Exception:
        pass
    return c

def _ensure_select(sql: str) -> str:
    s = sql.strip().lower()
    if not s.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    if " limit " not in s:
        sql = sql.strip() + " LIMIT 1000"
    return sql

# ------------------- endpoints: ingest / schema / sql -------------------
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
    c.execute(f"CREATE TABLE IF NOT EXISTS {qi(TABLE)} AS SELECT 1 AS _init LIMIT 0;")
    c.execute(f"DELETE FROM {qi(TABLE)};")

    # chunked load
    offset = 0
    total = 0
    while True:
        df, _ = pyreadstat.read_sav(
            sav_path,
            row_offset=offset,
            row_limit=BATCH,
            apply_value_formats=False  # keep raw coded values
        )
        if df.empty:
            break
        df = df.rename(columns=name_map)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string")
        if total == 0:
            c.register("chunk", df)
            c.execute(f"CREATE OR REPLACE TABLE {qi(TABLE)} AS SELECT * FROM chunk")
            c.unregister("chunk")
        else:
            c.register("chunk", df)
            c.execute(f"INSERT INTO {qi(TABLE)} SELECT * FROM chunk")
            c.unregister("chunk")
        n = len(df)
        total += n
        offset += n
        if n < BATCH:
            break

    # capture labels
    value_labels = {name_map.get(k, k): v for k, v in meta.variable_value_labels.items()}
    col_labels   = {name_map.get(k, k): v for k, v in meta.column_names_to_labels.items()}
    save_meta({
        "rows": total,
        "columns": col_names,
        "column_labels": col_labels,
        "value_labels": value_labels,
        "source_filename": file.filename
    })

    try:
        os.remove(sav_path)
    except Exception:
        pass

    return {"status": "ok", "rows": total, "columns": col_names}

@app.get("/schema")
def schema():
    c = con()
    info = c.execute(f"PRAGMA table_info({qi(TABLE)})").fetchall()
    meta = load_meta()
    cols = [{"name": i[1], "type": i[2], "label": meta.get("column_labels", {}).get(i[1], "")} for i in info]
    return {"table": TABLE, "columns": cols, "rows": meta.get("rows"), "source": meta.get("source_filename")}

class SqlIn(BaseModel):
    sql: str

def _run_sql_impl(sql: str):
    df = con().execute(_ensure_select(sql)).fetchdf()
    return {"rows": json.loads(df.to_json(orient="records"))}

@app.post("/sql")
def sql(q: SqlIn):
    try:
        return _run_sql_impl(q.sql)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# ------------------- simple tools (schema, value_counts, describe) -------------------
def t_schema():
    return schema()

def t_value_counts(column: str, top: int = 20):
    c = con()
    df = c.execute(
        f"""
        SELECT label({qi(column)!r}, {qi(column)}) AS value, COUNT(*) AS count
        FROM {qi(TABLE)}
        GROUP BY 1
        ORDER BY 2 DESC NULLS LAST
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
    expr = ", ".join([f"min({qi(cn)}) as {cn}_min, max({qi(cn)}) as {cn}_max, avg({qi(cn)}) as {cn}_mean" for cn in num_cols])
    df = c.execute(f"SELECT {expr} FROM {qi(TABLE)}").fetchdf()
    return {"columns": num_cols, "summary_row": json.loads(df.to_json(orient='records'))[0]}

# ------------------- pandas aggregate tool (no SQL required) -------------------
import pandas as pd

def _load_df_for(cols: list[str]) -> pd.DataFrame:
    if not cols:
        cols = ["*"]
    col_list = ", ".join(qi(c) for c in cols) if cols != ["*"] else "*"
    return con().execute(f"SELECT {col_list} FROM {qi(TABLE)}").df()

def _apply_labels(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    meta = load_meta()
    vmap_all = meta.get("value_labels", {})
    if not vmap_all:
        return df
    df = df.copy()
    for col in cols:
        if col in df.columns and col in vmap_all:
            mapping = {str(k): v for k, v in vmap_all[col].items()}
            df[col] = df[col].map(lambda x: mapping.get(str(x), x))
    return df

def _filter_df(df: pd.DataFrame, flt: dict | None) -> pd.DataFrame:
    if not flt:
        return df
    mask = pd.Series(True, index=df.index)
    for col, rule in flt.items():
        if col not in df.columns:
            continue
        s = df[col]
        if isinstance(rule, dict):
            if "in" in rule:     mask &= s.astype(str).isin([str(x) for x in rule["in"]])
            if "not_in" in rule: mask &= ~s.astype(str).isin([str(x) for x in rule["not_in"]])
            if "eq" in rule:     mask &= s == rule["eq"]
            if "ne" in rule:     mask &= s != rule["ne"]
            if "gt" in rule:     mask &= s > rule["gt"]
            if "gte" in rule:    mask &= s >= rule["gte"]
            if "lt" in rule:     mask &= s < rule["lt"]
            if "lte" in rule:    mask &= s <= rule["lte"]
        else:
            if isinstance(rule, (list, tuple, set)):
                mask &= s.astype(str).isin([str(x) for x in rule])
            else:
                mask &= s == rule
    return df[mask]

def t_aggregate(
    groupby: list[str],
    metrics: list[dict],
    filter: dict | None = None,
    weight: str | None = None,
    use_labels: bool = True,
    top: int | None = None,
    dropna: bool = True,
):
    """
    metrics examples:
      [{"op":"count"}]
      [{"op":"mean","col":"Q1"}]
      [{"op":"pct","col":"Q5","where":{"Q5":[5]}}]
      [{"op":"sum","col":"Spend"}]
    Supported ops: "count", "sum", "mean", "pct"
    """
    groupby = groupby or []
    metric_cols = [m.get("col") for m in metrics if m.get("col")]
    needed_cols = set(groupby + ([weight] if weight else []) + [c for c in metric_cols if c])

    if filter:
        for k in filter.keys():
            needed_cols.add(k)
    for m in metrics:
        if m.get("op") == "pct" and isinstance(m.get("where"), dict):
            for k in m["where"].keys():
                needed_cols.add(k)

    df = _load_df_for(sorted(needed_cols) if needed_cols else ["*"])

    if dropna and groupby:
        df = df.dropna(subset=groupby)

    df = _filter_df(df, filter)

    if use_labels and groupby:
        df = _apply_labels(df, groupby)

    w = None
    if weight and weight in df.columns:
        w = df[weight].astype(float)
        w = w.where(w.notna() & (w >= 0), 0.0)

    # build groups
    if groupby:
        groups = df.groupby(groupby, dropna=False, sort=False)
    else:
        groups = [ ((), df) ]

    out_rows = []
    for keys, gdf in (groups if isinstance(groups, list) else groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base_n = int(len(gdf))
        base_w = float(gdf[weight].astype(float).where(lambda s: s.notna() & (s>=0), 0.0).sum()) if w is not None else None

        row = {}
        for i, col in enumerate(groupby):
            row[col] = keys[i]

        for m in metrics:
            op = m.get("op")
            col = m.get("col")
            name = m.get("name") or (op if not col else f"{op}_{col}")

            if op == "count":
                val = base_w if w is not None else base_n

            elif op == "sum":
                s = gdf[col].astype(float)
                if w is None:
                    val = float(s.sum(skipna=True))
                else:
                    val = float((s.fillna(0.0) * gdf[weight].astype(float).fillna(0.0)).sum())

            elif op == "mean":
                s = gdf[col].astype(float)
                if w is None:
                    val = float(s.mean(skipna=True))
                else:
                    ww = gdf[weight].astype(float).fillna(0.0)
                    num = float((s.fillna(0.0) * ww).sum())
                    den = float(ww.where(s.notna(), 0.0).sum())
                    val = float(num / den) if den > 0 else math.nan

            elif op == "pct":
                sub = _filter_df(gdf, m.get("where"))
                if w is None:
                    den = len(gdf)
                    num = len(sub)
                else:
                    den = float(gdf[weight].astype(float).fillna(0.0).sum())
                    num = float(sub[weight].astype(float).fillna(0.0).sum())
                val = float(100.0 * num / den) if den > 0 else math.nan

            else:
                val = None

            row[name] = val

        row["_base_n"] = base_n
        if base_w is not None:
            row["_base_w"] = base_w

        out_rows.append(row)

    if top and groupby and metrics:
        first_metric = metrics[0].get("name") or (metrics[0]["op"] if not metrics[0].get("col") else f"{metrics[0]['op']}_{metrics[0]['col']}")
        out_rows = sorted(
            out_rows,
            key=lambda r: (r.get(first_metric) is None, r.get(first_metric)),
            reverse=True
        )[:top]

    return {"rows": out_rows, "groupby": groupby, "metrics": metrics}

# ------------------- LLM tool wiring -------------------
TOOLS = [
  {"type":"function","function":{"name":"get_schema","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"value_counts","parameters":{"type":"object","properties":{"column":{"type":"string"},"top":{"type":"integer"}},"required":["column"]}}},
  {"type":"function","function":{"name":"describe","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"run_sql","parameters":{"type":"object","properties":{"sql":{"type":"string"}},"required":["sql"]}}},

  # pandas aggregate tool
  {"type":"function","function":{
    "name":"aggregate",
    "parameters":{
      "type":"object",
      "properties":{
        "groupby":{"type":"array","items":{"type":"string"}},
        "metrics":{"type":"array","items":{"type":"object"}},
        "filter":{"type":"object"},
        "weight":{"type":"string"},
        "use_labels":{"type":"boolean"},
        "top":{"type":"integer"},
        "dropna":{"type":"boolean"}
      },
      "required":["groupby","metrics"]
    }
  }},
]

NAME_TO_TOOL = {
  "get_schema": lambda **kw: t_schema(),
  "value_counts": lambda **kw: t_value_counts(**kw),
  "describe":    lambda **kw: t_describe(),
  "run_sql":     lambda **kw: _run_sql_impl(kw["sql"]),
  "aggregate":   lambda **kw: t_aggregate(**kw),
}

# ------------------- /ask endpoint -------------------
class AskIn(BaseModel):
    question: str

@app.post("/ask")
def ask(body: AskIn, request: Request):
    if not OPENROUTER["client"]:
        return JSONResponse({"error":"OpenRouter client not initialized."}, status_code=400)

    try:
        msgs = [
            {
                "role":"system",
                "content":(
                  "You are a market-research data analyst. "
                  "Prefer the `aggregate` tool for counts, %, weighted means, and grouped rollups. "
                  "Use `value_counts` for quick 1D counts. Use `describe` for numeric summaries. "
                  "Avoid writing SQL unless the user explicitly asks for SQL. "
                  "When possible, return compact, readable tables and one-sentence summaries."
                )
            },
            {"role":"user","content": body.question}
        ]

        client = OPENROUTER["client"]; model = OPENROUTER["model"]

        # turn 1: let model pick tools
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
            # append assistant with tool_calls
            norm_tool_calls = []
            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None) if hasattr(tc, "function") else tc.get("function")
                name = getattr(fn, "name", None) if hasattr(fn, "name") else (fn.get("name") if fn else None)
                arguments = getattr(fn, "arguments", None) if hasattr(fn, "arguments") else (fn.get("arguments") if fn else None)
                tc_id = getattr(tc, "id", None) if hasattr(tc, "id") else tc.get("id")
                norm_tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments or "{}"},
                })

            msgs.append({"role":"assistant","content":None,"tool_calls":norm_tool_calls})

            # run tools
            for tc in norm_tool_calls:
                name = tc["function"]["name"]
                args_raw = tc["function"]["arguments"]
                try:
                    args = _json.loads(args_raw or "{}")
                except Exception:
                    args = {}
                fn = NAME_TO_TOOL.get(name)
                try:
                    result = fn(**args) if fn else {"error": "unknown tool: " + str(name)}
                except Exception as e:
                    logging.exception("Tool %s failed", name)
                    result = {"error": str(e)}
                tool_results.append({"tool": name, "args": args, "result": result})
                msgs.append({
                    "role":"tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": _json.dumps(result)
                })

            # turn 2: final answer
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
        logging.error("ASK endpoint failed: %s", e, exc_info=True)
        return JSONResponse({"error": f"/ask failed: {e}"}, status_code=500)

# ------------------- tiny UI -------------------
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
</style>
</head><body>
<h1>SPSS POC</h1>
<p><small>Model & key are hardcoded in server code.</small></p>
""" + stats + """
<div class="card">
  <h2>1) Upload .sav</h2>
  <input id="file" type="file" accept=".sav" />
  <p><small>POC: keep it modest in size.</small></p>
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
  <textarea id="q" rows="3">List out the counts of DB2 (use labels).</textarea><br/><br/>
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
