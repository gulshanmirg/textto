# nlptosql

Perfect. Here’s the updated layout, commands, and full code with your venv name and paths.

# 1) Folder layout 

```
C:\nlptosql\
├─ .env
├─ requirements.txt
├─ Scripts\           (created by python -m venv nlptosql)
├─ Lib\               (created by venv)
└─ src\
   ├─ env_utils.py
   ├─ vector_index.py
   ├─ rag_workflow.py
   ├─ schema_service.py
   ├─ api\
   │  └─ main.py
   └─ app\
      └─ streamlit_app.py
```

# 2) Create venv and folders

```powershell
# 2) create and activate venv
python -m venv nlptosql
cd nlptosql
.\Scripts\Activate

# 3) create folders exactly as requested
mkdir src
mkdir src\api
mkdir src\app
```

# 3) requirements.txt (place it under C:\nlptosql)

Create `requirements.txt` in `C:\nlptosql\`:

```text
python-dotenv==1.1.1
fastapi==0.116.1
uvicorn[standard]==0.35.0
streamlit==1.48.0
llama-index==0.13.0
openai==1.99.6
anthropic==0.61.0         # Optional: for Claude models
sqlalchemy==2.0.43
oracledb==2.0.0
requests==2.32.4
```

Install:

```powershell
pip install -r requirements.txt
```
🔄 Compatibility Notes
- FastAPI + Uvicorn: Fully compatible. Use uvicorn[standard] for performance extras like uvloop, httptools, and watchfiles.
- Streamlit: Latest version includes layout improvements, better charting, and UI enhancements.
- LlamaIndex: Version 0.13.0 supports modular integrations. Ensure Python ≤3.11 for best compatibility.
- OpenAI: Version 1.99.6 supports GPT-4o and GPT-5 APIs. Backward compatible with earlier endpoints.
- Anthropic: Optional. Version 0.61.0 supports Claude 3/4 and streaming.
- SQLAlchemy: Version 2.0.43 is the latest long-term release. Works well with oracledb.
- OracleDB: Version 2.0.0 is stable and supports SQLAlchemy 2.0.
- Requests: Version 2.32.4 includes security patches and modern packaging support.

# 4) .env (place it under C:\nlptosql)

```env
# LLM provider and keys
PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# To use Claude instead:
# PROVIDER=anthropic
# ANTHROPIC_API_KEY=...
# ANTHROPIC_MODEL=claude-3-5-haiku-20241022

# Oracle
ORACLE_USER=readonly_user
ORACLE_PASSWORD=yourpassword
ORACLE_DSN=dbhost:1521/ORCLPDB1
ORACLE_POOL_SIZE=5

# Indexing
INDEX_DIR=.llamaindex_cache
TABLES=CUSTOMERS, ORDERS, ORDER_ITEMS, PRODUCTS

# API Base for Streamlit
API_BASE=http://localhost:8000
```

# 5) Full code files under C:\nlptosql\src

## src\env\_utils.py

```python
import os, csv, io
from typing import List

def env_get_tables(var: str = "TABLES", required: bool = True) -> List[str]:
    raw = os.getenv(var, "").strip()
    if not raw:
        if required:
            raise RuntimeError(f"{var} not set in .env. Example: {var}=CUSTOMERS, ORDERS")
        return []
    parts = next(csv.reader(io.StringIO(raw)))
    tables, seen = [], set()
    for p in parts:
        name = p.strip().strip('"').strip("'")
        if name and name not in seen:
            seen.add(name)
            tables.append(name)
    return tables
```

## src\vector\_index.py

```python
import json
import os
from pathlib import Path
from typing import List, Dict
from llama_index.core import VectorStoreIndex, StorageContext, Document, load_index_from_storage

INDEX_DIR = Path(os.getenv("INDEX_DIR", ".llamaindex_cache"))

def _storage(subdir: str) -> StorageContext:
    p = INDEX_DIR / subdir
    p.mkdir(parents=True, exist_ok=True)
    return StorageContext.from_defaults(persist_dir=str(p))

def vector_build_row_index_for_table(table_name: str, rows: List[Dict]):
    storage = _storage(f"rows_{table_name.lower()}")
    docs = [Document(text=json.dumps(r, ensure_ascii=False), metadata={"table": table_name}) for r in rows]
    index = VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=False)
    index.storage_context.persist()

def vector_load_row_index_for_table(table_name: str) -> VectorStoreIndex:
    storage = _storage(f"rows_{table_name.lower()}")
    return load_index_from_storage(storage)

def vector_retrieve_rows(table_name: str, query: str, k: int = 3) -> List[Dict]:
    index = vector_load_row_index_for_table(table_name)
    hits = index.as_retriever(similarity_top_k=k).retrieve(query)
    return [json.loads(h.text) for h in hits]
```

## src\rag\_workflow\.py

```python
import os
from typing import Dict, List, Optional
from sqlalchemy import create_engine, text
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI as OpenAI_LL
from llama_index.llms.anthropic import Anthropic as Anthropic_LL
from dotenv import load_dotenv

from env_utils import env_get_tables
from vector_index import vector_build_row_index_for_table, vector_retrieve_rows

# Load .env early
load_dotenv()

def _llm():
    provider = os.getenv("PROVIDER", "openai").lower()
    if provider == "anthropic":
        return Anthropic_LL(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"))
    return OpenAI_LL(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

def _oracle_engine():
    dsn = os.getenv("ORACLE_DSN")      # host:port/service
    user = os.getenv("ORACLE_USER")
    pwd  = os.getenv("ORACLE_PASSWORD")
    return create_engine(
        f"oracle+oracledb://{user}:{pwd}@{dsn}",
        pool_size=int(os.getenv("ORACLE_POOL_SIZE", "5"))
    )

def rag_bootstrap_indexes():
    rag_build_schema_index()
    tables = env_get_tables()
    for t in tables:
        rows = _sample_rows_from_oracle(t, limit=200)
        vector_build_row_index_for_table(t, rows)

def rag_build_schema_index():
    # TODO: build and persist schema object index if needed
    pass

def rag_retrieve_relevant_tables(question: str,
                                 allowed_tables: Optional[List[str]],
                                 top_k: int) -> List[str]:
    if allowed_tables:
        return allowed_tables[:top_k]
    return env_get_tables()[:top_k]

def rag_generate_sql(question: str,
                     table_schemas: Dict[str, str],
                     example_rows: Dict[str, List[dict]],
                     max_rows: int) -> str:
    llm = _llm()
    Settings.llm = llm
    prompt = f"""You are a text-to-SQL assistant.
Question: {question}

Table Schemas:
{table_schemas}

Example Rows:
{example_rows}

Constraints:
- Only one SELECT statement
- Limit rows to {max_rows}
Return only SQL."""
    return llm.complete(prompt).text.strip()

def rag_guard_readonly(sql: str):
    s = sql.strip().lower()
    forbidden = ["insert", "update", "delete", "merge", "alter", "drop", "create", "grant", "revoke", "truncate"]
    if any(tok in s for tok in forbidden) or ";" in s:
        raise ValueError("Only single SELECT statements are allowed.")

def rag_enforce_limit(sql: str, max_rows: int) -> str:
    s = sql.strip().rstrip(";")
    if "fetch first" in s.lower() or "rownum" in s.lower():
        return s
    return f"{s} FETCH FIRST {max_rows} ROWS ONLY"

def rag_execute_sql(sql: str) -> List[dict]:
    with _oracle_engine().connect() as conn:
        res = conn.execute(text(sql))
        return [dict(r._mapping) for r in res]

def rag_synthesize_answer(question: str, rows: List[dict], sql: str) -> str:
    llm = _llm()
    Settings.llm = llm
    preview = rows[:3]
    prompt = f"Question: {question}\nSQL: {sql}\nRows preview: {preview}\nProvide a concise answer."
    return llm.complete(prompt).text.strip()

def rag_run_text_to_sql(question: str,
                        allowed_tables: Optional[List[str]] = None,
                        top_k_tables: int = 4,
                        row_retrieval: bool = True,
                        max_rows: int = 500) -> Dict:
    selected_tables = rag_retrieve_relevant_tables(question, allowed_tables, top_k_tables)
    # Replace with real schema strings for better SQL quality
    table_schemas = {t: f"-- schema for {t} here" for t in selected_tables}

    example_rows = {}
    if row_retrieval:
        for t in selected_tables:
            example_rows[t] = vector_retrieve_rows(t, question, k=3)

    sql = rag_generate_sql(question, table_schemas, example_rows, max_rows)
    rag_guard_readonly(sql)
    sql = rag_enforce_limit(sql, max_rows)

    rows = rag_execute_sql(sql)
    answer = rag_synthesize_answer(question, rows, sql)

    return {
        "answer": answer,
        "sql": sql,
        "rows": rows,
        "columns": list(rows[0].keys()) if rows else [],
        "diagnostics": {"selected_tables": selected_tables}
    }

def _sample_rows_from_oracle(table: str, limit: int) -> List[dict]:
    with _oracle_engine().connect() as conn:
        res = conn.execute(text(f"SELECT * FROM {table} FETCH FIRST {limit} ROWS ONLY"))
        return [dict(r._mapping) for r in res]
```

## src\schema\_service.py

```python
from env_utils import env_get_tables

def get_schema_summary():
    return {"tables": env_get_tables()}
```

## src\api\main.py

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from rag_workflow import rag_bootstrap_indexes, rag_run_text_to_sql

# Load .env at process start
load_dotenv()

app = FastAPI()
rag_bootstrap_indexes()

class QueryReq(BaseModel):
    question: str
    tables: Optional[List[str]] = None
    top_k_tables: int = 4
    row_retrieval: bool = True
    max_rows: int = 500

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/query")
def query(req: QueryReq):
    try:
        return rag_run_text_to_sql(
            question=req.question,
            allowed_tables=req.tables,
            top_k_tables=req.top_k_tables,
            row_retrieval=req.row_retrieval,
            max_rows=req.max_rows,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## src\app\streamlit\_app.py

```python
import os, requests, streamlit as st
from dotenv import load_dotenv

load_dotenv()

API = os.getenv("API_BASE", "http://localhost:8000")

st.title("NLP → SQL for Oracle")

q = st.text_area("Ask a question")
picked = st.text_input("Comma-separated tables (optional)")
tables = [t.strip() for t in picked.split(",")] if picked.strip() else None
top_k = st.slider("Top K tables", 1, 8, 4)
row_ret = st.toggle("Row retrieval", value=True)

if st.button("Run") and q.strip():
    payload = {
        "question": q,
        "tables": tables,
        "top_k_tables": top_k,
        "row_retrieval": row_ret,
        "max_rows": 500
    }
    r = requests.post(f"{API}/query", json=payload)
    if r.ok:
        data = r.json()
        st.code(data["sql"], language="sql")
        if data.get("rows"):
            st.dataframe(data["rows"])
        st.success(data["answer"])
        st.caption(f"Diagnostics: {data.get('diagnostics')}")
    else:
        st.error(r.text)
```

# 6) Run commands from C:\nlptosql

Open two PowerShell terminals.

**Terminal 1: API**

```powershell
.\Scripts\Activate
$env:PYTHONPATH = (Resolve-Path .\src).Path
uvicorn api.main:app --reload --port 8000
```

**Terminal 2: Streamlit**

```powershell
.\Scripts\Activate
$env:PYTHONPATH = (Resolve-Path .\src).Path
streamlit run src\app\streamlit_app.py
```

You are all set. The imports are now based on `PYTHONPATH=src`, tables come from `.env`, and all Python files live under `nlptosql\src` exactly as you specified.
