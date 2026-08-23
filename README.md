# DocStruct AI

An AI-powered document intelligence and structuring platform.

Upload a collection of files. DocStruct AI extracts their contents, structures the
information, summarizes it, explains it in plain language, and answers your questions
about it — with citations pointing back to the exact page, row, or paragraph the answer
came from.

> **Status: Milestone 2 (Document Ingestion) complete. Milestone 3 has not started.**
> You can upload PDF, DOCX, TXT, CSV and XLSX files and see their contents extracted into
> structured blocks, each traceable back to its exact page, line, row or cell. Summaries,
> plain-language explanations, and Q&A are **not implemented yet** — they arrive in later
> milestones. See [Development phases](#development-phases).

---

## Table of contents

- [The problem](#the-problem)
- [MVP scope](#mvp-scope)
- [Supported file types](#supported-file-types)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Project structure](#project-structure)
- [Data flow](#data-flow)
- [How each file type is processed](#how-each-file-type-is-processed)
- [Development phases](#development-phases)
- [Getting started](#getting-started)
- [Development practices](#development-practices)
- [Security](#security)
- [Future features](#future-features)

---

## The problem

Information that people need is trapped in files. A folder of PDFs, spreadsheets, and Word
documents is searchable only by filename, and answering a question like "what is the total
of the late-payment penalties across these three contracts?" means opening each file and
reading it by hand.

DocStruct AI turns that folder into something you can ask questions of — while remaining
honest about where every answer came from.

---

## MVP scope

The MVP delivers nine capabilities:

| # | Capability | What it means |
|---|---|---|
| 1 | **Upload** | Upload one or more files, see their processing status |
| 2 | **Extract** | Pull text, tables, and structure out of each file |
| 3 | **Structure** | Normalize everything into one consistent internal representation |
| 4 | **Summarize** | Generate a concise summary of each document |
| 5 | **Explain** | Re-express a document in simple, non-expert language |
| 6 | **Ask** | Ask free-form questions about the documents |
| 7 | **Extract on request** | Pull out specific named fields ("the invoice total", "all deadlines") |
| 8 | **Cite** | Every answer points back to its source location |
| 9 | **Multi-document** | Ask a question spanning several documents at once |

**Explicitly out of scope for the MVP:** OCR and scanned documents, document comparison,
financial analysis, knowledge graphs, agents, and advanced analytics. The architecture is
designed to accommodate them later — see [Future features](#future-features).

---

## Supported file types

| Format | Extension | What is extracted |
|---|---|---|
| PDF | `.pdf` | Text per page, tables, page numbers |
| Word | `.docx` | Paragraphs, heading hierarchy, tables, lists |
| Plain text | `.txt` | Text with line numbers |
| CSV | `.csv` | Column schema, typed rows, data profile |
| Excel | `.xlsx` | Per-sheet schema, typed rows, data profile |

---

## Architecture

DocStruct AI is a **modular monolith**: one deployable backend application, internally
divided into modules with clear boundaries and interfaces.

This is a deliberate choice. Microservices would add network calls, deployment complexity,
and distributed-failure handling to a system that has none of the problems microservices
solve. The module boundaries are real and enforced; if a module ever genuinely needs to
scale independently, it can be extracted later because the interface is already there.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React + TypeScript)               │
│        Upload · Document list · Summary view · Chat + citations │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST / JSON
┌────────────────────────────▼────────────────────────────────────┐
│                       API layer (FastAPI)                       │
│      routes · request/response schemas · validation · errors    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                        Service layer                            │
│   DocumentService · AnalysisService · QAService · JobService    │
│         (orchestration — knows what happens in what order)      │
└──────┬──────────────────┬───────────────────┬───────────────────┘
       │                  │                   │
┌──────▼───────┐  ┌───────▼────────┐  ┌───────▼────────┐
│  Ingestion   │  │   Retrieval    │  │      AI        │
│              │  │                │  │                │
│ ┌──────────┐ │  │ ┌────────────┐ │  │ ┌────────────┐ │
│ │ Parsers  │ │  │ │ Retriever  │ │  │ │LLMProvider │ │
│ │ (per     │ │  │ │ interface  │ │  │ │ interface  │ │
│ │  format) │ │  │ └────────────┘ │  │ └────────────┘ │
│ └──────────┘ │  │  keyword /     │  │   Claude       │
│  normalize   │  │  vector /      │  │   (Anthropic)  │
│  chunk       │  │  hybrid        │  │                │
└──────┬───────┘  └───────┬────────┘  └───────┬────────┘
       │                  │                   │
┌──────▼──────────────────▼───────────────────▼───────────────────┐
│                      Repository layer                           │
│              (the only code that touches the DB)                │
└──────┬───────────────────────────────────┬─────────────────────-┘
       │                                   │
┌──────▼──────────────┐        ┌───────────▼────────────┐
│  PostgreSQL         │        │  File storage          │
│  metadata · blocks  │        │  (local disk; S3-      │
│  chunks · answers   │        │   ready interface)     │
└─────────────────────┘        └────────────────────────┘
```

### The three extension points

Everything about the future-proofing claim rests on three interfaces:

- **`DocumentParser`** — one implementation per file format. Adding OCR or PPTX support
  later means writing one new parser and registering it. No other code changes.
- **`LLMProvider`** — wraps the model. Swapping or adding a provider does not touch
  service code.
- **`Retriever`** — wraps the search strategy. The MVP can start with simple keyword
  search and move to vector or hybrid search without the Q&A service noticing.

### The canonical document model

The key idea that makes the whole system work: **every format is parsed into the same
shape.**

```
ParsedDocument
 └── ContentBlock[]
      ├── block_type   heading | paragraph | table | list_item | sheet_row | ...
      ├── text         the normalized text
      ├── order        position in the document
      └── provenance   WHERE THIS CAME FROM  ← makes citations possible
                       PDF   → page 4
                       DOCX  → paragraph 17
                       TXT   → lines 40-48
                       CSV   → rows 120-140
                       XLSX  → sheet "Q3", rows 12-30
```

Downstream, nothing knows or cares what format a block came from. The summarizer, the
chunker, the retriever, and the question-answerer all operate on `ContentBlock` objects.
That is why adding a file format costs one file.

Provenance is carried all the way through to the answer, which is what allows a citation to
say "page 4 of contract.pdf" instead of a vague gesture at the document.

---

## Technology stack

| Layer | Choice | Why this choice |
|---|---|---|
| **Backend language** | **Python 3.11+** | Every mature document-parsing library (PDF, DOCX, XLSX) is Python. Choosing anything else means fighting the ecosystem for the project's core capability. |
| **Web framework** | **FastAPI** | Async by default (important when waiting on the LLM), automatic OpenAPI docs, and Pydantic validation built in — request validation is a security control here, not a nicety. |
| **Validation / settings** | **Pydantic v2** | One declarative way to validate API input *and* load configuration from environment variables, with a startup error if a required secret is missing. |
| **Database** | **PostgreSQL** | Relational data (documents, blocks, chunks) with strong integrity guarantees. Also gives full-text search built in, and `pgvector` for semantic search later — so the search strategy can evolve without adding a separate search service. |
| **ORM / migrations** | **SQLAlchemy 2 + Alembic** | Explicit, typed queries and versioned schema migrations. Schema changes must be reviewable and reversible. |
| **AI model** | **Claude Opus 5** (`claude-opus-5`) via the official `anthropic` SDK | Strong document reasoning, a 1M-token context window (large documents often fit whole, no retrieval needed), and **native citation support** that returns verifiable source spans rather than model-written prose citations. |
| **PDF parsing** | **pypdf** (text) + **pdfplumber** (tables/layout) | pypdf is fast and light for straightforward text; pdfplumber handles table structure and word positions when layout matters. |
| **DOCX parsing** | **python-docx** | The standard library for the format; exposes heading styles, which gives real document structure rather than a flat text blob. |
| **XLSX parsing** | **openpyxl** | Read-only, streaming mode keeps memory bounded on large workbooks. |
| **CSV parsing** | Python standard library `csv` | No dependency needed. Handles dialect sniffing and quoting correctly. |
| **Testing** | **pytest** | The Python standard; fixtures fit parser testing well. |
| **Lint / format** | **ruff** | One fast tool replacing flake8, isort, and black. Fewer tools, fewer configs. |
| **Type checking** | **mypy** | Catches whole classes of errors before runtime. |
| **Frontend** | **React + TypeScript + Vite** | Types across the API boundary catch integration mistakes at compile time. Vite keeps the toolchain simple. |
| **Frontend styling** | **Tailwind CSS** | Fast to build with, no separate design system to maintain for an MVP. |

### Deliberate non-choices

- **No microservices** — one deployable unit, real internal boundaries.
- **No Celery / Redis / RabbitMQ initially** — FastAPI background tasks handle MVP
  processing volume. A `JobService` abstraction means moving to a real queue later is a
  contained change. (Docker is not currently installed on this machine, which is another
  reason to defer infrastructure that assumes it.)
- **No LangChain / LlamaIndex** — these frameworks add a large dependency surface and an
  abstraction layer over an API that is already simple. Calling the Anthropic SDK directly
  keeps the data flow visible and debuggable, which matters more while learning.
- **No separate vector database** — `pgvector` inside the existing PostgreSQL avoids
  running and syncing a second datastore.

---

## Project structure

```
DocStruct-AI/
├── README.md                     # this file
├── CLAUDE.md                     # persistent development instructions
├── .gitignore
├── .env.example                  # every env var, with placeholder values
│
├── backend/
│   ├── pyproject.toml            # dependencies, ruff/mypy/pytest config
│   ├── alembic/                  # database migrations
│   │
│   ├── app/
│   │   ├── main.py               # FastAPI app creation, middleware, routers
│   │   │
│   │   ├── core/                 # cross-cutting concerns
│   │   │   ├── config.py         # env-var settings (Pydantic)
│   │   │   ├── exceptions.py     # typed domain exceptions
│   │   │   ├── logging.py        # structured logging setup
│   │   │   └── security.py       # upload validation, size/type limits
│   │   │
│   │   ├── api/                  # HTTP layer — no business logic
│   │   │   ├── deps.py           # shared dependencies (db session, current user)
│   │   │   ├── errors.py         # exception -> HTTP status mapping
│   │   │   └── routes/
│   │   │       ├── documents.py  # upload, list, status, delete
│   │   │       ├── analysis.py   # summary, explain, extract fields
│   │   │       └── qa.py         # ask questions
│   │   │
│   │   ├── services/             # orchestration
│   │   │   ├── document_service.py
│   │   │   ├── analysis_service.py
│   │   │   ├── qa_service.py
│   │   │   └── job_service.py    # background processing
│   │   │
│   │   ├── ingestion/            # file -> structured blocks
│   │   │   ├── registry.py       # maps file type -> parser
│   │   │   ├── normalizer.py     # cleanup, whitespace, encoding
│   │   │   ├── chunker.py        # blocks -> retrieval chunks
│   │   │   └── parsers/
│   │   │       ├── base.py       # DocumentParser interface
│   │   │       ├── pdf_parser.py
│   │   │       ├── docx_parser.py
│   │   │       ├── txt_parser.py
│   │   │       ├── csv_parser.py
│   │   │       └── xlsx_parser.py
│   │   │
│   │   ├── ai/                   # everything LLM-related
│   │   │   ├── providers/
│   │   │   │   ├── base.py       # LLMProvider interface
│   │   │   │   └── claude.py     # Anthropic implementation
│   │   │   ├── prompts/          # versioned prompt templates
│   │   │   │   ├── summarize.py
│   │   │   │   ├── explain.py
│   │   │   │   ├── qa.py
│   │   │   │   └── extract.py
│   │   │   └── citations.py      # map model citations -> provenance
│   │   │
│   │   ├── retrieval/            # finding relevant content
│   │   │   ├── base.py           # Retriever interface
│   │   │   ├── keyword.py        # PostgreSQL full-text search
│   │   │   └── context_builder.py# assemble context within a token budget
│   │   │
│   │   ├── storage/              # file persistence
│   │   │   ├── base.py           # FileStorage interface
│   │   │   └── local.py          # local disk implementation
│   │   │
│   │   ├── repositories/         # the only code that touches the DB
│   │   │   ├── document_repo.py
│   │   │   ├── block_repo.py
│   │   │   └── chunk_repo.py
│   │   │
│   │   └── models/
│   │       ├── db/               # SQLAlchemy tables
│   │       └── domain/           # ParsedDocument, ContentBlock, Provenance
│   │
│   └── tests/
│       ├── fixtures/             # tiny synthetic sample files
│       ├── unit/
│       └── integration/
│
└── frontend/
    ├── package.json
    ├── index.html
    └── src/
        ├── main.tsx
        ├── api/                  # single typed API client
        ├── components/
        ├── pages/
        └── types/                # shared API types
```

---

## Data flow

The complete path from a file on the user's machine to a cited answer.

### Stage 1 — Upload and validation (synchronous, fast)

```
User selects file
      ↓
POST /api/documents  (multipart)
      ↓
Size check          — reject oversized files before buffering
Content sniffing    — verify actual file type from magic bytes, not the extension
Parser lookup       — is this type supported? reject early if not
      ↓
Store file under a generated UUID (original filename kept as display metadata only)
      ↓
Create Document row, status = PENDING
      ↓
Enqueue background processing job
      ↓
Respond 202 Accepted { document_id, status: "pending" }
```

The response returns immediately. Parsing a large PDF can take many seconds, and an HTTP
request should never be held open that long.

### Stage 2 — Processing (background)

```
status = PROCESSING
      ↓
Registry selects the parser for this file type
      ↓
Parser produces ParsedDocument = ordered ContentBlock[]
      each block carries text + provenance (page / paragraph / row / cell range)
      ↓
Normalizer: fix encoding, collapse whitespace, drop repeated headers/footers
      ↓
Persist blocks (with provenance) to the database
      ↓
Chunker: group blocks into retrieval-sized chunks
      respecting structure — do not split mid-table or mid-section
      each chunk records which block IDs it came from  ← the citation link
      ↓
Persist chunks (+ full-text search index)
      ↓
status = READY   (or FAILED, with a stored, user-visible reason)
```

Frontend polls `GET /api/documents/{id}` until the status leaves `PROCESSING`.

### Stage 3 — Analysis (summarize / explain)

```
POST /api/documents/{id}/summary
      ↓
Load blocks from the database
      ↓
ContextBuilder assembles them within a token budget
      Small/medium document → send the whole document (Opus 5 has a 1M-token window)
      Very large document   → summarize section by section, then combine
      ↓
LLMProvider.complete(prompt=summarize, content=..., citations_enabled=True)
      ↓
Map returned citation spans back to block provenance
      ↓
Persist and return { summary, citations[] }
```

Summaries are cached: the same document does not get re-summarized on every view.

### Stage 4 — Question answering

```
POST /api/qa   { question, document_ids[] }
      ↓
Decide the strategy by total size:
      fits in the context window  → send the documents whole (most accurate)
      too large                   → Retriever finds the most relevant chunks
      ↓
ContextBuilder assembles the selected content, each piece labelled with its source
      ↓
LLMProvider.complete(prompt=qa, context=..., citations_enabled=True)
      system prompt: answer ONLY from the provided documents;
                     if the answer is not there, say so
      ↓
Citations mapper: model citation spans → block provenance → human-readable references
      ("contract.pdf, page 4" / "budget.xlsx, sheet Q3, rows 12-30")
      ↓
Respond { answer, citations[], documents_used[] }
```

Multi-document questions use exactly this path with more than one document ID — each
piece of context is labelled with its source document, so the model can attribute
correctly and citations stay unambiguous.

### Why citations work

The chain is unbroken at every step:

```
original file → block(provenance) → chunk(block_ids) → context(labelled)
    → model citation span → block → provenance → "report.pdf, page 7"
```

No link in that chain is guessed. If provenance were dropped at any stage, citations would
become plausible-sounding fiction — which is exactly the failure this design exists to
prevent.

---

## How each file type is processed

Each parser's only job is to produce `ContentBlock` objects with accurate provenance.

### PDF (`.pdf`)

- **Library:** `pypdf` for text extraction; `pdfplumber` when table structure is needed.
- **Approach:** iterate pages; extract text per page; detect and extract tables separately
  so rows and columns survive as structure instead of collapsing into a text smear.
- **Provenance:** page number (1-indexed), plus character offsets within the page.
- **Structure detection:** infer headings from font-size and weight changes — PDF has no
  semantic heading markup, so this is a heuristic and is treated as one.
- **Known limits:** a scanned PDF is images, not text. If a page yields little or no text,
  the document is flagged `requires_ocr` and the user is told plainly. **The MVP does not
  silently return an empty result for a scanned document.** OCR arrives in Phase 6.
- **Security:** reject encrypted/password-protected PDFs with a clear message; cap page
  count.

### DOCX (`.docx`)

- **Library:** `python-docx`.
- **Approach:** walk paragraphs and tables in document order. Heading styles
  (`Heading 1`, `Heading 2`, ...) give a genuine section hierarchy — better structure than
  any other format here provides for free.
- **Provenance:** paragraph index, table index, and the enclosing section heading path.
- **Security:** DOCX is a ZIP archive — enforce decompressed-size limits (zip bombs) and
  disable external entity and remote template resolution in XML parsing.

### TXT (`.txt`)

- **Library:** standard library, with encoding detection.
- **Approach:** detect the encoding (do not assume UTF-8; a Windows-authored file is often
  cp1252, and guessing wrong produces mojibake). Split into blocks on blank lines.
  Heuristically detect headings (short standalone lines, ALL CAPS, numbered prefixes).
- **Provenance:** start and end line numbers.
- **Security:** cap file size; reject invalid byte sequences rather than replacing
  characters silently.

### CSV (`.csv`)

- **Library:** standard library `csv` with dialect sniffing.
- **Approach:** treated as **tabular data, not prose** — this is the important distinction.
  Detect the delimiter and header row; infer a column type for each column (number, date,
  text); build a data profile (row count, per-column ranges, null counts, distinct values).
- **Blocks produced:** one `schema` block describing the columns, plus `sheet_row` blocks
  batched into row groups (a single row is too small to be a useful retrieval unit).
- **Provenance:** row range, e.g. rows 120-140.
- **Why the schema block matters:** for a 50,000-row file, sending every row to the model
  is impossible and pointless. The schema and profile let the model answer structural
  questions ("what columns are here?", "what is the date range?") without reading the data,
  and the row blocks serve lookups over specific records.
- **Security:** escape leading `=`, `+`, `-`, `@` on any export path (CSV injection); cap
  row and column counts; handle malformed quoting without crashing.

### XLSX (`.xlsx`)

- **Library:** `openpyxl` in read-only, values-only mode (streams rather than loading the
  whole workbook into memory).
- **Approach:** same tabular treatment as CSV, applied per worksheet. Each sheet gets its
  own schema block and row-group blocks. Formulas are read as their computed values.
- **Provenance:** sheet name plus cell range, e.g. `Q3 Budget!A12:F30`.
- **Extra considerations:** multiple sheets mean multiple logical tables in one file;
  merged cells and header rows that do not start at row 1 are common and must be handled
  rather than assumed away.
- **Security:** ZIP archive — same decompressed-size limits as DOCX. Never evaluate
  formulas. Cap sheet, row, and column counts.

### Shared by all parsers

Every parser handles, with a clear error rather than a crash or a silent empty result:
an empty file, a corrupt file, a file whose extension lies about its content, a file that
exceeds size limits, and a file containing no extractable text.

**Adding a format costs one file.** Write a parser implementing `DocumentParser`, add one
line to `app/ingestion/registry.py`, and nothing else in the system changes — the
normalizer, chunker, storage and API all work on `ContentBlock` and never learn what a
page or a worksheet is.

---

## Development phases

Each phase produces something that runs and is tested before the next begins.

| Phase | Deliverable | Done when |
|---|---|---|
| **0. Planning** ✅ | Architecture, stack, structure, this document, `CLAUDE.md`, `.gitignore` | Approved by the project owner |
| **1. Foundation** ✅ | Backend skeleton, config from env vars, logging, error handling, health + readiness probes, database layer with migrations, document read API, React frontend talking to it, test harness | `ruff`, `mypy`, and `pytest` all green; both servers start; frontend reaches the backend |
| **2. Ingestion** ✅ | All five parsers, the canonical model, normalizer, chunker, upload endpoint, background processing | Every parser is tested against fixtures, **including provenance assertions** and failure cases |
| **3. AI core** *(not started)* | `LLMProvider` interface, Claude implementation, summarize and explain, prompt templates, citation mapping | Works end-to-end on a real document; unit tests run with a mocked provider |
| **4. Q&A** | Retrieval, context builder, single- and multi-document Q&A with citations | Answers are correct and citations resolve to the right page/row |
| **5. Frontend** | Upload UI, document list with live status, summary view, chat interface with clickable citations | A person can complete the whole flow in a browser without touching the API directly |
| **6. Hardening** | Full security review, rate limiting, observability, performance work on large files, documentation pass | Security checklist verified; large-file behaviour measured, not assumed |

Post-MVP work (OCR, comparison, knowledge graphs, agents) begins only after Phase 6.

---

## Development practices

- **One change at a time.** Small, reviewable increments over large rewrites.
- **Tests alongside code**, not after. A parser without a provenance test is not finished.
- **Run the checks before calling anything done:**

```bash
ruff format --check .
ruff check .
mypy backend/app
pytest -q
```

- **Never mock what you are testing**; mock what it depends on. Parser tests use real
  fixture files. Service tests use a fake LLM provider.
- **Never call the real Claude API in automated tests** — slow, costly, non-deterministic.
- **Every explanation covers**: what changed, why, how it works, which files, how it was
  verified. See `CLAUDE.md` §3.

---

## Security

Security considerations are built in from Phase 1, not retrofitted in Phase 6.

| Risk | Mitigation |
|---|---|
| **Malicious upload** | Validate by content sniffing, not file extension; enforce size limits before buffering; allowlist file types |
| **Path traversal** | Store under generated UUIDs; never build a filesystem path from a user-supplied filename |
| **Zip bombs** (DOCX/XLSX are ZIP archives) | Enforce decompressed-size and entry-count limits |
| **XXE / XML external entities** | Disable external entity and remote template resolution in all XML parsing |
| **CSV injection** | Escape cells beginning with `=`, `+`, `-`, `@` on export |
| **Prompt injection via document content** | Treat all document text as untrusted; keep it clearly delimited from instructions; never let document content grant capability |
| **Secret leakage** | Environment variables only; `.env` gitignored; never log keys; startup fails loudly if a required secret is missing |
| **Data leakage between users** | Every repository read is scoped by owner — enforced at the repository layer, not per-route |
| **Resource exhaustion** | Limits on file size, page count, row count, and concurrent processing jobs |
| **Sensitive data in logs** | Log identifiers and metadata only — never document content |
| **Denial of wallet** (LLM cost abuse) | Rate limiting per user; token budgets per request; cached summaries |

---

## Future features

The architecture accommodates these without a rewrite:

| Feature | How it fits |
|---|---|
| **OCR / scanned documents** | A new `DocumentParser` implementation, or an OCR fallback inside the PDF parser. The `requires_ocr` flag is already in the Phase 2 design. |
| **More file formats** (PPTX, HTML, EML, images) | One new parser per format, plus a registry entry. |
| **Semantic search** | A new `Retriever` implementation using `pgvector`. The Q&A service is unchanged. |
| **Document comparison** | A new service operating on the existing `ContentBlock` structure. |
| **Financial analysis** | Builds on the typed schema and profile the CSV/XLSX parsers already produce. |
| **Knowledge graphs** | An entity-extraction pass over the stored blocks; provenance is already attached. |
| **AI agents** | The existing services become the agent's tools. |
| **Advanced analytics** | New service on existing structured data. |

None of these are being built now. They are listed to demonstrate that the boundaries were
drawn with them in mind.

---

## Getting started

### Prerequisites

* **Python 3.11+** (3.13 tested)
* **Node.js 20+** (24 tested)
* **PostgreSQL** — optional. The project defaults to SQLite so it runs with no database
  server installed. See [Database](#database) below.

### 1. Configure

```bash
cp .env.example .env
```

The defaults work as-is. `.env` is gitignored and must never be committed.

### 2. Backend

```bash
cd backend

python -m venv .venv                  # first time only
source .venv/Scripts/activate         # Windows (Git Bash)
# .venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source .venv/bin/activate           # macOS / Linux

pip install -e ".[dev]"               # first time only
alembic upgrade head                  # create/update the database schema
uvicorn app.main:app --reload
```

The API is now on **http://127.0.0.1:8000**:

| URL | What it is |
|---|---|
| `/api/health` | Liveness — is the process up? |
| `/api/ready` | Readiness — are its dependencies reachable? |
| `POST /api/documents` | Upload a file (returns 202; processing runs in the background) |
| `/api/documents` | Document list with processing status |
| `/api/documents/{id}/content` | Extracted content blocks, each with its source location |
| `/docs` | Interactive API documentation (Swagger UI) |
| `/redoc` | The same API reference in ReDoc |

Every failing request returns the same envelope, so the frontend has one error path:

```json
{
  "error": {
    "code": "not_found",
    "message": "Document not found",
    "details": {},
    "request_id": "9f1c2d3e4a5b6c7d8e9f0a1b2c3d4e5f"
  }
}
```

Branch on `code` (stable), not `message` (may be reworded). `request_id` also comes back
in the `X-Request-ID` header on **every** response — quote it when reporting a problem,
because the matching server log entry holds the full technical detail. Server-side
failures return a generic message on purpose: stack traces and internal details go to the
log, never to the client.

### 3. Frontend

In a second terminal:

```bash
cd frontend
npm install                           # first time only
npm run dev
```

Open **http://localhost:5173** (use `localhost`, not `127.0.0.1` — on Windows Vite binds
only to `localhost`). You can drag files onto the page, watch each one move from
*Waiting* to *Processing* to *Ready*, and click **View content** to see the extracted
blocks with the page, line, row or cell each came from.

The Vite dev server proxies `/api` to the backend, so the browser talks to a single origin
and no CORS round-trip is involved in development.

### Running the checks

Everything below must pass before a task is considered complete.

```bash
# Backend (from backend/, with the virtualenv active)
ruff format --check .     # formatting
ruff check .              # linting
mypy app                  # type checking
pytest                    # tests

# Frontend (from frontend/)
npm run typecheck         # tsc --noEmit
npm run lint              # eslint
npm run build             # production build
```

### Database

The `DATABASE_URL` environment variable selects the backend. Both are supported by the
same models and the same migrations.

```bash
# SQLite — the zero-setup default; nothing to install
DATABASE_URL=sqlite+aiosqlite:///./docstruct.db

# PostgreSQL — the production target
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/docstruct
```

Migrations are managed with Alembic, run from `backend/`:

```bash
alembic upgrade head                              # apply all migrations
alembic downgrade -1                              # roll back one
alembic revision --autogenerate -m "add x table"  # after changing a model
alembic current                                   # what is applied now
```

After changing any model in `app/models/db/`, generate a migration. A test
(`tests/integration/test_migrations.py`) fails if models and migrations drift apart.
