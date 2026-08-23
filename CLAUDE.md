# CLAUDE.md — Development Instructions for DocStruct AI

This file holds persistent instructions for AI-assisted development on this project.
Read it before making changes. Keep it updated when the architecture changes.

---

## 1. What this project is

DocStruct AI is an AI-powered document intelligence platform. Users upload collections of
files (PDF, DOCX, TXT, CSV, XLSX); the system extracts their contents, structures them,
summarizes and explains them, and answers questions about them with source citations.

**Status: Milestone 2 (Document Ingestion) complete. Milestone 3 has NOT started.**
Files can be uploaded and are parsed into structured, traceable content. No AI, LLM,
embeddings, retrieval, OCR, or authentication code exists yet — those belong to later
milestones.

---

## 2. Non-negotiable rules

These override convenience. Do not violate them without explicit approval from the user.

1. **Never hardcode secrets.** API keys, database URLs, and credentials come from
   environment variables only, loaded through `backend/app/core/config.py`. Never write a
   key into source, tests, fixtures, logs, or documentation. `.env` is gitignored;
   `.env.example` documents every variable with a placeholder value.
2. **Never silently ignore errors.** No bare `except:`, no `except Exception: pass`, no
   swallowed promise rejections. Every failure is either handled meaningfully, converted
   into a typed domain error, or propagated. Log with context.
   **Corollary — safe outward, complete inward:** a 5xx response carries only a generic
   message plus a `request_id`; the traceback and any internal detail go to the log under
   that same ID. Never put an exception type, stack trace, file path, connection string,
   or rejected input value into an HTTP response. See `app/api/errors.py`.
3. **Never claim a feature works without running it.** "Tests pass" means you ran them and
   saw them pass. If you could not test something, say so explicitly.
4. **Fix failures before declaring a task complete.** A red test suite is not "done".
5. **Do not modify unrelated files.** Keep diffs scoped to the task.
6. **Do not add dependencies casually.** Every new dependency needs a stated justification
   and user approval. Prefer the standard library. Prefer one library that does the job
   over three that overlap.
7. **Do not introduce microservices, message brokers, or Kubernetes** unless a concrete,
   measured need appears. This is a modular monolith by design.
8. **Do not implement future-phase features early.** OCR, document comparison, knowledge
   graphs, agents, and advanced analytics are explicitly out of scope for the MVP.

---

## 3. Explaining your work

The user is learning while building this project. For every significant change, explain:

1. **What** you are changing.
2. **Why** it is needed.
3. **How** it works — the mechanism, not just the outcome.
4. **Which files** are affected.
5. **How you verified it** — the exact commands run and their result.

Prefer plain language over jargon. When you use a term of art (embedding, chunking,
provenance, idempotency), define it the first time it appears in the explanation.

---

## 4. Architecture rules

### 4.1 Layering

Dependencies flow in one direction only:

```
api  ->  services  ->  {ingestion, ai, retrieval}  ->  repositories  ->  models
```

- `api/` — HTTP concerns only: routing, request/response schemas, auth, status codes.
  No business logic, no direct database queries, no AI calls.
- `services/` — orchestration and business rules. Knows *what* happens in what order.
- `ingestion/`, `ai/`, `retrieval/` — capability modules behind interfaces.
- `repositories/` — all database access. Nothing else touches the ORM session.
- `models/` — persistence models and domain dataclasses.

A lower layer must never import an upper layer. If you feel the need to, the abstraction
is wrong — raise it rather than working around it.

### 4.2 Extension points (the reason the project is structured this way)

Three interfaces exist so that new capability can be added without editing existing code:

| Interface | Location | Adding a new implementation means |
|---|---|---|
| `DocumentParser` | `ingestion/parsers/base.py` | Support a new file type — write one parser, register it. Nothing else changes. |
| `LLMProvider` | `ai/providers/base.py` | Swap or add a model provider without touching services. |
| `Retriever` | `retrieval/base.py` | Change the search strategy (keyword to vector to hybrid) without touching the Q&A service. |

**When you add a file format, you write exactly one new file plus one registry line.**
If supporting a new format requires editing the pipeline, the abstraction has leaked —
fix the abstraction, do not special-case the format.

### 4.3 The canonical document model

Every parser, regardless of format, produces the same output: a `ParsedDocument`
containing an ordered list of `ContentBlock` objects. Each block carries:

- `block_type` — heading, paragraph, table, list_item, sheet_row, etc.
- `text` — the normalized text content.
- `order` — position within the document.
- `provenance` — **where this came from in the original file.** Page number for PDF,
  paragraph index for DOCX, line range for TXT, row range for CSV, sheet plus cell range
  for XLSX.

**Provenance is mandatory, not optional.** Citations are an MVP requirement, and a
citation is only possible if every piece of text can be traced back to its origin. A
parser that drops provenance is a broken parser. Never invent, guess, or approximate a
provenance value — if the source location is genuinely unknown, the field must record that
it is unknown rather than hold a plausible-looking number.

---

## 5. AI / LLM rules

- **Provider:** Anthropic Claude via the official `anthropic` Python SDK. Never use an
  OpenAI-compatible shim or raw HTTP.
- **Model:** `claude-opus-5` (the default; do not downgrade for cost without being asked).
- **Thinking:** use `thinking={"type": "adaptive"}`. Do not use `budget_tokens` — it is
  removed on this model family and returns a 400.
- **Streaming:** use streaming for anything with long input or a high `max_tokens`.
- **Citations:** prefer the API's native citation support (`citations: {enabled: true}` on
  document content blocks) over asking the model to write citations in prose. Native
  citations return verifiable character and page spans; prose citations can be fabricated.
- **Grounding:** every answer must be grounded in retrieved content. The system prompt must
  instruct the model to say "the documents do not contain this information" rather than
  answer from general knowledge. This is the single most important quality rule in the
  product — a confident wrong answer about a user's own contract or spreadsheet is worse
  than no answer at all.
- **Never truncate user content silently.** If content exceeds the context window, chunk it
  or summarize it deliberately, and surface that in the response metadata.
- **All prompts live in `ai/prompts/`** as named, versioned templates — never inline string
  literals scattered through service code. Prompts are behaviour; they deserve to be
  reviewable in a diff.

Before writing or changing any Claude API call, consult the `claude-api` skill for current
model IDs and parameter shapes. This API has changed materially; do not rely on recall.

---

## 6. Coding standards

### Python (backend)

- Python 3.11+ (3.13 is installed locally). Target 3.11 syntax for compatibility.
- **Type hints on every function signature.** `mypy` must pass.
- Format and lint with `ruff` (`ruff format`, `ruff check`).
- Pydantic v2 for all API request/response schemas and for settings.
- `async def` for I/O-bound route handlers and external calls; plain `def` for CPU-bound
  parsing work (FastAPI runs those in a threadpool automatically — do not block the loop).
- Raise typed domain exceptions from `core/exceptions.py`; translate them to HTTP status
  codes in one place (an exception handler), not at each call site.

### TypeScript (frontend)

- TypeScript strict mode. No `any` without a comment explaining why.
- Functional components with hooks. No class components.
- All API calls go through a single typed client module — components never call `fetch`
  directly.

### Naming

- `snake_case` in Python, `kebab-case` for frontend filenames, `PascalCase` for React
  components.
- Be explicit over clever. `extract_tables_from_page` beats `proc`.

---

## 7. Testing practices

Run tests after every meaningful change. A change is not complete until they pass.

| Layer | What to test | Tool |
|---|---|---|
| Parsers | Each format against small committed fixtures; assert block count, text, **and provenance correctness** | `pytest` |
| Services | Orchestration logic with parsers and LLM mocked | `pytest` plus fakes |
| Repositories | Real queries against a test database | `pytest` |
| API | Request/response contract, status codes, validation errors | `pytest` plus `httpx` |
| AI | Prompt construction and response parsing — **with the LLM mocked** | `pytest` |

Rules:

- **Never call the real Claude API in unit tests.** It is slow, costs money, and is
  non-deterministic. Use a fake `LLMProvider`. A separate, explicitly marked
  (`@pytest.mark.live`) suite may hit the real API — run manually, never in CI.
- **Fixture files must be tiny and synthetic.** Generate them; never commit real user
  documents or anything containing personal data.
- Test the failure paths: corrupt file, empty file, wrong extension, oversized file,
  password-protected PDF, malformed CSV. These are the cases that break in production.
- When you fix a bug, add the test that would have caught it. Confirm it fails before the
  fix and passes after.

**Standard verification sequence** (run before declaring any task complete):

```bash
# Backend — run from backend/ with the virtualenv active
ruff format --check .   # formatting
ruff check .            # linting
mypy app                # type checking
pytest                  # tests

# Frontend — run from frontend/
npm run typecheck       # tsc --noEmit
npm run lint            # eslint
npm run build           # production build
```

---

## 8. Security rules

- **Validate uploads by content, not by filename.** Check the magic bytes / sniffed type,
  not the extension. A file called `report.pdf` can be anything.
- **Enforce a maximum file size** at the web-server layer, before the file is buffered.
- **Never use the user-supplied filename as a path.** Store files under a generated UUID
  and keep the original name as display-only metadata. This blocks path traversal
  (`../../etc/passwd`) and filename collisions in one move.
- **XLSX and DOCX are ZIP archives** and can be zip bombs. Enforce decompressed-size limits.
  Disable external-entity and remote-template resolution in every XML parser.
- **CSV injection:** a cell beginning with `=`, `+`, `-`, or `@` is executed as a formula by
  spreadsheet applications. Escape these on any export path.
- **Treat all document content as untrusted input to the LLM** (prompt injection). A PDF can
  contain "ignore previous instructions". Keep document text clearly delimited from
  instructions in the prompt, and never let document content grant capability.
- **Scope every query by owner.** Every repository method that reads documents takes an
  owner identifier. Multi-tenant data leaks come from a single forgotten `WHERE` clause.
- **Never log document content, file contents, or API keys.** Log identifiers and metadata.

---

## 9. Git practices

- Do not commit or push unless the user asks.
- Never commit `.env`, uploaded files, virtual environments, or `node_modules`.
- Small, focused commits with messages that explain *why*, not just *what*.

---

## 10. Documentation upkeep

When the architecture changes, update in the same change:

- `README.md` — if the change affects the architecture, stack, or phases.
- This file — if the change affects rules, layering, or extension points.
- `.env.example` — if a new environment variable is introduced.

Documentation that describes a system which no longer exists is worse than no documentation.


---

## 11. Current state (Milestone 2)

### What exists

| Area | Status |
|---|---|
| Configuration from environment variables | Done — `app/core/config.py` |
| Structured logging | Done — `app/core/logging.py` |
| Typed domain exceptions + HTTP translation | Done — `app/core/exceptions.py`, `app/api/errors.py` |
| Async SQLAlchemy + Alembic migrations | Done — `app/core/database.py`, `alembic/` |
| `documents` table and domain model | Done — `app/models/` |
| Document repository (owner-scoped) | Done — `app/repositories/document_repo.py` |
| File storage interface + local implementation | Done — `app/storage/` |
| Health + readiness probes | Done — `app/api/routes/health.py` |
| Request correlation IDs | Done — `app/api/middleware.py`, `app/core/context.py` |
| OpenAPI documentation + examples | Done — `app/api/responses.py`, schema `Field` descriptions |
| Document read API (list / get / delete) | Done — `app/api/routes/documents.py` |
| React frontend talking to the backend | Done — `frontend/src/` |
| Upload + validation + detection | Done — `app/core/security.py`, `app/ingestion/detection.py` |
| Five parsers (PDF/DOCX/TXT/CSV/XLSX) | Done — `app/ingestion/parsers/` |
| Canonical `ContentBlock` + `Provenance` | Done — `app/models/domain/content.py` |
| Normalizer + structure-aware chunker | Done — `app/ingestion/` |
| Background processing + startup reaper | Done — `app/services/ingestion_service.py`, `job_service.py` |
| Frontend upload, live status, content viewer | Done — `frontend/src/components/` |
| Test suite | Done — 352 tests |

### What deliberately does not exist yet

* **`LLMProvider`, prompts, citations** — Milestone 3. No stub interface was created,
  because an empty interface written before its first use gets the abstraction wrong.
* **`Retriever`, embeddings, search** — Milestone 4. Chunks are stored and carry their
  `block_ids`, so the retrieval layer has everything it needs when it arrives.
* **OCR** — a scanned PDF fails with `failure_code="requires_ocr"` rather than
  silently returning nothing.
* **Authentication** — every request runs as `DEFAULT_OWNER_ID` in `app/api/deps.py`.
  The seam is real: routes, services, and every query already take and filter on an
  owner. Adding auth means changing `get_current_owner_id()` and nothing else.

### Database note

PostgreSQL is the production target, but the development default is SQLite so the project
runs on a machine with no database server. The custom `GUID` and `UTCDateTime` column
types in `app/models/db/base.py` absorb the dialect differences, and `render_as_batch` in
`alembic/env.py` makes migrations work on SQLite too. **Do not add PostgreSQL-only SQL
without either guarding it by dialect or moving the project fully onto PostgreSQL.**


---

## 12. Conventions established in Milestone 1

Follow these when adding code; they are already applied throughout.

### Error responses

* Raise a typed error from `core/exceptions.py`. Never raise `HTTPException` in a service.
* `app/api/errors.py` owns the status mapping. Add new mappings there, not at call sites.
* Every error response is `{"error": {code, message, details, request_id}}`. Keep the
  shape — the frontend has exactly one error path.
* `details` is for structured context a client can act on (which field failed). It must
  never carry secrets, file paths, stack traces, rejected input values, or document text.

### Request correlation IDs

Every request gets a server-generated ID (`app/api/middleware.py`). It appears in the
`X-Request-ID` response header, in the error body, and on every log line for that request.
**Never accept the ID from the client** — a forged or repeated ID corrupts the logs the
system relies on.

### Document status copy

The frontend's user-facing wording for `pending` / `processing` / `ready` / `failed` lives
in **one** place: `frontend/src/lib/documentStatus.ts`. Do not inline status strings in
components. When a document has failed, the backend's `error_message` is shown in
preference to the generic sentence — which is why a `FAILED` document must always carry
a reason.

### Documenting an endpoint

Every route needs `summary`, a docstring (becomes the description), `response_description`,
and its error cases via `error_responses(...)` from `app/api/responses.py`.
`tests/integration/test_openapi.py` fails the build if any of these are missing.


---

## 13. Ingestion rules (Milestone 2)

### The pipeline

```
upload -> validate -> store -> record (PENDING)
                                  |
                    background: PROCESSING -> parse -> normalize -> chunk
                                  -> persist -> READY / FAILED(+reason)
```

### Adding a file format

Write one parser in `app/ingestion/parsers/`, add one line to `_PARSERS` in
`registry.py`. Nothing else changes. If a format needs a change anywhere else, the
abstraction has leaked — fix the abstraction.

### Parser contract

* **Synchronous.** Parsing is CPU-bound; `async def` would be a lie. The caller runs it
  via `asyncio.to_thread`.
* **Pure.** Bytes in, `ParsedDocument` out. No database, storage, config, or logging
  setup. This is what makes parsers trivially testable.
* **Provenance on every block, never invented.** Use `Provenance.unknown(reason)` when a
  location genuinely cannot be determined, and `is_exact=False` when it is inferred
  (a PDF heading guessed from font size). A parser that drops provenance is broken.
* **Enforce `ParserLimits`.** Never return an unbounded block list.
* **Surface warnings.** Partial success must not look like clean success.

### Background processing

* The ingestion job **creates its own database session** from the factory. The
  request-scoped session is committed and closed by the time it runs — reusing it fails
  at runtime, and a mocked test would not notice.
* A document must **never** be left in `PROCESSING`. The one legitimate bare
  `except Exception` in the codebase guarantees this; it logs in full and records a
  user-facing reason.
* The startup reaper fails anything stranded by a crash.

### Failure handling

A parse failure is not an HTTP error — the response was sent long ago. It becomes
`status=FAILED` plus:

* `error_message` — plain language, written for the person who uploaded the file. No
  library names, paths, or tracebacks; those go to the log.
* `failure_code` — stable and machine-readable, so the frontend can branch. Current
  values: `requires_ocr`, `encrypted_file`, `corrupt_file`, `no_content`, `too_complex`,
  `interrupted`, `internal_error`.

### Tabular files are data, not prose

CSV and XLSX produce one `schema` block (columns, inferred types, profile) plus batched
`sheet_row` blocks. Sending 50,000 rows to a model is impossible and pointless; the
schema answers structural questions without reading any data.

### Chunking

Headings start new chunks; tables and schema blocks are never split. Every chunk records
`block_ids` — the middle link in the citation chain. Sizes come from settings
(`chunk_target_chars`, `chunk_overlap_chars`); the token count is a documented
character-based estimate, not a real tokenizer.
