"""FastAPI application factory and entry point.

Run locally with::

    uvicorn app.main:app --reload

The application is built by a factory (``create_app``) rather than at import time so
tests can construct isolated instances with overridden dependencies.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import documents, health
from app.core.config import Settings, get_settings
from app.core.database import dispose_engine, get_session_factory
from app.core.logging import configure_logging, get_logger
from app.services.ingestion_service import reap_interrupted_documents
from app.services.job_service import JobService

logger = get_logger(__name__)

API_VERSION = "0.1.0"

DESCRIPTION = """
**DocStruct AI** turns collections of documents into information that is structured,
searchable, and easy to extract.

### Current status — Milestone 2 (document ingestion)

Upload a PDF, DOCX, TXT, CSV or XLSX file and it is parsed into structured content, with
every piece traceable back to where it came from in the original file.

Uploads return **202 Accepted** immediately and are processed in the background: poll
`GET /documents/{document_id}` until `status` becomes `ready` or `failed`, then read the
extracted content from `GET /documents/{document_id}/content`.

Summaries, plain-language explanations, and question answering arrive in later milestones.

### Error format

Every failing request returns the same envelope:

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

Branch on `code`, which is stable, rather than on `message`, which may be reworded.
`request_id` also comes back in the `X-Request-ID` header on every response; quote it
when reporting a problem, because the server log holds the full technical detail under
that same ID.

Server-side failures (5xx) deliberately return a generic message — stack traces and
internal details are written to the log, never to the response.
"""

TAGS_METADATA = [
    {
        "name": "health",
        "description": (
            "Liveness and readiness probes. `/health` says the process is up; `/ready` "
            "says its dependencies are reachable. They are separate on purpose."
        ),
    },
    {
        "name": "documents",
        "description": (
            "Upload, metadata, processing status, and extracted content. A document "
            "moves through `pending` → `processing` → `ready` (or `failed`); poll "
            "`GET /documents/{document_id}` to follow it. Every extracted block records "
            "where in the source file it came from."
        ),
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start-up and shut-down work.

    On start-up the storage directory is created so a misconfigured path fails here,
    loudly, rather than on the first upload. On shut-down pooled database connections are
    closed so the process exits cleanly.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting %s env=%s db=%s storage=%s",
        settings.app_name,
        settings.app_env.value,
        settings.database_dialect,
        settings.storage_dir,
    )

    # One scheduler for the whole process, so its concurrency limit is global rather
    # than being reset per request.
    app.state.job_service = JobService(max_concurrent=settings.max_concurrent_processing)

    # Any document still marked PROCESSING belongs to a previous run that died mid-parse.
    # Nothing will ever finish it, so it is failed with an explanation instead of
    # spinning in the UI forever.
    try:
        await reap_interrupted_documents(get_session_factory())
    except Exception:
        # A missing or unreachable database must not stop the application from starting;
        # the readiness probe reports that separately. Logged in full, never swallowed.
        logger.exception("Could not reap interrupted documents at startup")

    yield

    # Give in-flight ingestion a chance to finish rather than abandoning it mid-write.
    await app.state.job_service.wait_for_all(timeout=10.0)
    await dispose_engine()
    logger.info("Shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        settings: Override settings, used by tests. Defaults to the environment.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=API_VERSION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        # Interactive API docs are useful in development but expose the full API surface,
        # so they are disabled in production.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    # Middleware runs in reverse registration order for responses, so registering the
    # request-context middleware last puts it outermost: every request gets a correlation
    # ID before anything else runs, including the CORS layer.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        # Browsers hide non-standard response headers from JavaScript unless they are
        # explicitly exposed; without this the frontend could not read the request ID.
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)

    return app


app = create_app()
