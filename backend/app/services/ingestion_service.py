"""The ingestion pipeline.

Runs after the upload response has already been sent:

    PENDING -> PROCESSING -> parse -> normalize -> chunk -> persist -> READY
                                 └── any failure ──────────────────-> FAILED + reason
                                                              └── embed chunks (best-effort)

Three things here are easy to get wrong and are handled deliberately:

* **The background task creates its own database session.** The request-scoped session is
  already committed and closed by the time this runs, so reusing it would fail at
  runtime — and a mocked unit test would never notice.
* **Parsing runs in a worker thread.** It is CPU-bound; running it on the event loop
  would stall every other request for the duration.
* **Embedding generation is best-effort, after the document is already READY.** It powers
  `/ask` (retrieval) only, not the core ingestion guarantee — a document with no AI
  provider configured, or a transient embedding failure, must still ingest successfully.
  A failure here is logged and never changes the document's terminal state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.embeddings.base import Embedder
from app.core.config import Settings
from app.core.exceptions import (
    AIConfigurationError,
    AIUnavailableError,
    DocStructError,
    EmptyDocumentError,
    InvalidModelResponseError,
    NotFoundError,
    ProcessingError,
)
from app.core.logging import get_logger
from app.ingestion.chunker import chunk_blocks, estimate_tokens
from app.ingestion.normalizer import merge_short_blocks, normalize_document
from app.ingestion.parsers.base import ParserLimits
from app.ingestion.registry import get_parser
from app.models.domain.content import ParsedDocument
from app.models.domain.document import Document, ProcessingStatus
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_embedding_repo import ChunkEmbeddingRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.retrieval.hashing import compute_content_hash
from app.storage.base import FileStorage

logger = get_logger(__name__)

#: Failure code used when a document is found stranded in PROCESSING at startup.
INTERRUPTED_FAILURE_CODE = "interrupted"
INTERRUPTED_MESSAGE = (
    "Processing was interrupted when the server restarted. Please upload the file again."
)

#: Failure code for an unexpected internal error, as opposed to a problem with the file.
INTERNAL_FAILURE_CODE = "internal_error"
INTERNAL_MESSAGE = (
    "Something went wrong while processing this document. The problem has been logged."
)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """What a completed ingestion produced."""

    document_id: UUID
    block_count: int
    chunk_count: int
    page_count: int | None
    warnings: list[str]


class IngestionService:
    """Turns a stored file into content blocks and retrieval chunks.

    Args:
        session_factory: Used to open a fresh session per job. Deliberately a *factory*
            and not a session: this service runs outside any request, so there is no
            request-scoped session to borrow.
        storage: Where the uploaded bytes live.
        settings: Limits and chunking configuration.
        embedder: Used to compute chunk embeddings after a document is ready, for `/ask`.
            Optional — and checked for `is_configured` before use — so ingestion keeps
            working exactly as before on a deployment with no AI provider configured,
            and so existing tests that construct this service without one are unaffected.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        storage: FileStorage,
        settings: Settings,
        embedder: Embedder | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._settings = settings
        self._limits = ParserLimits.from_settings(settings)
        self._embedder = embedder

    async def process_document(self, document_id: UUID, *, owner_id: str) -> IngestionResult:
        """Run the full pipeline for one document.

        Every failure path ends with the document in a terminal state carrying a reason.
        A document left in ``PROCESSING`` would spin in the UI forever, so that outcome is
        not permitted regardless of what goes wrong.
        """
        logger.info("Ingestion started document_id=%s", document_id)

        try:
            await self._set_status(document_id, owner_id, ProcessingStatus.PROCESSING)
            content = await self._storage.load(document_id)
            document = await self._load_document(document_id, owner_id)

            # Parsing is CPU-bound. Running it inline would block the event loop and
            # stall every concurrent request until this document finished.
            parsed = await asyncio.to_thread(
                get_parser(document.file_type).parse, content, self._limits
            )

            result = await self._persist(document_id, owner_id, parsed)

            logger.info(
                "Ingestion finished document_id=%s blocks=%d chunks=%d",
                document_id,
                result.block_count,
                result.chunk_count,
            )

            # Best-effort, after the document is already READY: embeddings power `/ask`
            # only, never the core ingestion guarantee. Anything that goes wrong here —
            # no provider configured, a transient API failure, a database hiccup — must
            # be logged, never allowed to undo a successful ingestion or leave the
            # document in a worse state than "ready, but not yet searchable."
            try:
                await self._generate_embeddings(document_id)
            except Exception:
                logger.exception(
                    "Embedding generation failed after successful ingestion document_id=%s",
                    document_id,
                )

            return result

        except DocStructError as exc:
            # A typed error means we know what went wrong and can explain it. Even for
            # server-side ones the user gets a sentence; the detail goes to the log.
            is_user_fixable = not isinstance(exc, ProcessingError)
            await self._fail(
                document_id,
                owner_id,
                message=exc.message if is_user_fixable else INTERNAL_MESSAGE,
                failure_code=exc.code if is_user_fixable else INTERNAL_FAILURE_CODE,
                log_detail=f"{type(exc).__name__}: {exc.message}",
                include_traceback=not is_user_fixable,
            )
            raise

        except Exception as exc:
            # A bare `except Exception` is correct here: the alternative is a document
            # stuck in PROCESSING forever. Logged in full, never swallowed. (The other
            # broad catch in this file, around `_generate_embeddings` above, guards a
            # different and unrelated risk — an optional, best-effort step corrupting an
            # already-successful ingestion's terminal state.)
            await self._fail(
                document_id,
                owner_id,
                message=INTERNAL_MESSAGE,
                failure_code=INTERNAL_FAILURE_CODE,
                log_detail=f"{type(exc).__name__}: {exc}",
                include_traceback=True,
            )
            raise

    async def _persist(
        self, document_id: UUID, owner_id: str, parsed: ParsedDocument
    ) -> IngestionResult:
        """Normalize, chunk, and write everything in one transaction."""
        normalized = normalize_document(parsed)
        blocks = merge_short_blocks(normalized.blocks)
        warnings = [*parsed.warnings, *normalized.warnings]

        if not blocks:
            raise EmptyDocumentError("No readable content could be extracted from this file")

        chunks = chunk_blocks(
            blocks,
            target_chars=self._settings.chunk_target_chars,
            overlap_chars=self._settings.chunk_overlap_chars,
        )

        async with self._session_factory() as session:
            block_repository = ContentBlockRepository(session)
            chunk_repository = ChunkRepository(session)
            document_repository = DocumentRepository(session)

            block_ids = await block_repository.replace_for_document(document_id, blocks)

            # Map each chunk's block positions to the IDs just assigned. This is the
            # link that makes a retrieved chunk citable.
            index_to_id = {
                block.order: block_ids[position] for position, block in enumerate(blocks)
            }
            chunk_rows = [
                (
                    chunk.text,
                    [index_to_id[index] for index in chunk.block_indices if index in index_to_id],
                    estimate_tokens(chunk.text),
                    chunk.location_label,
                )
                for chunk in chunks
            ]
            await chunk_repository.replace_for_document(document_id, chunk_rows)

            await document_repository.update_status(
                document_id,
                owner_id=owner_id,
                status=ProcessingStatus.READY,
                page_count=parsed.stats.page_count,
            )
            await session.commit()

        if warnings:
            logger.info("Ingestion warnings document_id=%s count=%d", document_id, len(warnings))

        return IngestionResult(
            document_id=document_id,
            block_count=len(blocks),
            chunk_count=len(chunks),
            page_count=parsed.stats.page_count,
            warnings=warnings,
        )

    async def _generate_embeddings(self, document_id: UUID) -> None:
        """Compute and persist an embedding for each of a document's chunks.

        No ``owner_id``: chunks and embeddings are addressed by ``document_id`` alone at
        the repository level (only ``DocumentRepository`` is owner-scoped) — by the time
        this runs, the document has already been loaded once under the correct owner.

        Reuses exactly the mechanism M4.3 built for this: ``Embedder.embed()`` for the
        vector, ``compute_content_hash()`` for the staleness hash
        ``ChunkEmbeddingRepository`` compares against later, and
        ``ChunkEmbeddingRepository.replace_for_chunk()`` to store it. Nothing about how a
        chunk becomes an embedding is invented here — this is the missing call site, not
        new mechanism.

        One chunk's embedding failing does not stop the rest: a partially-embedded
        document can still be searched on whatever succeeded, which is strictly better
        than none. The caller wraps this whole method in its own broad catch, so a
        deeper failure (the session itself, a database error) is contained too.
        """
        if self._embedder is None or not self._embedder.is_configured:
            logger.info(
                "Skipping embedding generation (no embedder configured) document_id=%s",
                document_id,
            )
            return

        async with self._session_factory() as session:
            chunk_repository = ChunkRepository(session)
            embedding_repository = ChunkEmbeddingRepository(session)

            chunks = await chunk_repository.list_for_document(document_id)
            embedded = 0

            for chunk in chunks:
                try:
                    result = await self._embedder.embed(chunk.text)
                except (
                    AIConfigurationError,
                    AIUnavailableError,
                    InvalidModelResponseError,
                ) as exc:
                    logger.warning(
                        "Could not embed chunk chunk_id=%s document_id=%s reason=%s",
                        chunk.id,
                        document_id,
                        type(exc).__name__,
                    )
                    continue

                await embedding_repository.replace_for_chunk(
                    chunk_id=chunk.id,
                    document_id=document_id,
                    model=self._embedder.model_name,
                    values=result.values,
                    content_hash=compute_content_hash(chunk.text),
                    token_count=result.token_count,
                )
                embedded += 1

            await session.commit()

        log = logger.info if embedded == len(chunks) else logger.warning
        log(
            "Embedding generation finished document_id=%s embedded=%d total=%d",
            document_id,
            embedded,
            len(chunks),
        )

    async def _load_document(self, document_id: UUID, owner_id: str) -> Document:
        """Fetch the document record, in its own session."""
        async with self._session_factory() as session:
            document = await DocumentRepository(session).get(document_id, owner_id=owner_id)
        if document is None:
            raise NotFoundError("Document not found", details={"document_id": str(document_id)})
        return document

    async def _set_status(self, document_id: UUID, owner_id: str, status: ProcessingStatus) -> None:
        async with self._session_factory() as session:
            await DocumentRepository(session).update_status(
                document_id, owner_id=owner_id, status=status
            )
            await session.commit()

    async def _fail(
        self,
        document_id: UUID,
        owner_id: str,
        *,
        message: str,
        failure_code: str,
        log_detail: str,
        include_traceback: bool,
    ) -> None:
        """Record a failure on the document and log the technical detail.

        The message written to the document is what the user reads, so it never contains
        a library name, a path, or a traceback. Those go to the log.
        """
        logger.error(
            "Ingestion failed document_id=%s code=%s detail=%s",
            document_id,
            failure_code,
            log_detail,
            exc_info=include_traceback,
        )

        try:
            async with self._session_factory() as session:
                await DocumentRepository(session).update_status(
                    document_id,
                    owner_id=owner_id,
                    status=ProcessingStatus.FAILED,
                    error_message=message,
                    failure_code=failure_code,
                )
                await session.commit()
        except Exception:
            # If even recording the failure fails, say so loudly rather than losing it.
            logger.exception("Could not record ingestion failure document_id=%s", document_id)


async def reap_interrupted_documents(
    session_factory: async_sessionmaker[AsyncSession],
) -> int:
    """Fail any document left mid-processing by a previous run. Returns how many.

    A document is only ever ``PROCESSING`` while a background task is actively working on
    it. If one is in that state at startup, the previous process died mid-parse and
    nothing will ever finish it — so without this it would spin in the UI indefinitely.
    """
    async with session_factory() as session:
        count = await DocumentRepository(session).fail_stranded_processing(
            reason=INTERRUPTED_MESSAGE,
            failure_code=INTERRUPTED_FAILURE_CODE,
        )
        await session.commit()

    if count:
        logger.warning("Reaped %d document(s) interrupted by a restart", count)
    return count
