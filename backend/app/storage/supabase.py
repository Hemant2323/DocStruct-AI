"""Supabase Storage implementation of :class:`FileStorage`.

The second implementation of the storage seam, and the reason that seam exists: a host
with no persistent disk (Render's free tier, for one) cannot keep uploaded documents on
the filesystem, and everything above this file is written against the interface rather
than against a directory.

**No SDK.** The Supabase Python client would pull in gotrue, postgrest, realtime and
storage3 to make four HTTP calls. ``httpx`` is already a production dependency for
exactly this reason — the Gemini and NVIDIA providers call their REST APIs the same way —
so this follows that pattern rather than adding a package.

The routes used are the Storage API's object routes, under ``/storage/v1/object``:

===========  ==========================================  ==============================
Operation    Request                                     Notes
===========  ==========================================  ==============================
save         ``POST   /object/{bucket}/{path}``          ``x-upsert: true`` to overwrite
load         ``GET    /object/{bucket}/{path}``
delete       ``DELETE /object/{bucket}/{path}``
exists       ``HEAD   /object/{bucket}/{path}``          returns no body
===========  ==========================================  ==============================

``x-upsert`` is not optional here. Without it a repeat upload to the same path is
rejected as "Asset Already Exists", and :class:`~app.storage.base.FileStorage` requires
``save`` to overwrite — a document re-uploaded under the same id must replace its bytes,
which is behaviour the local backend gets free from ``write_bytes`` and this one has to
ask for.

``HEAD`` rather than a metadata ``GET`` for ``exists``: it answers the only question
being asked, and returns no body at all, so there is nothing in the response that could
reach a log.

**The key never leaves this object.** It is read once in ``__init__`` to build the
client's headers and is not kept on an attribute of its own, matching ``GeminiProvider``.
No method logs a header, a URL query, or a response body — an error body from Storage can
quote the request, and the object path contains a document id.
"""

from __future__ import annotations

from uuid import UUID

import httpx

from app.core.exceptions import NotFoundError, StorageError
from app.core.logging import get_logger
from app.storage.base import FileStorage

logger = get_logger(__name__)

#: Sent on every upload so a repeat save replaces the object instead of being refused.
_UPSERT_HEADER = {"x-upsert": "true"}

#: Uploads are raw bytes of an already-validated document. The real type is recorded on
#: the database row; Storage only has to hand back the same bytes.
_CONTENT_TYPE = "application/octet-stream"


class SupabaseFileStorage(FileStorage):
    """Stores document bytes in a Supabase Storage bucket.

    Args:
        base_url: The project URL, e.g. ``https://abcdefgh.supabase.co``. Not the storage
            endpoint — the ``/storage/v1`` prefix is added here.
        service_key: The service-role key. Used for both the ``apikey`` and
            ``Authorization`` headers, which is how the Storage API authenticates a
            trusted server-side caller.
        bucket: Bucket name. Should be private; objects are named by document id, which
            is unguessable but not a secret.
        timeout_seconds: Per-request timeout.

    Raises:
        StorageError: If the URL or key is missing. Constructing a client that cannot
            authenticate would turn a configuration mistake into a failed upload much
            later, with a much worse error.
    """

    def __init__(
        self,
        *,
        base_url: str,
        service_key: str,
        bucket: str = "documents",
        timeout_seconds: float = 30.0,
    ) -> None:
        url = (base_url or "").strip().rstrip("/")
        key = (service_key or "").strip()
        if not url or not key:
            raise StorageError("Supabase storage is not configured.")

        self._bucket = bucket
        self._object_root = f"{url}/storage/v1/object/{bucket}"
        self._client = httpx.AsyncClient(
            headers={
                # Both, deliberately. The Storage API expects `apikey` for the project
                # and `Authorization` for the caller's identity; with a service-role key
                # they are the same value.
                "apikey": key,
                "Authorization": f"Bearer {key}",
            },
            timeout=timeout_seconds,
        )
        logger.info("Supabase storage configured bucket=%s", bucket)

    def _url_for(self, document_id: UUID) -> str:
        """The object URL for a document.

        The path is built from the UUID alone, exactly as the local backend builds its
        filename: no user-supplied string reaches it, so a name cannot traverse out of
        the bucket or collide with another document.
        """
        return f"{self._object_root}/{document_id}.bin"

    async def save(self, document_id: UUID, content: bytes) -> int:
        try:
            response = await self._client.post(
                self._url_for(document_id),
                content=content,
                headers={**_UPSERT_HEADER, "Content-Type": _CONTENT_TYPE},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._failure("write document to", document_id, exc) from exc

        logger.info("Stored document bytes=%d document_id=%s", len(content), document_id)
        return len(content)

    async def load(self, document_id: UUID) -> bytes:
        try:
            response = await self._client.get(self._url_for(document_id))
            if response.status_code == httpx.codes.NOT_FOUND:
                raise NotFoundError(
                    "Document content not found in storage",
                    details={"document_id": str(document_id)},
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._failure("read document from", document_id, exc) from exc

        return response.content

    async def delete(self, document_id: UUID) -> None:
        try:
            response = await self._client.delete(self._url_for(document_id))
            # Deleting something already absent succeeds: the desired end state is "not
            # present", and it has been reached. The local backend expresses this as
            # `unlink(missing_ok=True)`; over HTTP it is a 404 that is not an error.
            if response.status_code != httpx.codes.NOT_FOUND:
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._failure("delete document from", document_id, exc) from exc

        logger.info("Deleted document from storage document_id=%s", document_id)

    async def exists(self, document_id: UUID) -> bool:
        try:
            response = await self._client.head(self._url_for(document_id))
        except httpx.HTTPError as exc:
            raise self._failure("check document in", document_id, exc) from exc

        if response.status_code == httpx.codes.NOT_FOUND:
            return False
        if response.is_success:
            return True

        # Anything else — 401, 403, 500 — is not an answer to the question. Reporting it
        # as "no" would let a broken bucket look like an empty one.
        raise StorageError(
            "Failed to check document in storage",
            details={"document_id": str(document_id)},
        )

    @staticmethod
    def _failure(action: str, document_id: UUID, exc: httpx.HTTPError) -> StorageError:
        """Turn any httpx failure into the error the rest of the app expects.

        The upstream status is logged; the response body never is. A Storage error body
        can echo the request that produced it, and this project's rule is that upstream
        text does not reach a log line or an HTTP response — only our own wording, plus
        identifiers the caller already has.
        """
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        logger.error(
            "Supabase storage failed to %s storage document_id=%s status=%s error=%s",
            action,
            document_id,
            status,
            type(exc).__name__,
        )
        return StorageError(
            f"Failed to {action} storage",
            details={"document_id": str(document_id)},
        )
