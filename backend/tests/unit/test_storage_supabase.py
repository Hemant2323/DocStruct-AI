"""Tests for Supabase file storage, with the HTTP client mocked.

Never touches the network — every test patches the client's request method. What is under
test is *our* code: the URL and headers a request carries, how each status code is
interpreted, and that every failure becomes one of the project's typed errors with no
credential in it.

The assertions deliberately mirror `test_storage.py`, because the two classes implement
the same ABC and a caller must not be able to tell them apart. Where the local backend
gets a behaviour free from the filesystem — overwrite from `write_bytes`, idempotent
delete from `unlink(missing_ok=True)` — this backend has to ask for it explicitly, so
those are the cases most worth pinning here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import httpx
import pytest

from app.core.exceptions import NotFoundError, StorageError
from app.storage.supabase import SupabaseFileStorage

PROJECT_URL = "https://abcdefgh.supabase.co"
SERVICE_KEY = "test-service-key-not-real"
BUCKET = "documents"


def _storage(**overrides: Any) -> SupabaseFileStorage:
    settings: dict[str, Any] = {
        "base_url": PROJECT_URL,
        "service_key": SERVICE_KEY,
        "bucket": BUCKET,
    }
    settings.update(overrides)
    return SupabaseFileStorage(**settings)


def _response(
    status_code: int = 200,
    *,
    content: bytes = b"",
    method: str = "GET",
    url: str = f"{PROJECT_URL}/storage/v1/object/{BUCKET}/x.bin",
) -> httpx.Response:
    """A real `httpx.Response`, so `raise_for_status()` and `.content` behave exactly as
    they do against a live server — only the transport is faked."""
    return httpx.Response(
        status_code=status_code,
        content=content,
        request=httpx.Request(method, url),
    )


def _object_url(document_id: UUID) -> str:
    return f"{PROJECT_URL}/storage/v1/object/{BUCKET}/{document_id}.bin"


class TestConfiguration:
    def test_a_missing_url_is_refused_at_construction(self) -> None:
        """Rather than building a client that cannot authenticate and failing later, at
        the first upload, with a much worse error."""
        with pytest.raises(StorageError):
            _storage(base_url="")

    def test_a_missing_key_is_refused_at_construction(self) -> None:
        with pytest.raises(StorageError):
            _storage(service_key="   ")

    def test_the_key_is_sent_in_both_authentication_headers(self) -> None:
        """The Storage API expects `apikey` for the project and `Authorization` for the
        caller; a service-role key is the same value in both."""
        storage = _storage()

        assert storage._client.headers["apikey"] == SERVICE_KEY
        assert storage._client.headers["authorization"] == f"Bearer {SERVICE_KEY}"

    def test_a_trailing_slash_on_the_project_url_does_not_double_up(self) -> None:
        storage = _storage(base_url=f"{PROJECT_URL}/")
        document_id = uuid4()

        assert storage._url_for(document_id) == _object_url(document_id)

    def test_the_object_path_is_the_document_uuid(self) -> None:
        """The same rule the local backend follows: no user-supplied string reaches the
        path, so a filename cannot traverse out of the bucket or collide."""
        document_id = uuid4()

        object_url = _storage()._url_for(document_id)

        assert object_url == _object_url(document_id)
        assert object_url.endswith(f"/{document_id}.bin")


class TestSave:
    async def test_save_posts_the_content_and_returns_the_byte_count(self) -> None:
        storage = _storage()
        document_id = uuid4()
        content = b"DocStruct AI test content"
        captured: dict[str, Any] = {}

        async def capture(url: str, **kwargs: Any) -> httpx.Response:
            captured["url"] = url
            captured.update(kwargs)
            return _response(200, method="POST")

        with patch.object(storage._client, "post", side_effect=capture):
            written = await storage.save(document_id, content)

        assert written == len(content)
        assert captured["url"] == _object_url(document_id)
        assert captured["content"] == content

    async def test_save_requests_an_overwrite(self) -> None:
        """The contract requires `save` to replace existing bytes. Without `x-upsert` the
        Storage API refuses a repeat upload as "Asset Already Exists", so this header is
        the whole of that guarantee."""
        storage = _storage()
        captured: dict[str, Any] = {}

        async def capture(url: str, **kwargs: Any) -> httpx.Response:
            captured.update(kwargs)
            return _response(200, method="POST")

        with patch.object(storage._client, "post", side_effect=capture):
            await storage.save(uuid4(), b"content")

        assert captured["headers"]["x-upsert"] == "true"

    async def test_saving_twice_overwrites_rather_than_failing(self) -> None:
        """The behaviour `test_overwrite_replaces_content` pins for the local backend."""
        storage = _storage()
        document_id = uuid4()

        with patch.object(
            storage._client, "post", AsyncMock(return_value=_response(200, method="POST"))
        ):
            await storage.save(document_id, b"first")
            second = await storage.save(document_id, b"second longer")

        assert second == len(b"second longer")

    async def test_empty_content_is_a_real_save(self) -> None:
        """An empty file is a genuine case and must not be confused with a missing one."""
        storage = _storage()

        with patch.object(
            storage._client, "post", AsyncMock(return_value=_response(200, method="POST"))
        ):
            assert await storage.save(uuid4(), b"") == 0

    async def test_an_upload_failure_becomes_a_storage_error(self) -> None:
        storage = _storage()

        with (
            patch.object(
                storage._client, "post", AsyncMock(return_value=_response(500, method="POST"))
            ),
            pytest.raises(StorageError) as exc_info,
        ):
            await storage.save(uuid4(), b"content")

        assert exc_info.value.code == "storage_error"


class TestLoad:
    async def test_load_returns_the_exact_bytes(self) -> None:
        """Bytes go to the parsers untouched, so no encoding or newline translation may
        happen anywhere on this path."""
        storage = _storage()
        content = bytes(range(256)) + b"\r\n\x00binary\xff"

        with patch.object(
            storage._client, "get", AsyncMock(return_value=_response(200, content=content))
        ):
            assert await storage.load(uuid4()) == content

    async def test_load_requests_the_document_s_own_object(self) -> None:
        storage = _storage()
        document_id = uuid4()
        captured: dict[str, Any] = {}

        async def capture(url: str, **_: Any) -> httpx.Response:
            captured["url"] = url
            return _response(200, content=b"x")

        with patch.object(storage._client, "get", side_effect=capture):
            await storage.load(document_id)

        assert captured["url"] == _object_url(document_id)

    async def test_a_missing_object_raises_not_found(self) -> None:
        """Matching the local backend, where a missing file is `NotFoundError` and not a
        storage failure — the difference decides whether a caller retries."""
        storage = _storage()

        with (
            patch.object(storage._client, "get", AsyncMock(return_value=_response(404))),
            pytest.raises(NotFoundError) as exc_info,
        ):
            await storage.load(uuid4())

        assert exc_info.value.code == "not_found"

    async def test_a_server_failure_becomes_a_storage_error_not_not_found(self) -> None:
        """A 500 means we do not know whether the object exists. Reporting it as missing
        would invite a caller to delete a row whose bytes are still there."""
        storage = _storage()

        with (
            patch.object(storage._client, "get", AsyncMock(return_value=_response(503))),
            pytest.raises(StorageError),
        ):
            await storage.load(uuid4())


class TestDelete:
    async def test_delete_sends_a_delete_for_the_object(self) -> None:
        storage = _storage()
        document_id = uuid4()
        captured: dict[str, Any] = {}

        async def capture(url: str, **_: Any) -> httpx.Response:
            captured["url"] = url
            return _response(200, method="DELETE")

        with patch.object(storage._client, "delete", side_effect=capture):
            await storage.delete(document_id)

        assert captured["url"] == _object_url(document_id)

    async def test_delete_is_idempotent_for_a_missing_object(self) -> None:
        """The desired end state is "not present", and it has already been reached."""
        storage = _storage()

        with patch.object(
            storage._client, "delete", AsyncMock(return_value=_response(404, method="DELETE"))
        ):
            await storage.delete(uuid4())  # must not raise

    async def test_a_delete_failure_becomes_a_storage_error(self) -> None:
        storage = _storage()

        with (
            patch.object(
                storage._client, "delete", AsyncMock(return_value=_response(500, method="DELETE"))
            ),
            pytest.raises(StorageError),
        ):
            await storage.delete(uuid4())


class TestExists:
    async def test_exists_is_true_for_a_stored_object(self) -> None:
        storage = _storage()

        with patch.object(
            storage._client, "head", AsyncMock(return_value=_response(200, method="HEAD"))
        ):
            assert await storage.exists(uuid4()) is True

    async def test_exists_is_false_for_a_missing_object(self) -> None:
        storage = _storage()

        with patch.object(
            storage._client, "head", AsyncMock(return_value=_response(404, method="HEAD"))
        ):
            assert await storage.exists(uuid4()) is False

    async def test_an_unexpected_status_is_raised_rather_than_answered_false(self) -> None:
        """A broken bucket must not look like an empty one: 403 is not "no"."""
        storage = _storage()

        with (
            patch.object(
                storage._client, "head", AsyncMock(return_value=_response(403, method="HEAD"))
            ),
            pytest.raises(StorageError),
        ):
            await storage.exists(uuid4())


class TestNetworkFailures:
    """Transport failures, which carry no HTTP status at all."""

    @pytest.mark.parametrize(
        "failure",
        [
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("timed out"),
            httpx.ConnectError("refused"),
        ],
    )
    async def test_a_transport_failure_on_save_becomes_a_storage_error(
        self, failure: httpx.HTTPError
    ) -> None:
        storage = _storage()

        with (
            patch.object(storage._client, "post", AsyncMock(side_effect=failure)),
            pytest.raises(StorageError),
        ):
            await storage.save(uuid4(), b"content")

    async def test_a_transport_failure_on_load_becomes_a_storage_error(self) -> None:
        storage = _storage()

        with (
            patch.object(
                storage._client, "get", AsyncMock(side_effect=httpx.ConnectError("refused"))
            ),
            pytest.raises(StorageError),
        ):
            await storage.load(uuid4())

    async def test_no_bare_exception_type_escapes(self) -> None:
        """Every failure path must produce one of our own typed errors, never let an
        httpx exception reach a caller that only knows about domain errors."""
        storage = _storage()

        for method in ("post", "get", "delete", "head"):
            with (
                patch.object(
                    storage._client, method, AsyncMock(side_effect=httpx.ConnectError("refused"))
                ),
                pytest.raises((StorageError, NotFoundError)),
            ):
                if method == "post":
                    await storage.save(uuid4(), b"c")
                elif method == "get":
                    await storage.load(uuid4())
                elif method == "delete":
                    await storage.delete(uuid4())
                else:
                    await storage.exists(uuid4())


class TestCredentialsNeverLeak:
    """The service-role key bypasses row-level security, so it is the most dangerous
    secret in the project. Nothing it touches may carry it outward."""

    @pytest.mark.parametrize("status", [400, 401, 403, 500])
    async def test_the_key_is_absent_from_a_storage_error(self, status: int) -> None:
        storage = _storage()

        with (
            patch.object(
                storage._client, "post", AsyncMock(return_value=_response(status, method="POST"))
            ),
            pytest.raises(StorageError) as exc_info,
        ):
            await storage.save(uuid4(), b"content")

        rendered = f"{exc_info.value.message} {exc_info.value.details}"
        assert SERVICE_KEY not in rendered
        assert "Bearer" not in rendered
        assert "apikey" not in rendered

    async def test_an_upstream_error_body_is_not_repeated_to_the_caller(self) -> None:
        """A Storage error body can quote the request that produced it, including the
        object path. Our own wording goes out instead."""
        storage = _storage()
        body = b'{"error":"InvalidJWT","message":"token test-service-key-not-real is bad"}'

        with (
            patch.object(
                storage._client,
                "post",
                AsyncMock(return_value=_response(401, content=body, method="POST")),
            ),
            pytest.raises(StorageError) as exc_info,
        ):
            await storage.save(uuid4(), b"content")

        assert SERVICE_KEY not in exc_info.value.message
        assert "InvalidJWT" not in exc_info.value.message

    async def test_the_key_is_not_written_to_the_log_on_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        storage = _storage()

        with (
            caplog.at_level("DEBUG"),
            patch.object(
                storage._client, "post", AsyncMock(return_value=_response(401, method="POST"))
            ),
            pytest.raises(StorageError),
        ):
            await storage.save(uuid4(), b"content")

        assert SERVICE_KEY not in caplog.text
        assert "Bearer" not in caplog.text

    async def test_the_key_is_not_written_to_the_log_on_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("DEBUG"):
            storage = _storage()

            with patch.object(
                storage._client, "post", AsyncMock(return_value=_response(200, method="POST"))
            ):
                await storage.save(uuid4(), b"content")

        assert SERVICE_KEY not in caplog.text
