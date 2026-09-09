"""Tests for storage selection: which ``FileStorage`` ``STORAGE_BACKEND`` builds.

Calls ``app.api.deps._build_file_storage`` directly rather than going through FastAPI's
dependency injection or a live app — the branching logic is the only thing being tested,
and it needs no request, database or network. Sits beside ``test_provider_selection.py``
and ``test_embedder_selection.py``, which test the other two seams the same way.

No test here makes a network call: constructing either backend never does. The Supabase
credentials below are synthetic and never leave the process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.deps import (
    _build_file_storage,
    get_file_storage,
    reset_file_storage,
)
from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.storage.local import LocalFileStorage
from app.storage.supabase import SupabaseFileStorage

SUPABASE_URL = "https://abcdefgh.supabase.co"
SUPABASE_KEY = "test-service-key-not-real"


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "supabase_url": SUPABASE_URL,
        "supabase_service_key": SUPABASE_KEY,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _reset_storage() -> None:
    """Drop the cached backend around every test in this module.

    The Supabase backend is a process-wide singleton, so without this a test would
    inherit one built from a previous test's settings — the same reason `_reset_provider`
    exists in conftest.
    """
    reset_file_storage()


class TestStorageSelection:
    def test_defaults_to_local(self) -> None:
        """STORAGE_BACKEND defaults to "local", so existing behaviour is unchanged for
        anyone who has not set it."""
        storage = _build_file_storage(_settings())

        assert isinstance(storage, LocalFileStorage)

    def test_explicit_local_builds_local_storage(self) -> None:
        storage = _build_file_storage(_settings(storage_backend="local"))

        assert isinstance(storage, LocalFileStorage)

    def test_supabase_builds_supabase_storage(self) -> None:
        """The wiring this step exists for: selecting supabase must actually reach it."""
        storage = _build_file_storage(_settings(storage_backend="supabase"))

        assert isinstance(storage, SupabaseFileStorage)

    def test_switching_backend_changes_the_returned_type(self) -> None:
        """Confirms the branch is genuinely live, not returning one hard-coded type."""
        local = _build_file_storage(_settings(storage_backend="local"))
        supabase = _build_file_storage(_settings(storage_backend="supabase"))

        assert type(local) is not type(supabase)

    def test_local_storage_receives_its_own_setting(self, tmp_path: Path) -> None:
        """The two backends must not share configuration by accident."""
        storage = _build_file_storage(_settings(storage_backend="local", storage_dir=tmp_path))

        assert isinstance(storage, LocalFileStorage)
        assert storage.root == tmp_path.resolve()

    def test_supabase_storage_receives_its_own_settings(self) -> None:
        storage = _build_file_storage(
            _settings(storage_backend="supabase", supabase_storage_bucket="uploads")
        )

        assert isinstance(storage, SupabaseFileStorage)
        assert storage._bucket == "uploads"
        assert storage._object_root == f"{SUPABASE_URL}/storage/v1/object/uploads"


class TestMissingSupabaseConfiguration:
    """Selecting supabase without credentials must fail, and say so.

    Falling back to local storage would be the dangerous outcome: uploads would be
    written to a disk the operator believes is unused, and a free host discards that disk
    on the next deploy — the documents would be gone with nothing having said so.
    """

    def test_a_missing_url_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            _build_file_storage(_settings(storage_backend="supabase", supabase_url=None))

        assert exc_info.value.code == "configuration_error"

    def test_a_missing_key_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError):
            _build_file_storage(_settings(storage_backend="supabase", supabase_service_key=None))

    def test_a_blank_key_is_a_configuration_error(self) -> None:
        """Blanking a value in the environment is how people turn one off; it must not
        read as "configured with an empty key"."""
        with pytest.raises(ConfigurationError):
            _build_file_storage(_settings(storage_backend="supabase", supabase_service_key=""))

    def test_it_does_not_silently_fall_back_to_local(self) -> None:
        """Stated as its own test because a fallback would look like success."""
        with pytest.raises(ConfigurationError):
            _build_file_storage(_settings(storage_backend="supabase", supabase_url=None))

    def test_the_message_names_the_settings_that_are_missing(self) -> None:
        """An operator reading this in a startup log should not have to guess."""
        with pytest.raises(ConfigurationError) as exc_info:
            _build_file_storage(_settings(storage_backend="supabase", supabase_url=None))

        assert "STORAGE_BACKEND" in exc_info.value.message
        assert "SUPABASE_URL" in exc_info.value.message
        assert "SUPABASE_SERVICE_KEY" in exc_info.value.message

    def test_the_message_does_not_contain_the_key(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            _build_file_storage(_settings(storage_backend="supabase", supabase_url=None))

        assert SUPABASE_KEY not in exc_info.value.message


class TestSingletonBehaviour:
    """Only the Supabase backend is cached, and deliberately so."""

    def test_supabase_storage_is_reused_across_calls(self) -> None:
        """It owns an httpx connection pool; rebuilding it per request would discard
        those connections, exactly as the provider and embedder comments describe."""
        settings = _settings(storage_backend="supabase")

        first = get_file_storage(settings)
        second = get_file_storage(settings)

        assert first is second

    def test_local_storage_is_not_cached(self) -> None:
        """It holds a resolved path and no pool, so caching it would buy nothing — and
        would pin the first caller's `storage_dir` for the life of the process, which is
        the value the test suite varies per test."""
        settings = _settings(storage_backend="local")

        first = get_file_storage(settings)
        second = get_file_storage(settings)

        assert first is not second

    def test_local_storage_follows_a_changed_storage_dir(self, tmp_path: Path) -> None:
        """The consequence that matters: a second caller with a different directory gets
        that directory, not the first one's."""
        first = get_file_storage(_settings(storage_backend="local", storage_dir=tmp_path))
        other = tmp_path / "elsewhere"
        second = get_file_storage(_settings(storage_backend="local", storage_dir=other))

        assert isinstance(first, LocalFileStorage)
        assert isinstance(second, LocalFileStorage)
        assert first.root != second.root

    def test_reset_drops_the_cached_backend(self) -> None:
        settings = _settings(storage_backend="supabase")
        first = get_file_storage(settings)

        reset_file_storage()
        second = get_file_storage(settings)

        assert first is not second

    def test_reset_lets_a_later_call_pick_up_new_settings(self) -> None:
        """What the reset is for: a test that switches backends must not inherit the
        previous one."""
        supabase = get_file_storage(_settings(storage_backend="supabase"))
        reset_file_storage()
        local = get_file_storage(_settings(storage_backend="local"))

        assert isinstance(supabase, SupabaseFileStorage)
        assert isinstance(local, LocalFileStorage)

    def test_missing_configuration_is_not_cached_as_a_failure(self) -> None:
        """A failed build must leave the singleton empty, so fixing the environment and
        retrying works rather than repeating the original error forever."""
        with pytest.raises(ConfigurationError):
            get_file_storage(_settings(storage_backend="supabase", supabase_url=None))

        storage = get_file_storage(_settings(storage_backend="supabase"))

        assert isinstance(storage, SupabaseFileStorage)
