"""File storage abstraction."""

from app.storage.base import FileStorage
from app.storage.local import LocalFileStorage

__all__ = ["FileStorage", "LocalFileStorage"]
