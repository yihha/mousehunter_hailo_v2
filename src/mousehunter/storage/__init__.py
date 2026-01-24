"""Storage module for local and cloud storage management."""

from mousehunter.storage.cloud_storage import (
    CloudStorage,
    get_cloud_storage,
    test_cloud_storage,
)

__all__ = [
    "CloudStorage",
    "get_cloud_storage",
    "test_cloud_storage",
]
