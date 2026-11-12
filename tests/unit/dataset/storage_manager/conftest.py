"""Common fixtures for the StorageManager test suite."""
import pytest

from hypex.dataset.storage_manager import StorageManager


@pytest.fixture(autouse=True)
def fresh_manager():
    """
    Each test starts with a clean registry. After the test, the registry is also cleaned:
    all references forgotten by the test are released (unpersist is called).
    """
    StorageManager.reset(release=True)
    yield StorageManager.instance()
    StorageManager.reset(release=True)
