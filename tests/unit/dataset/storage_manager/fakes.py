"""Emulators and mini-harness for StorageManager tests (without Spark).

SparkNavigation replicates your integration scheme (see docstring in
storage_manager.register): the persist label is set by the CALLER
(DataFrame.persist()), the manager only tracks references.
Rule 2 (persist with action) -- via release(key, keep=new_frame),
Rule 3 (unpersist) -- via release_all(key).
"""

from hypex.dataset.storage_manager import StorageManager


class FakeFrame:
    """Spark DataFrame emulator: counts persist/unpersist calls, supports groupby/agg."""

    def __init__(self, name: str = "frame"):
        self.name = name
        self.persisted = False
        self.persist_calls = 0
        self.unpersist_calls = 0
        self.storage_level = None
        self.blocking_values = []

    def persist(self, storage_level=None):
        self.persist_calls += 1
        self.persisted = True
        self.storage_level = storage_level
        return self

    def unpersist(self, blocking=True):
        self.unpersist_calls += 1
        self.blocking_values.append(blocking)
        self.persisted = False
        return self

    def groupby(self, *args, **kwargs):
        return FakeFrame(f"{self.name}.groupby")

    def agg(self, *args, **kwargs):
        return FakeFrame(f"{self.name}.agg")


class DeadFrame(FakeFrame):
    """Dead plan: unpersist raises an error (e.g., session is already closed)."""

    def unpersist(self, blocking=True):
        raise RuntimeError("session is closed")


class NoBlockingFrame(FakeFrame):
    """Backend whose unpersist method does not accept the blocking parameter."""

    def unpersist(self):
        self.unpersist_calls += 1
        self.blocking_values.append(None)
        self.persisted = False
        return self


class SparkNavigation:
    """Mini-harness of your navigation class (persist_key + manager delegation)."""

    def __init__(self, data, *, persist_key=None, **kwargs):
        self._data = data
        # Key is created once with the root navigator and inherited
        # by all derivatives (the string is copied by deepcopy and any wrappers).
        self._persist_key = persist_key or StorageManager.new_key()

    @property
    def data(self):
        return self._data

    @property
    def persist_key(self):
        return self._persist_key

    def with_new_persist_key(self):
        """Branch point with independent cache lifecycle."""
        self._persist_key = StorageManager.new_key()
        return self

    def persist(self, storage_level=None, *, action_triggered=False):
        manager = StorageManager.instance()
        # Label is set by the caller; manager only tracks the reference.
        if storage_level is None:
            self._data.persist()
        else:
            self._data.persist(storage_level)
        manager.register(self._persist_key, self._data, storage_level=storage_level)
        if action_triggered:
            # Rule 2: action triggered -- old references are no longer needed,
            # only the frame that triggered the action remains.
            manager.release(self._persist_key, keep=self._data)
        return self

    def unpersist(self, blocking=True):
        # Rule 3: release everything under the branch key.
        StorageManager.instance().release_all(self._persist_key, blocking=blocking)
        return self

    def groupby(self, *args, **kwargs):
        return SparkNavigation(self._data.groupby(*args, **kwargs), persist_key=self._persist_key)

    def agg(self, *args, **kwargs):
        return SparkNavigation(self._data.agg(*args, **kwargs), persist_key=self._persist_key)


class NavWrapper:
    """Wrapper around SparkNavigation (wrapper scenario: object is hidden inside)."""

    def __init__(self, nav: SparkNavigation):
        self._nav = nav

    def persist(self, *args, **kwargs):
        self._nav.persist(*args, **kwargs)
        return self

    def unpersist(self, *args, **kwargs):
        self._nav.unpersist(*args, **kwargs)
        return self

    @property
    def persist_key(self):
        return self._nav.persist_key
