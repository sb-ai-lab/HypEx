"""Minimal frame doubles for StorageManager unit tests."""


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
