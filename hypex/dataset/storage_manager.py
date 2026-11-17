from __future__ import annotations

import threading
import uuid
import warnings
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .abstract import DatasetBase


class PersistEntry:
    """A single registered persisted plan.

    Attributes:
        frame: A Dataset snapshot of a persisted plan. Exposes ``same_cache``
            and ``release_cache`` through the Dataset interface.
        storage_level: Storage level name the frame was persisted with.
    """

    __slots__ = ("frame", "storage_level")

    def __init__(self, frame: DatasetBase, storage_level: str | None = None):
        self.frame = frame
        self.storage_level = storage_level

    def __repr__(self) -> str:
        return (
            f"PersistEntry(frame={type(self.frame).__name__}, "
            f"storage_level={self.storage_level!r})"
        )


class StorageManager:
    """Track cache ownership through the Dataset interface, independently of backend."""

    _instance: StorageManager | None = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._storage: dict[str, list[PersistEntry]] = {}
        # Reentrant: release callbacks may query the manager back.
        self._lock = threading.RLock()

    # ── singlton ────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> StorageManager:
        """Return the process-wide manager instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def set_instance(cls, manager: StorageManager) -> None:
        with cls._instance_lock:
            cls._instance = manager

    @classmethod
    def reset(cls, *, release: bool = True) -> None:
        with cls._instance_lock:
            old, cls._instance = cls._instance, None
        if release and old is not None:
            old.clear()

    # ── keygen ────────────────────────────────────────────────

    @staticmethod
    def new_key() -> str:
        """Generate a new unique lineage key."""
        return uuid.uuid4().hex

    # ── main api ────────────────────────────────────────────────

    @staticmethod
    def _same_plan(first, second):
        return first is second or (second is not None and first.same_cache(second))

    def persist(
        self,
        data: DatasetBase,
        storage_level: str,
        action: Literal["count", "head", "none"],
    ) -> None:
        """Acquire and register a Dataset cache before materializing it."""
        if action not in ("count", "head", "none"):
            raise ValueError(f"Invalid action: {action!r}")
        with self._lock:
            cached = data.cache(storage_level)
            if cached is None:
                return
            key = cached.persist_key
            self.register(key, cached, storage_level=storage_level)
        cached.materialize(action)
        if action == "count":
            self.release(key, keep=cached, blocking=True)

    def _release_unowned(self, entries, blocking):
        for entry in entries:
            has_owner = any(
                self._same_plan(entry.frame, other.frame)
                for bucket in self._storage.values()
                for other in bucket
            )
            if not has_owner:
                try:
                    entry.frame.release_cache(blocking=blocking)
                except Exception as exc:  # Cleanup must not break the pipeline.
                    warnings.warn(
                        f"StorageManager: release_cache({entry!r}) failed: {exc!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    def register(
        self, key: str, data: DatasetBase, storage_level: str | None = None
    ) -> PersistEntry:
        """Register a persisted ``frame`` in the lineage bucket ``key``.

        Idempotent: equivalent cached plans are registered once per key.

        Args:
            key: Dataset lineage key.
            data: Dataset cache snapshot. Native objects stay inside its backend.
            storage_level: Storage level name, for introspection.

        Returns:
            The registry entry for the frame.
        """
        with self._lock:
            bucket = self._storage.setdefault(key, [])
            for entry in bucket:
                if self._same_plan(entry.frame, data):
                    entry.storage_level = storage_level
                    return entry
            entry = PersistEntry(data, storage_level=storage_level)
            bucket.append(entry)
            return entry

    def release(
        self,
        key: str,
        keep: DatasetBase | None = None,
        blocking: bool = True,
    ) -> int:
        """Release all accumulated references of the lineage except ``keep``.

        Rule 2 of the agreed semantics: called right after the action of a
        ``persist`` — every previously accumulated reference is unpersisted
        (the cache mark is removed from those DAG points), while the frame
        that triggered the action (``keep``) stays cached and becomes the
        only entry of the bucket. Plans still owned by another key stay cached.

        Args:
            key: Lineage key.
            keep: Frame that must remain cached (the one that triggered the
                action). ``None`` releases everything (rule 3 behaviour).
            blocking: Passed through to ``Dataset.release_cache``.

        Returns:
            Number of released references.
        """
        with self._lock:
            bucket = self._storage.get(key, [])
            kept, released = [], []
            for entry in bucket:
                if self._same_plan(entry.frame, keep):
                    kept.append(entry)
                else:
                    released.append(entry)

            if kept:
                self._storage[key] = kept
            else:
                self._storage.pop(key, None)
            self._release_unowned(released, blocking)
        return len(released)

    def release_all(self, key: str, blocking: bool = True) -> int:
        """Release **all** accumulated references of the lineage (rule 3).

        Args:
            key: Lineage key.
            blocking: Passed through to ``Dataset.release_cache``.

        Returns:
            Number of released references.
        """
        return self.release(key, blocking=blocking)

    # ── introspection ────────────────────────────────────────────────

    def entries(self, key: str) -> list[PersistEntry]:
        """Snapshot of the lineage bucket (safe to iterate)."""
        with self._lock:
            return list(self._storage.get(key, []))

    def frames(self, key: str) -> list[DatasetBase]:
        """Snapshot of the persisted frames of the lineage."""
        return [entry.frame for entry in self.entries(key)]

    def count(self, key: str) -> int:
        """Number of accumulated references in the lineage bucket."""
        with self._lock:
            return len(self._storage.get(key, ()))

    def keys(self):
        """All lineage keys that still hold at least one reference."""
        with self._lock:
            return list(self._storage.keys())

    def total_entries(self) -> int:
        """Total number of tracked references across all lineages."""
        with self._lock:
            return sum(len(bucket) for bucket in self._storage.values())

    def clear(self, blocking: bool = True) -> int:
        """Release everything (tests / session shutdown helper)."""
        with self._lock:
            return sum(
                self.release_all(key, blocking=blocking) for key in list(self._storage)
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StorageManager(lineages={len(self.keys())}, entries={self.total_entries()})"
