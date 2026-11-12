from __future__ import annotations

import threading
import time
import uuid
import warnings
from typing import Any, Iterator


class PersistEntry:
    """A single registered persisted plan.

    Attributes:
        frame: The persisted object (a ``pyspark.sql.DataFrame`` — the DAG
            point carrying the cache mark). Must expose ``unpersist``.
        storage_level: Storage level name the frame was persisted with.
        created_at: Monotonic timestamp of registration (debugging aid).
    """

    __slots__ = ("created_at", "frame", "storage_level")

    def __init__(self, frame: Any, storage_level: str | None = None):
        self.frame = frame
        self.storage_level = storage_level
        self.created_at = time.monotonic()

    def __repr__(self) -> str:
        return (
            f"PersistEntry(frame={type(self.frame).__name__}, "
            f"storage_level={self.storage_level!r})"
        )

class StorageManager:
    _instance: StorageManager| None = None
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

    def register(
        self, key: str, frame: Any, storage_level: str | None = None
    ) -> PersistEntry:
        """Register a persisted ``frame`` in the lineage bucket ``key``.

        Idempotent: the very same frame object is never registered twice.

        Args:
            key: Lineage key (``SparkNavigation._persist_key``).
            frame: Persisted object; the persist mark is already set by the
                caller, the manager only tracks the reference.
            storage_level: Storage level name, for introspection.

        Returns:
            The registry entry for the frame.
        """
        with self._lock:
            bucket = self._storage.setdefault(key, [])
            for entry in bucket:
                if entry.frame is frame:
                    return entry
            entry = PersistEntry(frame, storage_level=storage_level)
            bucket.append(entry)
            return entry

    def release(
        self,
        key: str,
        keep: Any | None = None,
        blocking: bool = True,
    ) -> int:
        """Release all accumulated references of the lineage except ``keep``.

        Rule 2 of the agreed semantics: called right after the action of a
        ``persist`` — every previously accumulated reference is unpersisted
        (the cache mark is removed from those DAG points), while the frame
        that triggered the action (``keep``) stays cached and becomes the
        only entry of the bucket.

        Args:
            key: Lineage key.
            keep: Frame that must remain cached (the one that triggered the
                action). ``None`` releases everything (rule 3 behaviour).
            blocking: Passed through to ``DataFrame.unpersist``.

        Returns:
            Number of released references.
        """
        with self._lock:
            bucket = self._storage.get(key, [])
            kept = [entry for entry in bucket if entry.frame is keep]
            released = [entry for entry in bucket if entry.frame is not keep]

            if kept:
                self._storage[key] = kept
            else:
                self._storage.pop(key, None)
        self._unpersist_all(released, blocking=blocking)
        return len(released)

    def release_all(self, key: str, blocking: bool = True) -> int:
        """Release **all** accumulated references of the lineage (rule 3).

        Args:
            key: Lineage key.
            blocking: Passed through to ``DataFrame.unpersist``.

        Returns:
            Number of released references.
        """
        with self._lock:
            released = self._storage.pop(key, [])
        self._unpersist_all(released, blocking=blocking)
        return len(released)

    # ── introspection ────────────────────────────────────────────────

    def entries(self, key: str) -> list[PersistEntry]:
        """Snapshot of the lineage bucket (safe to iterate)."""
        with self._lock:
            return list(self._storage.get(key, []))

    def frames(self, key: str) -> list[Any]:
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
            buckets = self._storage
            self._storage = {}
        total = 0
        for bucket in buckets.values():
            self._unpersist_all(bucket, blocking=blocking)
            total += len(bucket)
        return total

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StorageManager(lineages={len(self.keys())}, entries={self.total_entries()})"


    # ── internals ────────────────────────────────────────────────
    @staticmethod
    def _unpersist_all(entries: Iterator[PersistEntry], blocking: bool) -> None:
        """Best-effort ``unpersist`` of every entry; failures are warned
        about but never propagate — cleanup must not break the pipeline."""
        for entry in entries:
            unpersist = getattr(entry.frame, "unpersist", None)
            if not callable(unpersist):
                warnings.warn(
                    f"PersistManager: {entry!r} has no `unpersist`; "
                    f"the reference is dropped without releasing the cache",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            try:
                unpersist(blocking=blocking)
                continue
            except TypeError:
                pass
            except Exception as exc: #best-effort cleanup
                warnings.warn(
                    f"PersistManager: unpersist({entry!r}) failed: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            try:
                unpersist()
            except Exception as exc: #best-effort cleanup
                warnings.warn(
                    f"StorageManager: unpersist({entry!r}) failed: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
