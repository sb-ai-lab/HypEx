"""Unit tests for StorageManager (your storage_manager.py, without Dataset and without Spark)."""

import threading

import pytest

from hypex.dataset.storage_manager import PersistEntry, StorageManager

from .fakes import DeadFrame, FakeFrame, NoBlockingFrame


class TestSingleton:
    def test_instance_returns_same_object(self, fresh_manager):
        assert StorageManager.instance() is fresh_manager
        assert StorageManager.instance() is StorageManager.instance()

    def test_set_instance_replaces_singleton(self, fresh_manager):
        other = StorageManager()
        try:
            StorageManager.set_instance(other)
            assert StorageManager.instance() is other
        finally:
            StorageManager.reset(release=True)

    def test_reset_releases_frames_and_creates_new_instance(self, fresh_manager):
        key = fresh_manager.new_key()
        frame = FakeFrame()
        frame.persist()
        fresh_manager.register(key, frame)
        StorageManager.reset(release=True)  # FIX 1: previously raised TypeError
        assert frame.unpersist_calls == 1, "reset must release accumulated entries"
        assert StorageManager.instance() is not fresh_manager


class TestKeys:
    def test_new_keys_are_unique(self, fresh_manager):
        keys = {fresh_manager.new_key() for _ in range(500)}
        assert len(keys) == 500

    def test_new_key_format(self, fresh_manager):
        key = fresh_manager.new_key()
        assert isinstance(key, str)
        assert len(key) == 32
        int(key, 16)  # hex string without hyphens


class TestRegister:
    def test_manager_does_not_touch_frames(self, fresh_manager):
        """According to your scheme, the label is set by the caller: manager only tracks references."""
        key = fresh_manager.new_key()
        frame = FakeFrame()
        entry = fresh_manager.register(key, frame, storage_level="MEMORY_AND_DISK")
        assert frame.persist_calls == 0, "manager should not set the label itself"
        assert frame.unpersist_calls == 0
        assert entry.storage_level == "MEMORY_AND_DISK"

    def test_accumulation(self, fresh_manager):
        key = fresh_manager.new_key()
        f1, f2 = FakeFrame(), FakeFrame()
        f1.persist()
        fresh_manager.register(key, f1)
        f2.persist()
        fresh_manager.register(key, f2)
        assert fresh_manager.count(key) == 2
        assert fresh_manager.frames(key) == [f1, f2]

    def test_register_returns_persist_entry(self, fresh_manager):
        key = fresh_manager.new_key()
        frame = FakeFrame()
        entry = fresh_manager.register(key, frame)
        assert isinstance(entry, PersistEntry)
        assert entry.frame is frame

    def test_same_frame_twice_is_idempotent(self, fresh_manager):
        """FIX 3 (regression): previously deduplication did not work and the plan was duplicated."""
        key = fresh_manager.new_key()
        frame = FakeFrame()
        first = fresh_manager.register(key, frame)
        second = fresh_manager.register(key, frame)
        assert first is second, "should return the same entry"
        assert fresh_manager.count(key) == 1

    def test_triple_register_then_release_all_unpersists_once(self, fresh_manager):
        key = fresh_manager.new_key()
        frame = FakeFrame()
        for _ in range(3):
            fresh_manager.register(key, frame)
        assert fresh_manager.release_all(key) == 1
        assert frame.unpersist_calls == 1, "duplicates should not inflate cleanup"


class TestReleaseKeep:
    """Rule 2: persist with action -- releases all old references, only keep remains."""

    def test_keeps_only_trigger_frame(self, fresh_manager):
        key = fresh_manager.new_key()
        f1, f2, f3 = FakeFrame(), FakeFrame(), FakeFrame()
        for f in (f1, f2, f3):
            f.persist()
            fresh_manager.register(key, f)
        n = fresh_manager.release(key, keep=f3)
        assert n == 2
        assert f1.unpersist_calls == 1 and f2.unpersist_calls == 1
        assert f3.unpersist_calls == 0 and f3.persisted
        assert fresh_manager.frames(key) == [f3]
        assert fresh_manager.count(key) == 1

    def test_keep_not_registered_releases_all(self, fresh_manager):
        key = fresh_manager.new_key()
        f1, f2 = FakeFrame(), FakeFrame()
        fresh_manager.register(key, f1)
        fresh_manager.register(key, f2)
        assert fresh_manager.release(key, keep=FakeFrame("unknown")) == 2
        assert not f1.persisted and not f2.persisted
        assert key not in fresh_manager.keys()

    def test_keep_none_releases_everything(self, fresh_manager):
        key = fresh_manager.new_key()
        f1, f2 = FakeFrame(), FakeFrame()
        fresh_manager.register(key, f1)
        fresh_manager.register(key, f2)
        assert fresh_manager.release(key) == 2
        assert not f1.persisted and not f2.persisted
        assert fresh_manager.count(key) == 0

    def test_missing_key_returns_zero(self, fresh_manager):
        assert fresh_manager.release("missing", keep=FakeFrame()) == 0

    def test_missing_key_does_not_create_bucket(self, fresh_manager):
        fresh_manager.release("missing", keep=FakeFrame())
        assert "missing" not in fresh_manager.keys()

    def test_blocking_default_true(self, fresh_manager):
        key = fresh_manager.new_key()
        frame = FakeFrame()
        fresh_manager.register(key, frame)
        fresh_manager.release(key)
        assert frame.blocking_values == [True]

    def test_blocking_false_passthrough(self, fresh_manager):
        key = fresh_manager.new_key()
        frame = FakeFrame()
        fresh_manager.register(key, frame)
        fresh_manager.release(key, blocking=False)
        assert frame.blocking_values == [False]


class TestReleaseAll:
    """Rule 3: unpersist -- cleanup of all references in the branch."""

    def test_releases_all_and_returns_count(self, fresh_manager):
        key = fresh_manager.new_key()
        f1, f2 = FakeFrame(), FakeFrame()
        fresh_manager.register(key, f1)
        fresh_manager.register(key, f2)
        assert fresh_manager.release_all(key) == 2
        assert f1.unpersist_calls == 1 and f2.unpersist_calls == 1
        assert key not in fresh_manager.keys()
        assert fresh_manager.count(key) == 0

    def test_missing_key_returns_zero(self, fresh_manager):
        assert fresh_manager.release_all("missing") == 0

    def test_blocking_default_true(self, fresh_manager):
        key = fresh_manager.new_key()
        frame = FakeFrame()
        fresh_manager.register(key, frame)
        fresh_manager.release_all(key)
        assert frame.blocking_values == [True]

    def test_blocking_false_passthrough(self, fresh_manager):
        key = fresh_manager.new_key()
        frame = FakeFrame()
        fresh_manager.register(key, frame)
        fresh_manager.release_all(key, blocking=False)
        assert frame.blocking_values == [False]


class TestIntrospection:
    def test_entries_snapshot_copy(self, fresh_manager):
        key = fresh_manager.new_key()
        frame = FakeFrame()
        fresh_manager.register(key, frame)
        snapshot = fresh_manager.entries(key)
        snapshot.clear()
        assert fresh_manager.count(key) == 1, "snapshot should not affect the registry"

    def test_frames_returns_frames(self, fresh_manager):
        """FIX 5 (regression): previously frames() crashed with AttributeError due to typo 'enties'."""
        key = fresh_manager.new_key()
        f1, f2 = FakeFrame(), FakeFrame()
        fresh_manager.register(key, f1)
        fresh_manager.register(key, f2)
        assert fresh_manager.frames(key) == [f1, f2]

    def test_keys_and_total_entries(self, fresh_manager):
        k1, k2 = fresh_manager.new_key(), fresh_manager.new_key()
        fresh_manager.register(k1, FakeFrame())
        fresh_manager.register(k1, FakeFrame())
        fresh_manager.register(k2, FakeFrame())
        assert set(fresh_manager.keys()) == {k1, k2}
        assert fresh_manager.total_entries() == 3

    def test_clear_releases_everything(self, fresh_manager):
        k1, k2 = fresh_manager.new_key(), fresh_manager.new_key()
        frames = [FakeFrame(), FakeFrame(), FakeFrame()]
        fresh_manager.register(k1, frames[0])
        fresh_manager.register(k1, frames[1])
        fresh_manager.register(k2, frames[2])
        assert fresh_manager.clear() == 3
        assert all(not f.persisted for f in frames)
        assert fresh_manager.total_entries() == 0
        assert fresh_manager.keys() == []

    def test_repr(self, fresh_manager):
        key = fresh_manager.new_key()
        fresh_manager.register(key, FakeFrame())
        fresh_manager.register(key, FakeFrame())
        r = repr(fresh_manager)
        assert "StorageManager" in r, "FIX 7: repr should show the actual class name"
        assert "entries=2" in r


class TestRobustness:
    def test_dead_frame_warns_but_cleanup_continues(self, fresh_manager):
        key = fresh_manager.new_key()
        dead, alive = DeadFrame(), FakeFrame()
        fresh_manager.register(key, dead)
        fresh_manager.register(key, alive)
        with pytest.warns(RuntimeWarning):
            fresh_manager.release_all(key)
        assert not alive.persisted, "live plan must be released despite the dead neighbor"

    def test_frame_without_unpersist_method_warns(self, fresh_manager):
        class Bare:
            pass

        key = fresh_manager.new_key()
        bare, alive = Bare(), FakeFrame()
        fresh_manager.register(key, bare)
        fresh_manager.register(key, alive)
        with pytest.warns(RuntimeWarning):
            fresh_manager.release_all(key)
        assert not alive.persisted
        assert fresh_manager.count(key) == 0

    def test_frame_without_blocking_kwarg_falls_back(self, fresh_manager):
        """FIX 8: backend without blocking parameter should not receive warning, but should be released."""
        key = fresh_manager.new_key()
        frame = NoBlockingFrame()
        fresh_manager.register(key, frame)
        fresh_manager.release_all(key)
        assert not frame.persisted
        assert frame.unpersist_calls == 1


class TestThreadSafety:
    def test_concurrent_register_and_release_all(self, fresh_manager):
        key = fresh_manager.new_key()
        frames = [FakeFrame() for _ in range(60)]
        errors = []

        def worker(chunk):
            try:
                for f in chunk:
                    fresh_manager.register(key, f)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(frames[i::4],)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert fresh_manager.count(key) == 60
        assert fresh_manager.release_all(key) == 60
        assert all(not f.persisted for f in frames)
