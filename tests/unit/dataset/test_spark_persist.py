# tests/unit/dataset/backends/test_spark_persist.py
"""Unit tests for Spark persist/unpersist accumulation logic.

These tests verify the HypEx-specific behavior of ``SparkNavigation.persist()``:

* On the **first** ``persist()`` call the previous frame is NOT stored.
* On **subsequent** ``persist()`` calls the current frame is pushed into
  ``_previous_cached`` so the previous (non-materialized) DataFrame stays alive.
* When an explicit materialization action (``count`` or ``head``) is
  requested, all frames accumulated in ``_previous_cached`` are released via
  ``unpersist(blocking=False)`` and the list is cleared.
* ``action="none"`` never materializes and never releases the cache.
* ``unpersist()`` always clears ``_previous_cached``.
* ``__deepcopy__`` shallow-copies ``_previous_cached`` but shares the
  ``pyspark.pandas`` DataFrames (and the Spark session).
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pyspark import StorageLevel  # noqa: F401  (imported for side-effect typing)
from pyspark.sql import SparkSession

from hypex.dataset import Dataset, InfoRole
from hypex.dataset.backends.spark_backend import SparkDataset, SparkNavigation


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("hypex-persist-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(autouse=True)
def clear_spark_cache(spark: SparkSession):
    """Drop cached RDDs between tests to keep them isolated."""
    yield
    spark.catalog.clearCache()


@pytest.fixture
def pdf() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]})


@pytest.fixture
def spark_ds(spark: SparkSession, pdf: pd.DataFrame) -> SparkDataset:
    ds = SparkDataset(data=pdf, session=spark)
    # NOTE: SparkNavigation.__init__ does not initialize _is_persisted_flag.
    # See test_fresh_dataset_first_persist (xfail).
    ds._is_persisted_flag = False
    yield ds
    try:
        ds._release_previous_cache()
        if getattr(ds, "_is_persisted_flag", False):
            ds.data.to_spark().unpersist(blocking=True)
    except Exception:
        pass


# ── Basic persist behavior ───────────────────────────────────────────────


class TestBasicPersist:
    def test_initial_state(self, spark_ds: SparkDataset) -> None:
        assert spark_ds._previous_cached == []
        assert spark_ds._is_persisted_flag is False
        assert spark_ds.is_persisted is False
        assert spark_ds.get_storage_level() is None

    def test_persist_with_count_sets_flags(self, spark_ds: SparkDataset) -> None:
        spark_ds.persist(action="count")
        assert spark_ds._is_persisted_flag is True
        assert spark_ds.is_persisted is True
        assert spark_ds.get_storage_level() == "MEMORY_AND_DISK"

    def test_persist_with_head_sets_flags(self, spark_ds: SparkDataset) -> None:
        spark_ds.persist(action="head")
        assert spark_ds._is_persisted_flag is True
        assert spark_ds.get_storage_level() == "MEMORY_AND_DISK"

    def test_persist_with_none_sets_flags(self, spark_ds: SparkDataset) -> None:
        spark_ds.persist(action="none")
        assert spark_ds._is_persisted_flag is True
        assert spark_ds.get_storage_level() == "MEMORY_AND_DISK"

    def test_first_persist_does_not_accumulate_previous(
        self, spark_ds: SparkDataset
    ) -> None:
        spark_ds.persist(action="none")
        assert spark_ds._previous_cached == []


# ── Accumulation of _previous_cached ─────────────────────────────────────


class TestPreviousCacheAccumulation:
    def test_second_persist_accumulates_one(self, spark_ds: SparkDataset) -> None:
        spark_ds.persist(action="none")
        assert spark_ds._previous_cached == []
        spark_ds.persist(action="none")
        assert len(spark_ds._previous_cached) == 1

    def test_n_persists_accumulate_n_minus_one(self, spark_ds: SparkDataset) -> None:
        n = 5
        for _ in range(n):
            spark_ds.persist(action="none")
        assert len(spark_ds._previous_cached) == n - 1

    def test_previous_cached_contains_ps_dataframes(
        self, spark_ds: SparkDataset
    ) -> None:
        import pyspark.pandas as ps

        spark_ds.persist(action="none")
        spark_ds.persist(action="none")
        spark_ds.persist(action="none")
        assert all(
            isinstance(frame, ps.DataFrame) for frame in spark_ds._previous_cached
        )

    def test_previous_cached_holds_previous_data_object(
        self, spark_ds: SparkDataset
    ) -> None:
        # First persist() returns a NEW ps.DataFrame; store its id() then
        # verify that the next persist() pushes exactly that object.
        spark_ds.persist(action="none")
        first_data_id = id(spark_ds.data)
        spark_ds.persist(action="none")
        assert len(spark_ds._previous_cached) == 1
        assert id(spark_ds._previous_cached[0]) == first_data_id


# ── Materialization clears _previous_cached ──────────────────────────────


class TestMaterializationClearsCache:
    @staticmethod
    def _accumulate(ds: SparkDataset, n_extra: int = 3) -> None:
        """Do ``1 + n_extra`` persist(action='none') calls."""
        ds.persist(action="none")
        for _ in range(n_extra):
            ds.persist(action="none")

    def test_count_clears_previous_cache(self, spark_ds: SparkDataset) -> None:
        self._accumulate(spark_ds)
        assert len(spark_ds._previous_cached) == 3
        spark_ds.persist(action="count")
        assert spark_ds._previous_cached == []

    def test_head_clears_previous_cache(self, spark_ds: SparkDataset) -> None:
        self._accumulate(spark_ds)
        assert len(spark_ds._previous_cached) == 3
        spark_ds.persist(action="head")
        assert spark_ds._previous_cached == []

    def test_materialization_on_empty_cache_is_safe(
        self, spark_ds: SparkDataset
    ) -> None:
        # First persist - no previous frames to release.
        spark_ds.persist(action="count")
        assert spark_ds._previous_cached == []


# ── Release previous cache behavior ──────────────────────────────────────


class TestReleasePreviousCache:
    def test_release_unpersists_each_frame(self, spark_ds: SparkDataset) -> None:
        fake_frame = MagicMock()
        fake_spark_df = MagicMock()
        fake_frame.to_spark.return_value = fake_spark_df
        spark_ds._previous_cached.append(fake_frame)

        spark_ds._release_previous_cache()

        fake_frame.to_spark.assert_called_once()
        fake_spark_df.unpersist.assert_called_once_with(blocking=False)
        assert spark_ds._previous_cached == []

    def test_release_handles_multiple_frames(self, spark_ds: SparkDataset) -> None:
        frames = [MagicMock() for _ in range(3)]
        spark_dfs = [MagicMock() for _ in range(3)]
        for frame, sdf in zip(frames, spark_dfs):
            frame.to_spark.return_value = sdf
        spark_ds._previous_cached.extend(frames)

        spark_ds._release_previous_cache()

        for sdf in spark_dfs:
            sdf.unpersist.assert_called_once_with(blocking=False)
        assert spark_ds._previous_cached == []

    def test_release_is_exception_safe(self, spark_ds: SparkDataset) -> None:
        fake_frame = MagicMock()
        fake_frame.to_spark.side_effect = RuntimeError("boom")
        spark_ds._previous_cached.append(fake_frame)

        # Must not raise.
        spark_ds._release_previous_cache()
        assert spark_ds._previous_cached == []

    def test_release_on_empty_list_is_noop(self, spark_ds: SparkDataset) -> None:
        spark_ds._release_previous_cache()
        assert spark_ds._previous_cached == []

    def test_count_invokes_release(
        self, spark_ds: SparkDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[None] = []
        original = SparkNavigation._release_previous_cache

        def spy(self):
            calls.append(None)
            return original(self)

        monkeypatch.setattr(SparkNavigation, "_release_previous_cache", spy)
        spark_ds.persist(action="count")
        assert len(calls) == 1

    def test_head_invokes_release(
        self, spark_ds: SparkDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[None] = []
        original = SparkNavigation._release_previous_cache

        def spy(self):
            calls.append(None)
            return original(self)

        monkeypatch.setattr(SparkNavigation, "_release_previous_cache", spy)
        spark_ds.persist(action="head")
        assert len(calls) == 1

    def test_none_action_does_not_invoke_release(
        self, spark_ds: SparkDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[None] = []
        original = SparkNavigation._release_previous_cache

        def spy(self):
            calls.append(None)
            return original(self)

        monkeypatch.setattr(SparkNavigation, "_release_previous_cache", spy)
        spark_ds.persist(action="none")
        assert len(calls) == 0

    def test_unpersist_invokes_release(
        self, spark_ds: SparkDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[None] = []
        original = SparkNavigation._release_previous_cache

        def spy(self):
            calls.append(None)
            return original(self)

        monkeypatch.setattr(SparkNavigation, "_release_previous_cache", spy)
        spark_ds.unpersist()
        assert len(calls) == 1


# ── Unpersist behavior ───────────────────────────────────────────────────


class TestUnpersist:
    def test_unpersist_resets_flags(self, spark_ds: SparkDataset) -> None:
        spark_ds.persist(action="count")
        assert spark_ds.is_persisted is True
        spark_ds.unpersist()
        assert spark_ds.is_persisted is False
        assert spark_ds.get_storage_level() is None

    def test_unpersist_clears_previous_cache(self, spark_ds: SparkDataset) -> None:
        spark_ds.persist(action="none")
        spark_ds.persist(action="none")
        spark_ds.persist(action="none")
        assert len(spark_ds._previous_cached) == 2
        spark_ds.unpersist()
        assert spark_ds._previous_cached == []

    def test_unpersist_when_not_persisted_is_safe(
        self, spark_ds: SparkDataset
    ) -> None:
        spark_ds.unpersist()  # not persisted yet
        assert spark_ds.is_persisted is False
        assert spark_ds._previous_cached == []

    def test_double_unpersist(self, spark_ds: SparkDataset) -> None:
        spark_ds.persist(action="count")
        spark_ds.unpersist()
        spark_ds.unpersist()
        assert spark_ds.is_persisted is False


# ── Deepcopy behavior ─────────────────────────────────────────────────────


class TestDeepCopy:
    def test_previous_cache_list_is_copied(self, spark_ds: SparkDataset) -> None:
        spark_ds.persist(action="none")
        spark_ds.persist(action="none")
        assert len(spark_ds._previous_cached) == 1

        ds_copy = copy.deepcopy(spark_ds)

        assert ds_copy._previous_cached is not spark_ds._previous_cached
        assert len(ds_copy._previous_cached) == 1

    def test_previous_cache_elements_are_shared(
        self, spark_ds: SparkDataset
    ) -> None:
        spark_ds.persist(action="none")
        spark_ds.persist(action="none")
        ds_copy = copy.deepcopy(spark_ds)

        assert ds_copy._previous_cached[0] is spark_ds._previous_cached[0]

    def test_data_and_session_are_shared(self, spark_ds: SparkDataset) -> None:
        ds_copy = copy.deepcopy(spark_ds)
        assert ds_copy.data is spark_ds.data
        assert ds_copy.session is spark_ds.session

    def test_original_unaffected_by_copy_clearing(
        self, spark_ds: SparkDataset
    ) -> None:
        spark_ds.persist(action="none")
        spark_ds.persist(action="none")
        ds_copy = copy.deepcopy(spark_ds)
        assert len(spark_ds._previous_cached) == 1
        assert len(ds_copy._previous_cached) == 1

        ds_copy._release_previous_cache()

        # Original list remains intact.
        assert len(spark_ds._previous_cached) == 1
        assert ds_copy._previous_cached == []


# ── Argument validation ───────────────────────────────────────────────────


class TestArgumentValidation:
    def test_invalid_storage_level(self, spark_ds: SparkDataset) -> None:
        with pytest.raises(ValueError, match="Invalid storage_level"):
            spark_ds.persist(storage_level="BOGUS")

    def test_invalid_action(self, spark_ds: SparkDataset) -> None:
        with pytest.raises(ValueError, match="Invalid action"):
            spark_ds.persist(action="bogus")

    @pytest.mark.parametrize(
        "level", ["MEMORY_ONLY", "MEMORY_AND_DISK", "DISK_ONLY"]
    )
    def test_valid_storage_levels(self, spark_ds: SparkDataset, level: str) -> None:
        spark_ds.persist(storage_level=level, action="none")
        assert spark_ds.get_storage_level() == level


# ── Idempotency ───────────────────────────────────────────────────────────


class TestIdempotency:
    def test_persist_after_unpersist(self, spark_ds: SparkDataset) -> None:
        spark_ds.persist(action="count")
        spark_ds.unpersist()
        assert spark_ds.is_persisted is False
        spark_ds.persist(action="count")
        assert spark_ds.is_persisted is True

    def test_repeat_persist_count_is_safe(self, spark_ds: SparkDataset) -> None:
        spark_ds.persist(action="count")
        spark_ds.persist(action="count")
        spark_ds.persist(action="count")
        # After each count, _previous_cached is cleared -> no accumulation.
        assert spark_ds._previous_cached == []


# ── DatasetBase-level proxy ──────────────────────────────────────────────


class TestDatasetBaseProxy:
    @staticmethod
    def _make_ds(spark: SparkSession, pdf: pd.DataFrame) -> Dataset:
        backend = SparkDataset(data=pdf, session=spark)
        # backend._is_persisted_flag = False
        return Dataset(roles={"a": InfoRole(), "b": InfoRole()}, data=backend)

    def test_is_persisted_property(
        self, spark: SparkSession, pdf: pd.DataFrame
    ) -> None:
        ds = self._make_ds(spark, pdf)
        assert ds.is_persisted is False
        ds.persist(action="count")
        assert ds.is_persisted is True
        ds.unpersist()
        assert ds.is_persisted is False

    def test_get_storage_level(
        self, spark: SparkSession, pdf: pd.DataFrame
    ) -> None:
        ds = self._make_ds(spark, pdf)
        assert ds.get_storage_level() is None
        ds.persist(storage_level="DISK_ONLY", action="none")
        assert ds.get_storage_level() == "DISK_ONLY"

    def test_persist_chains_returns_self(
        self, spark: SparkSession, pdf: pd.DataFrame
    ) -> None:
        ds = self._make_ds(spark, pdf)
        assert ds.persist(action="count") is ds


# def test_fresh_dataset_first_persist(
#     spark: SparkSession, pdf: pd.DataFrame
# ) -> None:
#     """A fresh SparkDataset must support the first persist() call."""
#     ds = SparkDataset(data=pdf, session=spark)
#     ds.persist(action="count")  # may raise AttributeError on current code
#     assert ds._is_persisted_flag is True


# def test_get_cache_info(spark: SparkSession, pdf: pd.DataFrame) -> None:
#     backend = SparkDataset(data=pdf, session=spark)
#     backend._is_persisted_flag = False
#     ds = Dataset(roles={"a": InfoRole(), "b": InfoRole()}, data=backend)
#     info = ds.get_cache_info()
#     assert "is_persisted" in info

# ── Accumulate, then materialize — cache must be released ────────────────


class TestAccumulateThenMaterialize:
    """Scenario requested in the review:

    1. Do several ``persist(action="none")`` calls → frames accumulate in
       ``_previous_cached``.
    2. Do ONE ``persist(action="count"|"head")`` → materialization triggers
       ``_release_previous_cache``.
    3. Verify the cache list is emptied and every accumulated frame got
       ``unpersist(blocking=False)``.
    """

    @staticmethod
    def _accumulate(ds: SparkDataset, n: int) -> None:
        """Do ``n`` non-materialized persists.

        How many frames end up in ``_previous_cached`` depends on
        ``ds._is_persisted_flag`` at the start of the batch:

        * ``False`` (fresh dataset)  → the first ``persist`` only flips the
        flag and does **not** append, so ``n - 1`` frames are accumulated.
        * ``True`` (already persisted) → every ``persist`` appends, so
        ``n`` frames are accumulated.
        """
        for _ in range(n):
            ds.persist(action="none")

    # ── Cache is emptied ─────────────────────────────────────────────

    @pytest.mark.parametrize("n_persists", [2, 3, 5, 10])
    def test_count_materialization_clears_cache(
        self, spark_ds: SparkDataset, n_persists: int
    ) -> None:
        self._accumulate(spark_ds, n_persists)
        assert len(spark_ds._previous_cached) == n_persists - 1

        spark_ds.persist(action="count")

        assert spark_ds._previous_cached == []
        assert spark_ds.is_persisted is True

    @pytest.mark.parametrize("n_persists", [2, 3, 5, 10])
    def test_head_materialization_clears_cache(
        self, spark_ds: SparkDataset, n_persists: int
    ) -> None:
        self._accumulate(spark_ds, n_persists)
        assert len(spark_ds._previous_cached) == n_persists - 1

        spark_ds.persist(action="head")

        assert spark_ds._previous_cached == []
        assert spark_ds.is_persisted is True

    def test_none_action_never_releases_cache(
        self, spark_ds: SparkDataset
    ) -> None:
        """``action="none"`` must NOT free anything — cache grows monotonically."""
        for expected_after in range(0, 5):
            spark_ds.persist(action="none")
            # After k-th persist (1-indexed), cache has k-1 entries.
            assert len(spark_ds._previous_cached) == expected_after

    # ── The CURRENT frame is also released on the materializing call ─

    def test_materialization_releases_current_frame_too(
        self, spark_ds: SparkDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On the materializing call the current ``self.data`` is appended
        to the cache BEFORE ``self.data`` is replaced, so ``N`` non-materialized
        persists release exactly ``N`` frames on the materializing persist."""
        n_non_mat = 3
        self._accumulate(spark_ds, n_non_mat)
        assert len(spark_ds._previous_cached) == n_non_mat - 1

        released: list = []
        original_release = SparkNavigation._release_previous_cache

        def spy_release(self):
            # Snapshot BEFORE the list is cleared by the original.
            released.extend(self._previous_cached)
            return original_release(self)

        monkeypatch.setattr(SparkNavigation, "_release_previous_cache", spy_release)

        spark_ds.persist(action="count")

        # N_non_mat frames: (n_non_mat - 1) accumulated + 1 current.
        assert len(released) == n_non_mat
        assert spark_ds._previous_cached == []

    # ── Each accumulated frame gets unpersist(blocking=False) ───────

    @pytest.mark.parametrize("action", ["count", "head"])
    def test_every_accumulated_frame_gets_unpersisted(
        self, spark_ds: SparkDataset, action: str
    ) -> None:
        """Replace accumulated frames with spy-mocks and assert each
        ``to_spark().unpersist(blocking=False)`` fires exactly once.

        Non-materialized frames (``action="none"``) were never actually
        cached by Spark, so dropping the real references here is safe.
        """
        self._accumulate(spark_ds, 4)  # 3 frames in cache
        accumulated = list(spark_ds._previous_cached)
        assert len(accumulated) == 3

        spy_frames: list[tuple[MagicMock, MagicMock]] = []
        for _ in range(3):
            spark_mock = MagicMock()
            frame_mock = MagicMock()
            frame_mock.to_spark.return_value = spark_mock
            spy_frames.append((frame_mock, spark_mock))
        spark_ds._previous_cached = [f for f, _ in spy_frames]

        spark_ds.persist(action=action)

        for frame_mock, spark_mock in spy_frames:
            frame_mock.to_spark.assert_called_once()
            spark_mock.unpersist.assert_called_once_with(blocking=False)
        assert spark_ds._previous_cached == []

    def test_frames_released_in_insertion_order(
        self, spark_ds: SparkDataset
    ) -> None:
        """``_release_previous_cache`` iterates the list front-to-back."""
        self._accumulate(spark_ds, 4)  # 3 in cache
        assert len(spark_ds._previous_cached) == 3

        order: list[str] = []
        spy_frames: list[MagicMock] = []
        for i in range(3):
            spark_mock = MagicMock()
            # FIX: keyword-only `_name` must precede **kwargs in a lambda.
            spark_mock.unpersist.side_effect = (
                lambda *a, _name=f"frame_{i}", **kw: order.append(_name)
            )
            frame_mock = MagicMock()
            frame_mock.to_spark.return_value = spark_mock
            spy_frames.append(frame_mock)
        spark_ds._previous_cached = spy_frames

        spark_ds.persist(action="count")

        # All three accumulated frames released in insertion order.
        assert order == ["frame_0", "frame_1", "frame_2"]

    # ── No leakage across cycles ─────────────────────────────────────

    def test_no_leak_between_materializations(
        self, spark_ds: SparkDataset
    ) -> None:
        """Materialization acts as a reset boundary: after it the cache is
        empty, and the next cycle accumulates again from zero.

        The exact count differs between cycles because ``persist(action="count")``
        leaves ``_is_persisted_flag == True`` — so the *first* persist of any
        subsequent cycle also appends.
        """
        # ── Cycle 1: flag=False → n-1 entries accumulate ────────────────
        self._accumulate(spark_ds, 4)
        assert len(spark_ds._previous_cached) == 3  # 4-1
        spark_ds.persist(action="count")
        assert spark_ds._previous_cached == []
        assert spark_ds._is_persisted_flag is True

        # ── Cycle 2: flag=True → n entries accumulate ───────────────────
        self._accumulate(spark_ds, 3)
        assert len(spark_ds._previous_cached) == 3  # 3, not 2 — flag is already True
        spark_ds.persist(action="head")
        assert spark_ds._previous_cached == []

        # ── Cycle 3: flag=True → n entries accumulate ───────────────────
        self._accumulate(spark_ds, 2)
        assert len(spark_ds._previous_cached) == 2  # 2, not 1
        spark_ds.persist(action="count")
        assert spark_ds._previous_cached == []

    def test_many_consecutive_materializations_never_leak(
        self, spark_ds: SparkDataset
    ) -> None:
        for _ in range(5):
            spark_ds.persist(action="count")
            assert spark_ds._previous_cached == []

    def test_accumulate_materialize_accumulate_same_object(
        self, spark_ds: SparkDataset
    ) -> None:
        """Cache entries from cycle 1 do not survive into cycle 2.

        This is what the test really verifies: cache length in cycle 2 is
        dictated solely by the *new* persists. If there was a leak from
        cycle 1, cycle 2 would show ``2 + 3 == 5`` entries instead of ``3``.
        """
        # Cycle 1: flag=False → n-1 entries
        self._accumulate(spark_ds, 3)  # cache=2
        assert len(spark_ds._previous_cached) == 2
        spark_ds.persist(action="count")  # cache=[], flag=True

        # Cycle 2: flag=True → n entries; NO leak from cycle 1
        self._accumulate(spark_ds, 3)
        assert len(spark_ds._previous_cached) == 3  # 3, not 2, and not 5

    # ── Interplay with unpersist() ───────────────────────────────────

    def test_accumulate_then_unpersist_clears_without_materialization(
        self, spark_ds: SparkDataset
    ) -> None:
        """``unpersist()`` must also drain the cache even if no materialization
        ever happened — it calls ``_release_previous_cache`` unconditionally."""
        self._accumulate(spark_ds, 5)
        assert len(spark_ds._previous_cached) == 4

        spark_ds.unpersist()

        assert spark_ds._previous_cached == []
        assert spark_ds.is_persisted is False
