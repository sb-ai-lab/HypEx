"""Scenario tests: SparkNavigation mini-harness + StorageManager (emulators, without Spark).

They test end-to-end scenarios: your bug from the project, action mode,
key inheritance during transformations, deepcopy and wrappers.
"""

import copy

from hypex.dataset.storage_manager import StorageManager

from .fakes import FakeFrame, NavWrapper, SparkNavigation


class TestUserRepro:
    def test_two_plans_both_released(self):
        """Exactly your bug: persist -> groupby/agg -> persist -> unpersist."""
        manager = StorageManager.instance()
        p1 = FakeFrame("P1")

        nav = SparkNavigation(p1)
        nav.persist()
        nav = nav.groupby("feat_cat").agg("sum")
        nav.persist()

        frames = manager.frames(nav.persist_key)
        assert len(frames) == 2, "both plans should be registered under the key"

        nav.unpersist()
        assert all(f.unpersist_calls == 1 for f in frames), "each plan is released exactly once"
        assert all(not f.persisted for f in frames)
        assert not p1.persisted
        assert manager.total_entries() == 0

    def test_action_persist_keeps_only_trigger(self):
        """Rule 2 end-to-end: persist(action_triggered=True) releases old, keeps new."""
        manager = StorageManager.instance()
        p1 = FakeFrame("P1")
        nav = SparkNavigation(p1)
        nav.persist()
        assert p1.persisted

        nav2 = SparkNavigation(FakeFrame("P2"), persist_key=nav.persist_key)
        nav2.persist(action_triggered=True)

        assert not p1.persisted, "old plan should be released"
        assert nav2.data.persisted
        frames = manager.frames(nav2.persist_key)
        assert len(frames) == 1 and frames[0] is nav2.data

        nav2.unpersist()
        assert manager.total_entries() == 0


class TestKeyInheritance:
    def test_child_inherits_key(self):
        nav = SparkNavigation(FakeFrame())
        child = nav.groupby("x").agg("s")
        assert child.persist_key == nav.persist_key

    def test_long_chain_inherits_key(self):
        nav = SparkNavigation(FakeFrame())
        leaf = nav.groupby("x").agg("s").groupby("y").agg("s")
        assert leaf.persist_key == nav.persist_key

    def test_sibling_branches_share_key_by_design(self):
        nav = SparkNavigation(FakeFrame())
        a = nav.groupby("x").agg("s")
        b = nav.groupby("y").agg("s")
        assert a.persist_key == b.persist_key == nav.persist_key

    def test_with_new_persist_key_breaks_lineage(self):
        nav = SparkNavigation(FakeFrame())
        child = nav.groupby("x").agg("s").with_new_persist_key()
        assert child.persist_key != nav.persist_key


class TestDeepcopy:
    def test_copy_shares_key(self):
        nav = SparkNavigation(FakeFrame())
        twin = copy.deepcopy(nav)
        assert twin.persist_key == nav.persist_key

    def test_unpersist_via_copy_releases_original_plan(self):
        p1 = FakeFrame("P1")
        nav = SparkNavigation(p1)
        nav.persist()
        twin = copy.deepcopy(nav)
        twin.unpersist()
        assert not p1.persisted, "copy should be able to release the original plan"
        assert StorageManager.instance().total_entries() == 0


class TestWrapper:
    def test_wrapper_persist_unpersist(self):
        manager = StorageManager.instance()
        p1 = FakeFrame("P1")

        wrapped = NavWrapper(SparkNavigation(p1))
        wrapped.persist()

        nested = NavWrapper(wrapped._nav.groupby("x").agg("s"))
        nested.persist()

        frames = manager.frames(wrapped.persist_key)
        assert len(frames) == 2

        nested.unpersist()  # unpersist via wrapper
        assert all(not f.persisted for f in frames)
        assert not p1.persisted
