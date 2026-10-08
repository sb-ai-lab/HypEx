"""Tests for Experiment, OnRoleExperiment, CycledExperiment, GroupExperiment, ParamsExperiment."""

from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import (
    Dataset,
    ExperimentData,
    FeatureRole,
    GroupingRole,
    TargetRole,
)
from hypex.executor import Executor, IfExecutor
from hypex.experiments import (
    CycledExperiment,
    Experiment,
    GroupExperiment,
    IfParamsExperiment,
    OnRoleExperiment,
    ParamsExperiment,
)
from hypex.reporters import Reporter
from hypex.utils import BackendsEnum


class _Recorder(Executor):
    """Records its name (and the active temporary roles) into a shared log."""

    def __init__(self, name: str, log: list, alpha: int = 0, beta: int = 0, key=""):
        self.name = name
        self.log = log
        self.alpha = alpha
        self.beta = beta
        super().__init__(key)

    def execute(self, data):
        self.log.append((self.name, self.key, sorted(data.ds.tmp_roles)))
        return data


class _Transformer(_Recorder):
    @property
    def _is_transformer(self) -> bool:
        return True

    def execute(self, data):
        data.ds.data  # touch
        self.log.append(("transform", id(data)))
        return data


class _CountReporter(Reporter):
    """Reports number of rows, alpha of the first recorder and an iteration marker."""

    def __init__(self, recorder: _Recorder | None = None):
        self.recorder = recorder
        self.calls = 0

    def report(self, data):
        self.calls += 1
        return {
            "rows": len(data.ds),
            "alpha": self.recorder.alpha if self.recorder else -1,
            "call": self.calls,
        }


class _DatasetReporter(_CountReporter):
    """Same as _CountReporter but returns a Dataset (rename-able, unlike SmallDataset)."""

    def report(self, data):
        row = super().report(data)
        return Dataset(
            roles={k: FeatureRole() for k in row},
            data=pd.DataFrame([row]),
            backend=BackendsEnum.pandas,
        )


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": ["a", "a", "b", "b", "b"],
            "y1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y2": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )


@pytest.fixture
def data(frame) -> ExperimentData:
    roles = {"g": GroupingRole(), "y1": TargetRole(), "y2": TargetRole()}
    return ExperimentData(Dataset(roles=roles, data=frame, backend=BackendsEnum.pandas))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def test_executors_run_in_order(data) -> None:
    log: list = []
    Experiment([_Recorder("a", log), _Recorder("b", log), _Recorder("c", log)]).execute(
        data
    )
    assert [entry[0] for entry in log] == ["a", "b", "c"]


def test_experiment_key_is_propagated_to_executors(data) -> None:
    log: list = []
    Experiment([_Recorder("a", log)], key="K").execute(data)
    assert log[0][1] == "K"


def test_empty_experiment_returns_same_data(data) -> None:
    assert Experiment([]).execute(data) is data


def test_transformer_flag_is_detected() -> None:
    log: list = []
    assert Experiment([_Recorder("a", log)]).transformer is False
    assert Experiment([_Recorder("a", log), _Transformer("t", log)]).transformer is True


def test_explicit_transformer_flag_overrides_detection() -> None:
    log: list = []
    assert Experiment([_Transformer("t", log)], transformer=False).transformer is False
    assert Experiment([_Recorder("a", log)], transformer=True).transformer is True


def test_transformer_pipeline_works_on_a_copy(data) -> None:
    log: list = []
    Experiment([_Transformer("t", log)]).execute(data)
    assert log[0][1] != id(data)


def test_non_transformer_pipeline_works_in_place(data) -> None:
    seen = []

    class _Capture(Executor):
        def execute(self, d):
            seen.append(d)
            return d

    Experiment([_Capture()]).execute(data)
    assert seen[0] is data


def test_get_executor_ids_groups_by_class() -> None:
    log: list = []
    a, t = _Recorder("a", log), _Transformer("t", log)
    ids = Experiment([a, t]).get_executor_ids([_Recorder, _Transformer])
    assert ids[_Recorder] == [a.id, t.id]  # _Transformer is a _Recorder
    assert ids[_Transformer] == [t.id]


def test_get_executor_ids_without_classes_is_empty() -> None:
    assert Experiment([]).get_executor_ids(None) == {}
    assert Experiment([]).get_executor_ids([]) == {}


def test_set_params_by_class_reaches_nested_executors() -> None:
    log: list = []
    a = _Recorder("a", log)
    Experiment([a]).set_params({_Recorder: {"alpha": 7}})
    assert a.alpha == 7


def test_set_params_rejects_invalid_keys() -> None:
    with pytest.raises(ValueError, match="params must be"):
        Experiment([]).set_params({1: {"a": 1}})


def test_experiment_stores_results_via_set_value(data) -> None:
    exp = Experiment([])
    exp._set_value(data, pd.DataFrame())  # smoke: id-based storage
    assert exp.id in data.analysis_tables


# ---------------------------------------------------------------------------
# OnRoleExperiment
# ---------------------------------------------------------------------------
def test_on_role_runs_iterative_executor_once_per_target(data) -> None:
    log: list = []
    OnRoleExperiment([_Recorder("r", log)], role=TargetRole()).execute(data)
    assert [entry[2] for entry in log] == [["y1"], ["y2"]]


def test_on_role_runs_all_executors_for_each_target_in_order(data) -> None:
    log: list = []
    OnRoleExperiment(
        [_Recorder("a", log), _Recorder("b", log)], role=TargetRole()
    ).execute(data)
    assert [(e[0], e[2][0]) for e in log] == [
        ("a", "y1"),
        ("b", "y1"),
        ("a", "y2"),
        ("b", "y2"),
    ]


def test_on_role_without_matching_columns_returns_data_unchanged(data) -> None:
    log: list = []
    out = OnRoleExperiment([_Recorder("a", log)], role=FeatureRole()).execute(data)
    assert out is data and log == []


def test_on_role_clears_tmp_roles_afterwards(data) -> None:
    OnRoleExperiment([_Recorder("a", [])], role=TargetRole()).execute(data)
    assert data.ds.tmp_roles == {}


def test_on_role_restores_executor_list(data) -> None:
    executors = [_Recorder("a", [])]
    experiment = OnRoleExperiment(executors, role=TargetRole())
    experiment.execute(data)
    assert experiment.executors is executors


def test_on_role_accepts_sequence_of_roles() -> None:
    experiment = OnRoleExperiment([], role=[TargetRole(), FeatureRole()])
    assert len(experiment.role) == 2


def test_on_role_vector_executor_runs_once_with_all_targets(data) -> None:
    from hypex.comparators import StatsTTest

    log: list = []

    class _Vector(StatsTTest):
        def execute(self, d):
            log.append(sorted(d.ds.tmp_roles))
            return d

    OnRoleExperiment(
        [_Vector(grouping_role=GroupingRole())], role=TargetRole()
    ).execute(data)
    assert log == [["y1", "y2"]]


# ---------------------------------------------------------------------------
# CycledExperiment
# ---------------------------------------------------------------------------
def test_cycled_runs_n_iterations_and_collects_results(data) -> None:
    log: list = []
    reporter = _DatasetReporter()
    out = CycledExperiment([_Recorder("a", log)], reporter, n_iterations=3).execute(
        data
    )
    assert len(log) == 3
    assert reporter.calls == 3
    table = next(iter(out.analysis_tables.values()))
    frame = table.backend_data.data
    assert len(frame) == 3
    assert sorted(frame["call"]) == [1, 2, 3]


def test_cycled_iteration_keys_are_indices(data) -> None:
    log: list = []
    CycledExperiment([_Recorder("a", log)], _DatasetReporter(), n_iterations=3).execute(
        data
    )
    assert [entry[1] for entry in log] == ["0", "1", "2"]


@pytest.mark.xfail(
    strict=True,
    reason="Issue: CycledExperiment/GroupExperiment/ParamsExperiment define "
    "generate_params_hash (no leading underscore), so Executor._generate_id never uses it "
    "and the id ignores n_iterations/reporter",
)
def test_cycled_id_depends_on_iterations() -> None:
    reporter = _CountReporter()
    assert CycledExperiment([], reporter, 3).id != CycledExperiment([], reporter, 5).id


def test_cycled_params_hash_method_content() -> None:
    reporter = _CountReporter()
    assert (
        CycledExperiment([], reporter, 4).generate_params_hash() == "_CountReporter x 4"
    )


def test_cycled_zero_iterations_raises(data) -> None:
    with pytest.raises(IndexError):
        CycledExperiment([], _CountReporter(), n_iterations=0).execute(data)


# ---------------------------------------------------------------------------
# GroupExperiment
# ---------------------------------------------------------------------------
def test_group_experiment_runs_once_per_group(data) -> None:
    log: list = []
    reporter = _DatasetReporter()
    out = GroupExperiment(
        [_Recorder("a", log)], reporter, searching_role=GroupingRole()
    ).execute(data)
    assert [entry[1] for entry in log] == ["a", "b"]
    table = next(iter(out.analysis_tables.values())).backend_data.data
    assert table["a rows"].iloc[0] == 2
    assert table["b rows"].iloc[0] == 3


def test_group_experiment_prefixes_columns_with_group_key(data) -> None:
    out = GroupExperiment(
        [], _DatasetReporter(), searching_role=GroupingRole()
    ).execute(data)
    cols = next(iter(out.analysis_tables.values())).columns
    assert {"a rows", "a alpha", "b rows", "b alpha"} <= set(cols)


def test_group_experiment_with_dict_reporter(data) -> None:
    GroupExperiment([], _CountReporter(), searching_role=GroupingRole()).execute(data)


# ---------------------------------------------------------------------------
# ParamsExperiment
# ---------------------------------------------------------------------------
def _params_experiment(log, **kwargs):
    recorder = _Recorder("r", log)
    reporter = _CountReporter(recorder)
    experiment = ParamsExperiment(
        [recorder],
        reporter,
        params={_Recorder: {"alpha": [1, 2], "beta": [10, 20, 30]}},
        **kwargs,
    )
    return recorder, experiment


def test_flat_params_is_cartesian_product() -> None:
    _, experiment = _params_experiment([])
    experiment.params = experiment.params
    assert len(experiment.flat_params) == 6
    assert {frozenset(p[_Recorder].items()) for p in experiment.flat_params} == {
        frozenset({("alpha", a), ("beta", b)}) for a in (1, 2) for b in (10, 20, 30)
    }


def test_flat_params_for_multiple_classes() -> None:
    class _Other(_Recorder):
        pass

    experiment = ParamsExperiment(
        [],
        _CountReporter(),
        params={_Recorder: {"alpha": [1, 2]}, _Other: {"beta": [5, 6, 7]}},
    )
    experiment.params = experiment.params
    assert len(experiment.flat_params) == 6
    assert all(set(p) == {_Recorder, _Other} for p in experiment.flat_params)


def test_params_experiment_runs_every_combination(data) -> None:
    log: list = []
    _recorder, experiment = _params_experiment(log)
    out = experiment.execute(data)
    assert len(log) == 6
    table = next(iter(out.analysis_tables.values())).backend_data.data
    assert len(table) == 6
    assert sorted(table["alpha"]) == [1, 1, 1, 2, 2, 2]


def test_params_experiment_without_stopping_criterion_never_stops() -> None:
    _, experiment = _params_experiment([])
    assert experiment._stopping_criterion_met(None) is False


class _StopAfter(IfExecutor):
    def __init__(self, limit: int):
        self.limit = limit
        self.calls = 0
        super().__init__()

    def check_rule(self, data, **kwargs) -> bool:
        self.calls += 1
        return self.calls >= self.limit


def test_params_experiment_stops_when_criterion_met(data) -> None:
    log: list = []
    _recorder, experiment = _params_experiment(log, stopping_criterion=_StopAfter(2))
    out = experiment.execute(data)
    assert len(log) == 2
    assert len(next(iter(out.analysis_tables.values())).backend_data.data) == 2


def test_if_params_experiment_returns_first_satisfying_iteration(data) -> None:
    log: list = []
    recorder = _Recorder("r", log)
    reporter = _CountReporter(recorder)
    experiment = IfParamsExperiment(
        [recorder],
        reporter,
        params={_Recorder: {"alpha": [1, 2, 3]}},
        stopping_criterion=_StopAfter(2),
    )
    out = experiment.execute(data)
    assert len(log) == 2
    table = next(iter(out.analysis_tables.values())).backend_data.data
    assert table["alpha"].tolist() == [2]


def test_if_params_experiment_returns_data_when_never_satisfied(data) -> None:
    recorder = _Recorder("r", [])
    experiment = IfParamsExperiment(
        [recorder],
        _CountReporter(recorder),
        params={_Recorder: {"alpha": [1, 2]}},
        stopping_criterion=_StopAfter(99),
    )
    out = experiment.execute(data)
    assert out is data
    assert len(out.analysis_tables) == 0


# ---------------------------------------------------------------------------
# Dispatch to Experiment.execute (not the ExperimentWithReporter MRO hop)
# ---------------------------------------------------------------------------
def test_cycled_dispatches_each_iteration_to_experiment_execute(
    data, monkeypatch
) -> None:
    """Behaviour is identical to ``super(ExperimentWithReporter, self).execute``
    (the class adds no ``execute``); this pins the explicit dispatch."""
    calls: list = []
    original = Experiment.execute

    def spy(self, d):
        calls.append(self)
        return original(self, d)

    monkeypatch.setattr(Experiment, "execute", spy)
    experiment = CycledExperiment([], _DatasetReporter(), n_iterations=3)
    experiment.execute(data)
    assert calls == [experiment] * 3


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_group_experiment_dispatches_each_group_to_experiment_execute(
    data, monkeypatch
) -> None:
    calls: list = []
    original = Experiment.execute

    def spy(self, d):
        calls.append(self.key)
        return original(self, d)

    monkeypatch.setattr(Experiment, "execute", spy)
    GroupExperiment([], _DatasetReporter(), searching_role=GroupingRole()).execute(data)
    assert calls == ["a", "b"]
