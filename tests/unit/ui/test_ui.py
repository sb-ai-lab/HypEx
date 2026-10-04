"""Tests for Output, ExperimentShell and the AA/AB/Homogeneity/Matching outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex import AATest, ABTest, HomogeneityTest, Matching
from hypex.dataset import (
    Dataset,
    ExperimentData,
    FeatureRole,
    InfoRole,
    PreTargetRole,
    SmallDataset,
    TargetRole,
    TreatmentRole,
)
from hypex.executor import Executor
from hypex.experiments import Experiment
from hypex.reporters import Reporter
from hypex.ui.aa import AAOutput
from hypex.ui.ab import ABOutput, CupacOutput
from hypex.ui.base import ExperimentShell, Output
from hypex.ui.homo import HomoOutput
from hypex.ui.matching import MatchingOutput
from hypex.utils import ID_SPLIT_SYMBOL as S
from hypex.utils import BackendsEnum
from hypex.utils.enums import RenameEnum


class _ValueReporter(Reporter):
    def __init__(self, value):
        self.value = value
        self.seen = []

    def report(self, data):
        self.seen.append(data)
        return self.value


class _NoOp(Executor):
    def __init__(self, marker: str = "x"):
        self.marker = marker
        self.calls = 0
        super().__init__()

    def execute(self, data):
        self.calls += 1
        return data


def _df(n: int = 300, groups: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "treat": rng.randint(0, groups, n),
            "x": rng.normal(0, 1, n),
        }
    )
    df["y"] = df.x * 2 + (df.treat == 1) * 1.0 + rng.normal(0, 1, n)
    df["y_pre"] = df.x * 2 + rng.normal(0, 1, n)
    return df


ROLES = {
    "id": InfoRole(),
    "treat": TreatmentRole(),
    "x": FeatureRole(),
    "y": TargetRole(),
    "y_pre": PreTargetRole(),
}


def _dataset(df=None, backend=BackendsEnum.pandas, session=None) -> Dataset:
    return Dataset(
        roles=dict(ROLES),
        data=(df if df is not None else _df()).copy(),
        backend=backend,
        session=session if backend == BackendsEnum.spark else None,
    )


def _frame(ds) -> pd.DataFrame:
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def test_output_extract_populates_resume_and_additional_reports() -> None:
    resume, extra = _ValueReporter("resume!"), _ValueReporter({"k": 1})
    output = Output(resume, {"details": extra})
    data = ExperimentData(_dataset())
    output.extract(data)
    assert output.resume == "resume!"
    assert output.details == {"k": 1}
    assert resume.seen == [data] and extra.seen == [data]


def test_output_without_additional_reporters() -> None:
    output = Output(_ValueReporter(1))
    assert output.additional_reporters == {}
    output.extract(ExperimentData(_dataset()))
    assert output.resume == 1


def test_output_keeps_experiment_data_reference() -> None:
    output = Output(_ValueReporter(1))
    data = ExperimentData(_dataset())
    output.extract(data)
    assert output._experiment_data is data


def _split_dataset() -> SmallDataset:
    return SmallDataset(
        roles={f"a{S}b": InfoRole(), "plain": InfoRole()},
        data=pd.DataFrame(
            {f"a{S}b": [1, 2], "plain": [3, 4]}, index=[f"i{S}1", f"i{S}2"]
        ),
    )


def test_replace_splitters_columns_mode() -> None:
    result = Output._replace_splitters(_split_dataset(), RenameEnum.columns)
    assert list(result.columns) == ["a b", "plain"]
    assert list(result.roles) == ["a b", "plain"]
    assert list(result.index) == [f"i{S}1", f"i{S}2"]


def test_replace_splitters_index_mode() -> None:
    result = Output._replace_splitters(_split_dataset(), RenameEnum.index)
    assert list(result.index) == ["i 1", "i 2"]
    assert list(result.columns) == [f"a{S}b", "plain"]


def test_replace_splitters_all_mode() -> None:
    result = Output._replace_splitters(_split_dataset(), RenameEnum.all)
    assert list(result.columns) == ["a b", "plain"]
    assert list(result.index) == ["i 1", "i 2"]


def test_replace_splitters_empty_dataset_is_returned_unchanged() -> None:
    empty = SmallDataset.create_empty()
    assert Output._replace_splitters(empty) is empty


# ---------------------------------------------------------------------------
# ExperimentShell
# ---------------------------------------------------------------------------
def test_shell_wraps_dataset_runs_experiment_and_returns_output() -> None:
    executor = _NoOp()
    reporter = _ValueReporter("done")
    output = Output(reporter)
    shell = ExperimentShell(Experiment([executor]), output)
    returned = shell.execute(_dataset())
    assert returned is output
    assert executor.calls == 1
    assert output.resume == "done"
    assert isinstance(reporter.seen[0], ExperimentData)


def test_shell_accepts_experiment_data() -> None:
    reporter = _ValueReporter("done")
    data = ExperimentData(_dataset())
    ExperimentShell(Experiment([]), Output(reporter)).execute(data)
    seen = reporter.seen[0]
    # The shell copies the container so that set_value() does not leak into the caller's data.
    assert isinstance(seen, ExperimentData)
    assert seen is not data
    assert list(seen.ds.columns) == list(data.ds.columns)


def test_shell_applies_experiment_params() -> None:
    executor = _NoOp()
    ExperimentShell(
        Experiment([executor]), Output(_ValueReporter(0)), {_NoOp: {"marker": "y"}}
    )
    assert executor.marker == "y"


def test_shell_exposes_experiment() -> None:
    experiment = Experiment([])
    assert (
        ExperimentShell(experiment, Output(_ValueReporter(0))).experiment is experiment
    )


def test_shell_is_reusable() -> None:
    executor = _NoOp()
    shell = ExperimentShell(Experiment([executor]), Output(_ValueReporter(0)))
    shell.execute(_dataset())
    shell.execute(_dataset())
    assert executor.calls == 2


@pytest.mark.spark
def test_shell_auto_persist_spark_unpersists_only_what_it_persisted(
    spark_session,
) -> None:
    dataset = _dataset(backend=BackendsEnum.spark, session=spark_session)
    ExperimentShell(Experiment([_NoOp()]), Output(_ValueReporter(0))).execute(dataset)
    assert not dataset.is_persisted


@pytest.mark.spark
def test_shell_auto_persist_can_be_disabled(spark_session) -> None:
    dataset = _dataset(backend=BackendsEnum.spark, session=spark_session)
    seen = []

    class _Check(_NoOp):
        def execute(self, data):
            seen.append(data.ds.is_persisted)
            return data

    ExperimentShell(
        Experiment([_Check()]), Output(_ValueReporter(0)), auto_persist=False
    ).execute(dataset)
    assert seen == [False]


@pytest.mark.spark
def test_shell_auto_persist_persists_during_run(spark_session) -> None:
    dataset = _dataset(backend=BackendsEnum.spark, session=spark_session)
    seen = []

    class _Check(_NoOp):
        def execute(self, data):
            seen.append(data.ds.is_persisted)
            return data

    ExperimentShell(Experiment([_Check()]), Output(_ValueReporter(0))).execute(dataset)
    assert seen == [True]


# ---------------------------------------------------------------------------
# Real outputs through the public entry points
# ---------------------------------------------------------------------------
def test_ab_output_two_groups() -> None:
    output = ABTest().execute(_dataset())
    assert isinstance(output, ABOutput)
    resume = _frame(output.resume)
    assert resume.loc[0, "feature"] == "y"
    assert {
        "control mean",
        "test mean",
        "difference",
        "TTest pass",
        "TTest p-value",
    } <= set(resume.columns)
    assert resume["TTest pass"].iloc[0] in ("OK", "NOT OK")
    assert isinstance(output.multitest, str)
    sizes = _frame(output.sizes)
    assert sizes["control size"].iloc[0] + sizes["test size"].iloc[0] == 300
    assert isinstance(output.cupac, CupacOutput)
    assert output.cupac.variance_reductions is None
    assert output.cupac.feature_importances is None
    assert "no CUPAC data" in repr(output.cupac)


def test_ab_output_three_groups_has_multitest_table() -> None:
    output = ABTest(multitest_method="holm").execute(_dataset(_df(groups=3)))
    assert not isinstance(output.multitest, str)
    table = _frame(output.multitest)
    assert len(table) == 2
    assert sorted(table["group"]) == ["1", "2"]
    assert (
        table["new p-value"].astype(float) >= table["old p-value"].astype(float) - 1e-12
    ).all()


def test_ab_output_values_match_manual_computation() -> None:
    df = _df()
    output = ABTest().execute(_dataset(df))
    resume = _frame(output.resume).iloc[0]
    control, test = df.y[df.treat == 0], df.y[df.treat == 1]
    assert float(resume["control mean"]) == pytest.approx(control.mean(), abs=1e-6)
    assert float(resume["test mean"]) == pytest.approx(test.mean(), abs=1e-6)
    assert float(resume["difference"]) == pytest.approx(
        test.mean() - control.mean(), abs=1e-6
    )


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="Issue: ABOutput.variance_reduction_report calls ABTestReporter.report_variance_reductions, which no longer exists",
)
def test_ab_output_variance_reduction_report_without_cuped() -> None:
    output = ABTest().execute(_dataset())
    assert "No variance reduction data" in output.variance_reduction_report


def test_ab_output_with_cuped_adds_cuped_feature() -> None:
    output = ABTest(cuped_features={"y": "y_pre"}).execute(_dataset())
    features = list(_frame(output.resume)["feature"])
    assert "y" in features and "y_cuped" in features


def test_cupac_output_repr_describes_content() -> None:
    cupac = CupacOutput()
    assert repr(cupac) == "CupacOutput(no CUPAC data available)"


@pytest.mark.xfail(
    strict=True,
    raises=DeprecationWarning,
    reason="Issue: HomogeneityTest builds HomoOutput with the deprecated HomoDatasetReporter, so construction emits a DeprecationWarning",
)
def test_homo_output_resume() -> None:
    output = HomogeneityTest().execute(_dataset())
    assert isinstance(output, HomoOutput)
    resume = _frame(output.resume)
    assert {"feature", "group", "TTest pass", "KSTest pass"} <= set(resume.columns)


def test_aa_output_structure() -> None:
    output = AATest(n_iterations=4, random_states=range(4)).execute(_dataset())
    assert isinstance(output, AAOutput)
    for attribute in (
        "resume",
        "best_split",
        "experiments",
        "aa_score",
        "best_split_statistics",
    ):
        assert hasattr(output, attribute), attribute
    assert len(_frame(output.experiments)) == 4
    assert not any(S in c for c in output.experiments.columns)


def test_aa_output_reproducible_with_fixed_states() -> None:
    first = AATest(n_iterations=3, random_states=[1, 2, 3]).execute(_dataset())
    second = AATest(n_iterations=3, random_states=[1, 2, 3]).execute(_dataset())
    pd.testing.assert_frame_equal(_frame(first.experiments), _frame(second.experiments))


@pytest.mark.xfail(
    strict=True,
    raises=DeprecationWarning,
    reason="Issue: Matching builds MatchingOutput with the deprecated MatchingDictReporter, so construction emits a DeprecationWarning",
)
@pytest.mark.spark
def test_matching_output_structure_on_spark(spark_session) -> None:
    output = Matching().execute(
        _dataset(backend=BackendsEnum.spark, session=spark_session)
    )
    assert isinstance(output, MatchingOutput)
    for attribute in ("resume", "full_data", "indexes", "quality_results"):
        assert hasattr(output, attribute), attribute
    assert len(output.indexes) == 300


@pytest.mark.xfail(
    strict=True,
    raises=(DeprecationWarning, AttributeError),
    reason="Issue: Matching builds MatchingOutput with the deprecated MatchingDictReporter "
    "(DeprecationWarning) and, past that, MatchingOutput._extract_full_data calls the missing "
    "self._match_pandas, so Matching cannot run on the pandas backend",
)
def test_matching_output_on_pandas() -> None:
    Matching().execute(_dataset())


# ---------------------------------------------------------------------------
# AAOutput: NaN in a "pass" column
# ---------------------------------------------------------------------------
def test_aa_output_extract_experiments_keeps_nan_pass_and_casts_others() -> None:
    """A float NaN pass value is left as NaN (math.isnan guard); the rest become bool.

    Equivalent to the former ``val != val`` check, so this is a behaviour guard.
    """
    table = Dataset(
        roles={},
        data=pd.DataFrame(
            {
                f"TTest{S}x{S}pass": [np.nan, 1.0, 0.0],
                f"KSTest{S}x{S}pass": ["True", "no", "ok"],
                "n": [1, 2, 3],
            }
        ),
        backend=BackendsEnum.pandas,
        default_role=InfoRole(),
    )
    experiment_data = ExperimentData(_dataset())
    experiment_data.analysis_tables[f"ParamsExperiment{S}a"] = table
    output = AAOutput()
    output._extract_experiments(experiment_data)
    frame = _frame(output.experiments)
    assert np.isnan(frame["TTest x pass"].iloc[0])
    assert frame["TTest x pass"].tolist()[1:] == [True, False]
    assert frame["KSTest x pass"].tolist() == [True, False, True]
    assert frame["n"].tolist() == [1, 2, 3]
