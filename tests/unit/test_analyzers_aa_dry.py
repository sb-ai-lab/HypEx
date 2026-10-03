"""AADryTestAnalyzer (uniformity of A/A p-values) and AAScoreAnalyzer dry-score merge."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from hypex.analyzers.aa import AADryTestAnalyzer, AAScoreAnalyzer
from hypex.dataset import (
    Dataset,
    ExperimentData,
    FeatureRole,
    SmallDataset,
    StatisticRole,
    TargetRole,
)
from hypex.utils import BackendsEnum, ExperimentDataEnum
from hypex.utils.constants import ID_SPLIT_SYMBOL as S


def _table(**columns) -> Dataset:
    df = pd.DataFrame(columns)
    return Dataset(
        roles={c: StatisticRole() for c in df.columns},
        data=df,
        backend=BackendsEnum.pandas,
    )


def _uniform(n: int = 400, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).uniform(0, 1, n)


def _skewed(n: int = 400, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).beta(0.2, 5, n)


def _experiment_data(table: Dataset, targets=("y",)) -> ExperimentData:
    ds = Dataset(
        roles={t: TargetRole() for t in targets} | {"x": FeatureRole()},
        data=pd.DataFrame({**{t: [1.0, 2.0] for t in targets}, "x": [1.0, 2.0]}),
        backend=BackendsEnum.pandas,
    )
    data = ExperimentData(ds)
    return data.set_value(
        ExperimentDataEnum.analysis_tables,
        f"ParamsExperiment{S}hash{S}",
        table,
        key="",
    )


def _frame(ds) -> pd.DataFrame:
    return ds.backend_data.data


def test_uniform_test_passes_for_uniform_pvalues() -> None:
    table = _table(**{"y GroupTTest p-value test_1": _uniform()})
    result = AADryTestAnalyzer()._uniform_test(table, ["y"])
    frame = _frame(result)
    assert list(frame.index) == ["y TTest test_1"]
    assert bool(frame.loc["y TTest test_1", "pass"]) is True
    assert float(frame.loc["y TTest test_1", "p-value"]) > 0.05


def test_uniform_test_fails_for_skewed_pvalues() -> None:
    table = _table(**{"y GroupTTest p-value test_1": _skewed()})
    frame = _frame(AADryTestAnalyzer()._uniform_test(table, ["y"]))
    assert bool(frame.loc["y TTest test_1", "pass"]) is False
    assert float(frame.loc["y TTest test_1", "p-value"]) < 0.05


def test_uniform_test_only_uses_ttest_pvalue_columns_of_targets() -> None:
    table = _table(
        **{
            "y GroupTTest p-value test_1": _uniform(),
            "y GroupKSTest p-value test_1": _skewed(),
            "z GroupTTest p-value test_1": _skewed(),
            "y GroupTTest pass test_1": np.ones(400),
        }
    )
    frame = _frame(AADryTestAnalyzer()._uniform_test(table, ["y"]))
    assert list(frame.index) == ["y TTest test_1"]


def test_uniform_test_without_targets_returns_empty() -> None:
    result = AADryTestAnalyzer()._uniform_test(_table(a=[1.0]), None)
    assert result.is_empty()


def test_uniform_test_two_targets_one_row_each() -> None:
    table = _table(
        **{
            "y GroupTTest p-value test_1": _uniform(seed=1),
            "w GroupTTest p-value test_1": _skewed(seed=1),
        }
    )
    frame = _frame(AADryTestAnalyzer()._uniform_test(table, ["y", "w"]))
    assert bool(frame.loc["y TTest test_1", "pass"]) is True
    assert bool(frame.loc["w TTest test_1", "pass"]) is False


def test_plot_creates_one_axis_per_target_and_reports_fpr() -> None:
    table = _table(
        **{"y GroupTTest p-value test_1": np.r_[np.full(10, 0.01), np.full(90, 0.5)]}
    )
    analyzer = AADryTestAnalyzer(alpha=0.05)
    analyzer._uniform_test(table, ["y"])
    fig = analyzer._fig
    assert fig is not None
    assert len(fig.axes) == 1
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("FPR = 10.0%" in t for t in texts)
    assert fig._suptitle.get_text() == "AA-test: diagnostic with zero effect"


def test_plot_two_targets_makes_two_axes() -> None:
    table = _table(
        **{
            "y GroupTTest p-value test_1": _uniform(),
            "w GroupTTest p-value test_1": _uniform(seed=2),
        }
    )
    analyzer = AADryTestAnalyzer()
    analyzer._uniform_test(table, ["y", "w"])
    assert len(analyzer._fig.axes) == 2


def test_execute_stores_table_and_figure() -> None:
    table = _table(**{"y GroupTTest p-value test_1": _uniform()})
    analyzer = AADryTestAnalyzer()
    out = analyzer.execute(_experiment_data(table))
    stored = out.analysis_tables[analyzer.id]
    assert list(_frame(stored).index) == ["y TTest test_1"]
    assert out.variables[analyzer.id][""] is analyzer._fig


def test_build_dry_score_dataset_empty_placeholder() -> None:
    assert AADryTestAnalyzer._build_dry_score_dataset({}) is not None


# ---------------------------------------------------------------------------
# AAScoreAnalyzer merges the dry-score verdict into its own pass flag
# ---------------------------------------------------------------------------
def _score_table() -> Dataset:
    return _table(
        **{
            "y GroupTTest pass test_1": [1, 1, 1, 1, 1],
            "y GroupTTest p-value test_1": [0.5, 0.4, 0.6, 0.7, 0.3],
        }
    )


def _dry_table(passed: bool) -> SmallDataset:
    return SmallDataset(
        roles={"p-value": StatisticRole(), "pass": StatisticRole()},
        data=pd.DataFrame(
            {"p-value": [0.5], "pass": [passed]}, index=["y TTest test_1"]
        ),
    )


def test_dry_score_failure_overrides_good_pass_rate() -> None:
    data = _experiment_data(_score_table())
    data = data.set_value(
        ExperimentDataEnum.analysis_tables,
        f"AADryTestAnalyzer{S}{S}",
        _dry_table(False),
        key="",
    )
    analyzer = AAScoreAnalyzer(alpha=0.05)
    out = analyzer._analyze_aa_score(data, _score_table())
    frame = _frame(out.analysis_tables[analyzer.id])
    assert bool(frame.loc["y TTest test_1", "pass"]) is False


def test_dry_score_success_keeps_pass_flag() -> None:
    data = _experiment_data(_score_table())
    data = data.set_value(
        ExperimentDataEnum.analysis_tables,
        f"AADryTestAnalyzer{S}{S}",
        _dry_table(True),
        key="",
    )
    analyzer = AAScoreAnalyzer(alpha=0.05)
    out = analyzer._analyze_aa_score(data, _score_table())
    frame = _frame(out.analysis_tables[analyzer.id])
    assert bool(frame.loc["y TTest test_1", "pass"]) is True


def test_no_dry_score_leaves_pass_flag_untouched() -> None:
    analyzer = AAScoreAnalyzer(alpha=0.05)
    out = analyzer._analyze_aa_score(_experiment_data(_score_table()), _score_table())
    frame = _frame(out.analysis_tables[analyzer.id])
    assert bool(frame.loc["y TTest test_1", "pass"]) is True


# ---------------------------------------------------------------------------
# AAScoreAnalyzer best split selection / application
# ---------------------------------------------------------------------------
def _split_data(n: int = 200, strat_nan: int = 0) -> ExperimentData:
    from hypex.dataset import StratificationRole

    strat = np.array(["a", "b"] * (n // 2), dtype=object)
    if strat_nan:
        strat[:strat_nan] = None
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "s": strat})
    ds = Dataset(
        roles={"x": FeatureRole(), "s": StratificationRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )
    return ExperimentData(ds)


def test_set_best_split_applies_rebuilt_splitter() -> None:
    analyzer = AAScoreAnalyzer()
    splitter_id = f"AASplitter{S}rs 7{S}"
    out = analyzer._set_best_split(_split_data(), splitter_id)
    assert out.variables[analyzer.id]["best splitter"] == splitter_id
    split_cols = [c for c in out.additional_fields.columns if c.endswith("best")]
    assert len(split_cols) == 1
    labels = out.additional_fields.backend_data.data[split_cols[0]]
    assert set(labels) == {"control", "test_1"}
    assert len(labels) == 200
    # best splitter does not store per-group subsets
    assert not any(k.endswith("best") for k in out.groups)


def test_set_best_split_is_reproducible_for_same_id() -> None:
    splitter_id = f"AASplitter{S}rs 7{S}"
    a = AAScoreAnalyzer()._set_best_split(_split_data(), splitter_id)
    b = AAScoreAnalyzer()._set_best_split(_split_data(), splitter_id)
    col_a = next(c for c in a.additional_fields.columns if c.endswith("best"))
    col_b = next(c for c in b.additional_fields.columns if c.endswith("best"))
    assert (
        a.additional_fields.backend_data.data[col_a].tolist()
        == b.additional_fields.backend_data.data[col_b].tolist()
    )


def test_set_best_split_drops_rows_with_missing_stratification() -> None:
    splitter_id = f"AASplitterWithStratification{S}rs 3{S}"
    out = AAScoreAnalyzer()._set_best_split(_split_data(strat_nan=10), splitter_id)
    assert len(out.ds) == 190
    col = next(c for c in out.additional_fields.columns if c.endswith("best"))
    assert out.additional_fields.backend_data.data[col].notna().sum() == 190


def test_get_best_split_stores_statistics_row_and_returns_id() -> None:
    table = _table(
        splitter_id=[f"AASplitter{S}rs 1{S}", f"AASplitter{S}rs 2{S}"],
        **{"mean test score": [0.2, 0.9]},
    )
    analyzer = AAScoreAnalyzer()
    # no per-feature weights -> the first row wins
    split_id, data = analyzer._get_best_split(_split_data(), table)
    assert split_id == f"AASplitter{S}rs 1{S}"
    stored = next(
        v for k, v in data.analysis_tables.items() if k.startswith("AAScoreAnalyzer")
    )
    assert stored.backend_data.data["mean test score"].tolist() == [0.2]


def test_get_best_split_uses_weighted_score_when_weights_exist() -> None:
    table = _table(
        splitter_id=[f"AASplitter{S}rs 1{S}", f"AASplitter{S}rs 2{S}"],
        **{
            "mean test score": [0.1, 0.9],
            "y GroupTTest p-value test_1": [0.3, 0.8],
        },
    )
    analyzer = AAScoreAnalyzer()
    analyzer._feature_weights = {"y TTest test_1": 1.0}
    split_id, _ = analyzer._get_best_split(_split_data(), table)
    assert split_id == f"AASplitter{S}rs 2{S}"


def test_analyze_best_split_runs_selection_and_application() -> None:
    table = _table(
        splitter_id=[f"AASplitter{S}rs 4{S}", f"AASplitter{S}rs 5{S}"],
        **{"mean test score": [0.9, 0.1]},
    )
    out = AAScoreAnalyzer()._analyze_best_split(_split_data(), table)
    assert any(c.endswith("best") for c in out.additional_fields.columns)


def test_get_if_param_scores_absent_returns_none() -> None:
    assert AAScoreAnalyzer._get_if_param_scores(_split_data()) is None


def test_get_if_param_scores_returns_stored_table() -> None:
    table = _table(a=[1, 2])
    data = _split_data().set_value(
        ExperimentDataEnum.analysis_tables, f"IfParamsExperiment{S}h{S}", table, key=""
    )
    assert (
        AAScoreAnalyzer._get_if_param_scores(data)
        is data.analysis_tables[f"IfParamsExperiment{S}h{S}"]
    )
