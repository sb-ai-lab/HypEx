"""Tests for OneAAStatAnalyzer, AAScoreAnalyzer and ABAnalyzer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from hypex.analyzers.aa import (
    AAScoreAnalyzer,
    OneAAStatAnalyzer,
    _is_passed,
    _mean_key,
    _parse_metric_col,
    _resolve_column_parts,
)
from hypex.analyzers.ab import ABAnalyzer
from hypex.comparators import GroupTTest
from hypex.dataset import (
    Dataset,
    ExperimentData,
    FeatureRole,
    StatisticRole,
    TargetRole,
    TreatmentRole,
)
from hypex.experiments.base import OnRoleExperiment
from hypex.splitters import AASplitter, AASplitterWithStratification
from hypex.utils import ABNTestMethodsEnum, BackendsEnum
from hypex.utils.constants import ID_SPLIT_SYMBOL as S

TOL = 1e-6


def _grouped_experiment(n: int = 100, groups=("a", "b", "c"), shift_c: float = 1.0, seed: int = 0):
    rng = np.random.RandomState(seed)
    frames = []
    for i, g in enumerate(groups):
        frames.append(
            pd.DataFrame(
                {
                    "g": g,
                    "y1": rng.normal(shift_c if g == "c" else 0.0, 1, n),
                    "y2": rng.normal(0, 1, n),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    ds = Dataset(
        roles={"g": TreatmentRole(), "y1": TargetRole(), "y2": TargetRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )
    experiment = OnRoleExperiment(
        [GroupTTest(compare_by="groups", grouping_role=TreatmentRole())], role=TargetRole()
    )
    return df, experiment.execute(ExperimentData(ds))


def _ttest_table(df, group, target):
    from scipy import stats

    return stats.ttest_ind(df[target][df.g == "a"], df[target][df.g == group], equal_var=False)


# ---------------------------------------------------------------------------
# helpers shared by the AA analyzers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [(1, True), (1.0, True), (0, False), (0.0, False), (0.5, False), ("OK", True),
     ("true", True), ("1", True), ("FAIL", False), ("", False), (np.float64(1.0), True)],
)
def test_is_passed(value, expected) -> None:
    assert _is_passed(value) is expected


def test_mean_key_format() -> None:
    assert _mean_key("GroupTTest", "p-value") == f"mean{S}GroupTTest{S}p-value{S}all"


@pytest.mark.parametrize(
    "col,expected",
    [
        (f"y{S}GroupTTest{S}pass{S}b", ("y", "GroupTTest", "b")),
        (f"y{S}GroupTTest{S}pass{S}b┆y", ("y", "GroupTTest", "b")),
        ("y GroupTTest p-value b", ("y", "GroupTTest", "b")),
        ("garbage", None),
    ],
)
def test_resolve_column_parts(col, expected) -> None:
    assert _resolve_column_parts(col) == expected


@pytest.mark.parametrize(
    "col,expected",
    [
        ("y GroupTTest p-value b", ("y", "GroupTTest", "p-value", "b")),
        ("y GroupTTest control mean b", ("y", "GroupTTest", "control mean", "b")),
        (f"a{S}b{S}c", ("a", "b", "c", "")),
        ("x┆stats", ("", "", "", "")),
        ("nothing here", ("", "", "", "")),
    ],
)
def test_parse_metric_col(col, expected) -> None:
    assert _parse_metric_col(col) == expected


# ---------------------------------------------------------------------------
# OneAAStatAnalyzer
# ---------------------------------------------------------------------------
def test_sanitize_nan_replaces_only_nan() -> None:
    result = OneAAStatAnalyzer._sanitize_nan({"a": float("nan"), "b": 0.3})
    assert result == {"a": 0.0, "b": 0.3}


def test_composite_score_prefers_stats_over_group() -> None:
    data = {
        _mean_key("StatsTTest", "p-value"): 0.4,
        _mean_key("GroupTTest", "p-value"): 0.9,
    }
    assert OneAAStatAnalyzer._compute_composite_score(data) == pytest.approx(0.4)


def test_composite_score_weights_t_ks_chi2() -> None:
    data = {
        _mean_key("GroupTTest", "p-value"): 0.6,   # weight 1
        _mean_key("GroupKSTest", "p-value"): 0.3,  # weight 2
        _mean_key("GroupChi2Test", "p-value"): 0.9,  # weight 2
    }
    expected = (1 * 0.6 + 2 * 0.3 + 2 * 0.9) / 5
    assert OneAAStatAnalyzer._compute_composite_score(data) == pytest.approx(expected, abs=TOL)


def test_composite_score_skips_missing_families() -> None:
    data = {_mean_key("GroupKSTest", "p-value"): 0.5}
    assert OneAAStatAnalyzer._compute_composite_score(data) == pytest.approx(0.5)


def test_composite_score_without_tests_is_zero() -> None:
    assert OneAAStatAnalyzer._compute_composite_score({}) == 0.0


def test_one_aa_execute_means_and_score() -> None:
    df, experiment_data = _grouped_experiment(groups=("a", "b"))
    out = OneAAStatAnalyzer().execute(experiment_data)
    analyzer = OneAAStatAnalyzer()
    table = out.analysis_tables[analyzer.id].backend_data.data
    row = table.iloc[0] if len(table) == 1 else table.iloc[:, 0]

    p_values = [_ttest_table(df, "b", t).pvalue for t in ("y1", "y2")]
    key = _mean_key("GroupTTest", "p-value")
    assert float(row[key]) == pytest.approx(np.mean(p_values), abs=TOL)
    # only the T-test family is present -> composite score equals its mean p-value
    assert float(row["mean test score"]) == pytest.approx(np.mean(p_values), abs=TOL)
    pass_key = _mean_key("GroupTTest", "pass")
    assert float(row[pass_key]) == pytest.approx(
        np.mean([p < 0.05 for p in p_values]), abs=TOL
    )


# ---------------------------------------------------------------------------
# AAScoreAnalyzer
# ---------------------------------------------------------------------------
def _score_table(pass_pattern: list[list[int]], pvalues: list[list[float]] | None = None) -> Dataset:
    """Score table with columns ``feature┴test┴pass┴group`` (one row per split)."""
    n_rows = len(pass_pattern[0])
    data = {}
    for i, col_pass in enumerate(pass_pattern):
        data[f"f{i}{S}GroupTTest{S}pass{S}test_1"] = col_pass
        if pvalues:
            data[f"f{i}{S}GroupTTest{S}p-value{S}test_1"] = pvalues[i]
    df = pd.DataFrame(data)
    return Dataset(
        roles={c: StatisticRole() for c in df.columns}, data=df, backend=BackendsEnum.pandas
    )


def test_aa_score_init_threshold() -> None:
    analyzer = AAScoreAnalyzer(alpha=0.05)
    assert analyzer.threshold == pytest.approx(1 - 0.05 * 1.2)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: pass rate is computed by iterating a Dataset (yields column names), "
    "so every feature gets pass_rate 0 and weight 1 - alpha",
)
def test_aa_score_weights_follow_pass_rate() -> None:
    analyzer = AAScoreAnalyzer(alpha=0.1)
    table = _score_table([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1] * 10])
    out = analyzer._analyze_aa_score(ExperimentData(Dataset(
        roles={"x": FeatureRole()}, data=pd.DataFrame({"x": [1.0]}), backend=BackendsEnum.pandas
    )), table)
    weights = analyzer._feature_weights
    assert weights["f0 TTest test_1"] == pytest.approx(1 - abs(0.1 - 0.1), abs=TOL)
    assert weights["f1 TTest test_1"] == pytest.approx(1 - abs(0.1 - 1.0), abs=TOL)
    stored = out.analysis_tables[analyzer.id]
    frame = stored.backend_data.data
    assert bool(frame.loc["f0 TTest test_1", "pass"]) is True
    assert bool(frame.loc["f1 TTest test_1", "pass"]) is False


def test_aa_score_skips_columns_without_parts() -> None:
    analyzer = AAScoreAnalyzer()
    df = pd.DataFrame({"pass": [1, 0]})
    table = Dataset(roles={"pass": StatisticRole()}, data=df, backend=BackendsEnum.pandas)
    analyzer._analyze_aa_score(ExperimentData(Dataset(
        roles={"x": FeatureRole()}, data=pd.DataFrame({"x": [1.0]}), backend=BackendsEnum.pandas
    )), table)
    assert analyzer._feature_weights == {}


def test_extract_splitter_id_from_column_or_row_number() -> None:
    with_col = pd.DataFrame({"splitter_id": ["AASplitter┴rs 3┴", "AASplitter┴rs 9┴"]})
    table = Dataset(roles={"splitter_id": StatisticRole()}, data=with_col, backend=BackendsEnum.pandas)
    assert AAScoreAnalyzer._extract_splitter_id(table, 1) == "AASplitter┴rs 9┴"

    without = _score_table([[1, 0]])
    assert AAScoreAnalyzer._extract_splitter_id(without, 1) == f"AASplitter{S}rs 1{S}"


def test_mean_test_score_column_defaults_to_zero() -> None:
    assert AAScoreAnalyzer._get_mean_test_score_column(_score_table([[1, 0]])) == 0.0
    df = pd.DataFrame({"mean test score": [0.1, 0.9]})
    table = Dataset(roles={"mean test score": StatisticRole()}, data=df, backend=BackendsEnum.pandas)
    assert AAScoreAnalyzer._get_mean_test_score_column(table).tolist() == [0.1, 0.9]


def test_find_best_index_defaults_to_zero_without_weights() -> None:
    assert AAScoreAnalyzer()._find_best_index(_score_table([[1, 0]]), None) == 0


def test_find_best_index_returns_zero_when_if_params_scores_present() -> None:
    analyzer = AAScoreAnalyzer()
    analyzer._feature_weights = {"f0 TTest test_1": 1.0}
    assert analyzer._find_best_index(_score_table([[1, 0]]), if_param_scores=object()) == 0


def test_find_best_index_picks_highest_weighted_score() -> None:
    analyzer = AAScoreAnalyzer(alpha=0.05)
    table = _score_table([[0, 0, 0]], pvalues=[[0.2, 0.9, 0.5]])
    analyzer._analyze_aa_score(ExperimentData(Dataset(
        roles={"x": FeatureRole()}, data=pd.DataFrame({"x": [1.0]}), backend=BackendsEnum.pandas
    )), table)
    assert analyzer._find_best_index(table, None) == 1


@pytest.mark.parametrize("cls", [AASplitter, AASplitterWithStratification])
def test_build_splitter_from_id(cls) -> None:
    splitter = cls(control_size=0.4, random_state=11)
    rebuilt = AAScoreAnalyzer().build_splitter_from_id(splitter.id)
    assert type(rebuilt) is cls
    assert rebuilt.random_state == 11
    assert rebuilt.control_size == 0.4


def test_build_splitter_from_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="not a valid splitter id"):
        AAScoreAnalyzer().build_splitter_from_id(f"Unknown{S}x{S}")


def test_aa_score_dataset_for_empty_rows_is_placeholder() -> None:
    assert AAScoreAnalyzer._build_aa_score_dataset([]) is not None


# ---------------------------------------------------------------------------
# ABAnalyzer
# ---------------------------------------------------------------------------
def _analysis_row(out: ExperimentData, analyzer: ABAnalyzer) -> pd.Series:
    return out.analysis_tables[analyzer.id].backend_data.data.iloc[0]


def test_ab_aggregates_mean_pvalue_two_groups_single_comparison() -> None:
    df, experiment_data = _grouped_experiment(groups=("a", "b"))
    analyzer = ABAnalyzer()
    row = _analysis_row(analyzer.execute(experiment_data), analyzer)
    expected_p = np.mean([_ttest_table(df, "b", t).pvalue for t in ("y1", "y2")])
    assert float(row["GroupTTest p-value b"]) == pytest.approx(expected_p, abs=TOL)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: with several targets and groups, rows are target-major but are sliced "
    "as if group-major, so per-group means mix targets and groups",
)
def test_ab_aggregates_mean_pvalue_and_pass_per_group() -> None:
    df, experiment_data = _grouped_experiment()
    analyzer = ABAnalyzer()
    out = analyzer.execute(experiment_data)
    row = _analysis_row(out, analyzer)
    for group in ("b", "c"):
        expected_p = np.mean([_ttest_table(df, group, t).pvalue for t in ("y1", "y2")])
        expected_pass = np.mean([_ttest_table(df, group, t).pvalue < 0.05 for t in ("y1", "y2")])
        assert float(row[f"GroupTTest p-value {group}"]) == pytest.approx(expected_p, abs=TOL)
        assert float(row[f"GroupTTest pass {group}"]) == pytest.approx(expected_pass, abs=TOL)


def test_ab_without_method_has_no_multitest_table() -> None:
    _, experiment_data = _grouped_experiment()
    analyzer = ABAnalyzer()
    out = analyzer.execute(experiment_data)
    assert f"{analyzer.id}MultiTest" not in out.analysis_tables


@pytest.mark.parametrize(
    "method,sm_method",
    [
        (ABNTestMethodsEnum.bonferroni, "bonferroni"),
        (ABNTestMethodsEnum.holm, "holm"),
        (ABNTestMethodsEnum.sidak, "sidak"),
        (ABNTestMethodsEnum.fdr_bh, "fdr_bh"),
        (ABNTestMethodsEnum.fdr_by, "fdr_by"),
    ],
)
def test_ab_multitest_matches_statsmodels(method, sm_method) -> None:
    df, experiment_data = _grouped_experiment()
    analyzer = ABAnalyzer(multitest_method=method)
    out = analyzer.execute(experiment_data)
    table = out.analysis_tables[f"{analyzer.id}MultiTest"].backend_data.data

    raw = np.array([float(v) for v in table["old p-value"]])
    expected = multipletests(raw, alpha=0.05, method=sm_method)
    # statsmodels is applied within the single TTest family
    np.testing.assert_allclose(
        [float(v) for v in table["new p-value"]], expected[1], atol=TOL
    )
    assert [str(v) == "True" for v in table["rejected"]] == expected[0].tolist()
    assert sorted(table["group"]) == ["b", "b", "c", "c"]
    assert sorted(table["field"]) == ["y1", "y1", "y2", "y2"]


def test_ab_multitest_skipped_for_single_comparison() -> None:
    df = _grouped_experiment(groups=("a", "b"))[0]
    ds = Dataset(
        roles={"g": TreatmentRole(), "y1": TargetRole()},
        data=df[["g", "y1"]],
        backend=BackendsEnum.pandas,
    )
    experiment = OnRoleExperiment(
        [GroupTTest(compare_by="groups", grouping_role=TreatmentRole())], role=TargetRole()
    )
    data = experiment.execute(ExperimentData(ds))
    analyzer = ABAnalyzer(multitest_method=ABNTestMethodsEnum.bonferroni)
    out = analyzer.execute(data)
    assert f"{analyzer.id}MultiTest" not in out.analysis_tables


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="Issue: MultitestQuantile defines no calc(), so ABAnalyzer with the 'quantile' "
    "method always raises NotImplementedError",
)
def test_ab_quantile_method_runs() -> None:
    _, experiment_data = _grouped_experiment()
    ABAnalyzer(
        multitest_method=ABNTestMethodsEnum.quantile, iteration_size=200, random_state=1
    ).execute(experiment_data)


@pytest.mark.parametrize(
    "analysis_id,expected",
    [(f"GroupTTest{S}hash{S}y1", f"GroupTTest{S}hash"), ("single", "single")],
)
def test_extract_id_prefix(analysis_id, expected) -> None:
    assert ABAnalyzer._extract_id_prefix(analysis_id) == expected


def test_build_row_index_iterative_mode() -> None:
    analyzer = ABAnalyzer()
    t_data = pd.DataFrame({"p-value": [0.1, 0.2, 0.3, 0.4]})
    ds = Dataset(roles={"p-value": StatisticRole()}, data=t_data, backend=BackendsEnum.pandas)
    ids = [f"GroupTTest{S}h{S}y1", f"GroupTTest{S}h{S}y2"]
    index = analyzer._build_row_index(ds, ids, num_groups=2, group_labels=["b", "c"])
    assert index == [
        f"{ids[0]}{S}b", f"{ids[0]}{S}c", f"{ids[1]}{S}b", f"{ids[1]}{S}c",
    ]


def test_build_row_index_vector_mode_parses_existing_index() -> None:
    analyzer = ABAnalyzer()
    t_data = pd.DataFrame({"p-value": [0.1, 0.2, 0.3, 0.4]}, index=["b┆y1", "c┆y1", "b┆y2", "c┆y2"])
    ds = Dataset(roles={"p-value": StatisticRole()}, data=t_data, backend=BackendsEnum.pandas)
    index = analyzer._build_row_index(
        ds, [f"StatsTTest{S}h{S}['y1', 'y2']"], num_groups=2, group_labels=["b", "c"]
    )
    assert index == [
        f"StatsTTest{S}h{S}y1{S}b", f"StatsTTest{S}h{S}y1{S}c",
        f"StatsTTest{S}h{S}y2{S}b", f"StatsTTest{S}h{S}y2{S}c",
    ]


def test_add_pvalues_only_for_pvalue_field_and_non_quantile() -> None:
    marker = pd.DataFrame({"p": [1]})

    class _Acc:
        def __init__(self):
            self.items = []

        def append(self, value):
            self.items.append(value)
            return self

    analyzer = ABAnalyzer(multitest_method=ABNTestMethodsEnum.holm)
    acc = analyzer._add_pvalues(_Acc(), marker, "p-value")
    assert acc.items == [marker]
    assert analyzer._add_pvalues(_Acc(), marker, "pass").items == []
    assert ABAnalyzer()._add_pvalues(_Acc(), marker, "p-value").items == []
    quant = ABAnalyzer(multitest_method=ABNTestMethodsEnum.quantile)
    assert quant._add_pvalues(_Acc(), marker, "p-value").items == [marker]
