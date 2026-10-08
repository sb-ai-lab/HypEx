"""Tests for MatchingReporter, MatchingQualityReporter, HomogeneityReporter, CupacReporter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.analyzers.matching import MatchingAnalyzer
from hypex.comparators import GroupDifference, GroupSizes, GroupTTest
from hypex.dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    SmallDataset,
    StatisticRole,
    TargetRole,
    TreatmentRole,
)
from hypex.experiments import OnRoleExperiment
from hypex.ml.cupac import CUPACExecutor
from hypex.operators import MatchingMetrics
from hypex.reporters import DictReporter
from hypex.reporters.cupac import CupacReporter
from hypex.reporters.homo import HomogeneityReporter
from hypex.reporters.matching import MatchingQualityReporter, MatchingReporter
from hypex.utils import ID_SPLIT_SYMBOL as S
from hypex.utils import MATCHING_INDEXES_SPLITTER_SYMBOL as M
from hypex.utils import BackendsEnum, ExperimentDataEnum


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
@pytest.fixture
def matched():
    rng = np.random.RandomState(0)
    n = 20
    x = np.r_[rng.uniform(0, 10, n), rng.uniform(2, 12, n)]
    treat = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    y = 2.0 * x + 5.0 * treat + rng.normal(0, 1, 2 * n)
    nn = np.empty(2 * n, dtype=int)
    for i in range(2 * n):
        pool = np.where(treat != treat[i])[0]
        nn[i] = pool[np.argmin(np.abs(x[pool] - x[i]))]
    ds = Dataset(
        roles={
            "treat": TreatmentRole(),
            "x": FeatureRole(),
            "y": TargetRole(),
            "nn": AdditionalMatchingRole(),
        },
        data=pd.DataFrame({"treat": treat, "x": x, "y": y, "nn": nn}),
        backend=BackendsEnum.pandas,
    )
    data = MatchingMetrics(grouping_role=TreatmentRole()).execute(ExperimentData(ds))
    return MatchingAnalyzer().execute(data), nn


def test_matching_analyzer_stores_transposed_metric_table(matched) -> None:
    data, _ = matched
    (table,) = data.analysis_tables.values()
    frame = table.raw_data
    assert list(frame.index) == ["ATT", "ATC", "ATE"]
    assert list(frame.columns) == [
        "Effect Size",
        "Standard Error",
        "P-value",
        "CI Lower",
        "CI Upper",
    ]
    assert data.get_one_id(MatchingAnalyzer, ExperimentDataEnum.analysis_tables)


def test_matching_reporter_extracts_flat_metric_keys(matched) -> None:
    data, _ = matched
    result = MatchingReporter(output_format="dict")._extract_from_analyser(data)
    (table,) = data.analysis_tables.values()
    assert len(result) == 15
    assert result[f"Effect Size{S}ATT"] == table.raw_data.loc["ATT", "Effect Size"]
    assert result[f"CI Upper{S}ATE"] == table.raw_data.loc["ATE", "CI Upper"]


def test_matching_reporter_extract_indexes_joined_string(matched) -> None:
    data, nn = matched
    data.set_value(
        ExperimentDataEnum.additional_fields,
        f"FaissNearestNeighbors{S}h{S}{S}0",
        list(nn),
        role=AdditionalMatchingRole(),
    )
    result = MatchingReporter(output_format="dict")._extract_indexes(data)
    assert result == {f"indexes{S}0": M.join(str(int(i)) for i in nn)}


def test_matching_reporter_report_includes_indexes_for_analyzer(matched) -> None:
    data, nn = matched
    data.set_value(
        ExperimentDataEnum.additional_fields,
        f"FaissNearestNeighbors{S}h{S}{S}0",
        list(nn),
        role=AdditionalMatchingRole(),
    )
    result = MatchingReporter(output_format="dict").report(data)
    assert f"indexes{S}0" in result
    assert f"Effect Size{S}ATT" in result


def test_matching_reporter_custom_searching_class_skips_indexes(matched) -> None:
    class FakeAnalyzer(MatchingAnalyzer):
        pass

    data, nn = matched
    data.set_value(
        ExperimentDataEnum.additional_fields,
        f"FaissNearestNeighbors{S}h{S}{S}0",
        list(nn),
        role=AdditionalMatchingRole(),
    )
    (table,) = data.analysis_tables.values()
    data.set_value(ExperimentDataEnum.analysis_tables, f"FakeAnalyzer{S}{S}", table)
    result = MatchingReporter(FakeAnalyzer, output_format="dict").report(data)
    assert f"Effect Size{S}ATT" in result
    assert not any(k.startswith("indexes") for k in result)


def test_matching_reporter_dataset_format_is_dataset(matched) -> None:
    data, _ = matched
    data.set_value(
        ExperimentDataEnum.additional_fields,
        f"FaissNearestNeighbors{S}h{S}{S}0",
        [0] * 40,
        role=AdditionalMatchingRole(),
    )
    result = MatchingReporter().report(data)
    assert not isinstance(result, dict)


def test_matching_quality_reporter_with_no_tests_is_empty(matched) -> None:
    data, _ = matched
    assert (
        MatchingQualityReporter(DictReporter(), output_format="dict").report(data) == {}
    )


def test_matching_quality_reporter_lists_group_tests() -> None:
    names = {t.__name__ for t in MatchingQualityReporter.tests}
    assert {"GroupTTest", "GroupKSTest", "GroupChi2Test", "StatsUTest"} <= names


# ---------------------------------------------------------------------------
# Homogeneity
# ---------------------------------------------------------------------------
@pytest.fixture
def executed() -> ExperimentData:
    rng = np.random.RandomState(0)
    n = 60
    df = pd.DataFrame(
        {
            "g": ["a"] * n + ["b"] * n,
            "y1": np.r_[rng.normal(0, 1, n), rng.normal(0.5, 1, n)],
        }
    )
    ds = Dataset(
        roles={"g": TreatmentRole(), "y1": TargetRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )
    experiment = OnRoleExperiment(
        [
            GroupSizes(grouping_role=TreatmentRole()),
            GroupDifference(grouping_role=TreatmentRole()),
            GroupTTest(compare_by="groups", grouping_role=TreatmentRole()),
        ],
        role=TargetRole(),
    )
    return experiment.execute(ExperimentData(ds))


def test_homogeneity_reporter_collects_sizes_differences_and_tests(executed) -> None:
    result = HomogeneityReporter(DictReporter(), output_format="dict").report(executed)
    keys = list(result)
    assert any("GroupSizes" in k for k in keys)
    assert any("GroupDifference" in k for k in keys)
    assert any("GroupTTest" in k and "p-value" in k for k in keys)
    assert any("GroupTTest" in k and "pass" in k for k in keys)


def test_homogeneity_reporter_dataset_format(executed) -> None:
    result = HomogeneityReporter(DictReporter(), output_format="dataset").report(
        executed
    )
    assert not result.is_empty()


# ---------------------------------------------------------------------------
# CUPAC
# ---------------------------------------------------------------------------
def _cupac_data() -> ExperimentData:
    ds = Dataset(
        roles={"y": TargetRole()},
        data=pd.DataFrame({"y": [1.0, 2.0, 3.0]}),
        backend=BackendsEnum.pandas,
    )
    return ExperimentData(ds)


def _report_ds(model, cv, real) -> SmallDataset:
    return SmallDataset.from_dict(
        [
            {
                "cupac_best_model": model,
                "cupac_variance_reduction_cv": cv,
                "cupac_variance_reduction_real": real,
            }
        ],
        roles={
            "cupac_best_model": StatisticRole(),
            "cupac_variance_reduction_cv": StatisticRole(float),
            "cupac_variance_reduction_real": StatisticRole(float),
        },
    )


def _add_cupac_target(data, target, model, cv, real, importances) -> None:
    base = f"CUPACExecutor{S}abc{S}{target}"
    data.set_value(
        ExperimentDataEnum.analysis_tables, base, _report_ds(model, cv, real)
    )
    if importances:
        data.set_value(
            ExperimentDataEnum.analysis_tables,
            f"{base}{S}importances",
            SmallDataset.from_dict(
                [importances], roles={k: StatisticRole(float) for k in importances}
            ),
        )


def test_cupac_reporter_without_cupac_results_returns_none_tables() -> None:
    assert CupacReporter().report(_cupac_data()) == {
        "variance_reductions": None,
        "feature_importances": None,
    }


def test_cupac_reporter_builds_variance_reductions_and_importances() -> None:
    data = _cupac_data()
    _add_cupac_target(data, "y", "ridge", 38.2, 41.5, {"x1": 0.85, "x2": 0.15})
    _add_cupac_target(data, "z", "lasso", 10.0, 12.0, {"x1": 1.0})
    report = CupacReporter().report(data)

    vr = report["variance_reductions"].raw_data.reset_index(drop=True)
    assert vr.to_dict("records") == [
        {
            "target": "y",
            "best_model": "ridge",
            "variance_reduction_cv": 38.2,
            "variance_reduction_real": 41.5,
        },
        {
            "target": "z",
            "best_model": "lasso",
            "variance_reduction_cv": 10.0,
            "variance_reduction_real": 12.0,
        },
    ]
    fi = report["feature_importances"].raw_data.reset_index(drop=True)
    assert fi.to_dict("records") == [
        {"target": "y", "feature": "x1", "importance": 0.85, "model": "ridge"},
        {"target": "y", "feature": "x2", "importance": 0.15, "model": "ridge"},
        {"target": "z", "feature": "x1", "importance": 1.0, "model": "lasso"},
    ]


def test_cupac_reporter_without_importances_gives_none_feature_table() -> None:
    data = _cupac_data()
    _add_cupac_target(data, "y", "ridge", 1.0, 2.0, None)
    report = CupacReporter().report(data)
    assert report["variance_reductions"] is not None
    assert report["feature_importances"] is None


def test_cupac_reporter_importances_without_main_report_have_no_model() -> None:
    data = _cupac_data()
    data.set_value(
        ExperimentDataEnum.analysis_tables,
        f"CUPACExecutor{S}abc{S}y{S}importances",
        SmallDataset.from_dict([{"x1": 0.5}], roles={"x1": StatisticRole(float)}),
    )
    report = CupacReporter().report(data)
    assert report["variance_reductions"] is None
    fi = report["feature_importances"].raw_data
    assert fi["model"].isna().all() or fi["model"].tolist() == [None]
    assert fi["target"].tolist() == ["y"]


def test_cupac_reporter_skips_empty_tables() -> None:
    data = _cupac_data()
    base = f"CUPACExecutor{S}abc{S}y"
    data.analysis_tables[base] = None
    data.analysis_tables[f"{base}{S}importances"] = SmallDataset(
        data=pd.DataFrame(), roles={}
    )
    report = CupacReporter().report(data)
    assert report == {"variance_reductions": None, "feature_importances": None}


def test_cupac_reporter_ignores_other_executors() -> None:
    data = _cupac_data()
    data.set_value(
        ExperimentDataEnum.analysis_tables,
        f"OtherExecutor{S}abc{S}y",
        _report_ds("ridge", 1.0, 2.0),
    )
    assert CUPACExecutor.__name__ == "CUPACExecutor"
    assert CupacReporter().report(data)["variance_reductions"] is None
