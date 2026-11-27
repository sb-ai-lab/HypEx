from __future__ import annotations

import json

import pandas as pd
import pytest

from hypex.dataset import AdditionalFeatureRole, ExperimentData, FeatureRole, TargetRole
from hypex.experiments import MLExperiment
from hypex.ml import LinearModelExecutor
from hypex.transformers import DummyEncoder, MLScaler
from hypex.utils import BackendsEnum


def test_linear_pipeline_round_trip(make_dataset, tmp_path):
    def dataset(x, categories, *, train=False):
        offsets = {"a": 0, "b": 6, "c": -4}
        expected = [
            3 * value + offsets[group] + 1 for value, group in zip(x, categories)
        ]
        columns = {"x": x, "category": categories}
        roles = {"x": FeatureRole(float), "category": FeatureRole(str)}
        if train:
            columns["outcome"] = expected
            roles["outcome"] = TargetRole(float)
        return make_dataset(pd.DataFrame(columns), roles), expected

    training, expected_train = dataset(
        [1.0, 3.0, 5.0, 9.0, 2.0, 7.0], ["a", "b", "c", "a", "b", "c"], train=True
    )
    scoring, expected_score = dataset([11.0, 16.0, 12.0, 18.0], ["c", "a", "b", "a"])
    encoder = DummyEncoder(target_roles=FeatureRole(str))
    scaler = MLScaler()
    regression = LinearModelExecutor(model_params={"fit_intercept": True})
    experiment = MLExperiment(
        [encoder, scaler, regression],
        key="linear",
        save_models=False,
    )

    result = experiment.execute(ExperimentData(training))
    frame = result.ds.to_backend(BackendsEnum.pandas).data.sort_index()
    assert frame["x"].mean() == pytest.approx(0, abs=1e-10)
    dummy_columns = result.ds.search_columns(AdditionalFeatureRole())
    assert len(dummy_columns) == 2
    assert all(result.ds.roles[col].data_type is float for col in dummy_columns)
    assert frame["outcome_predicted"].to_list() == pytest.approx(expected_train)
    assert "outcome_predicted" in result.additional_fields.columns

    fitted_stages = MLExperiment([scaler, regression], key="linear")
    fitted_stages.save(path=str(tmp_path))
    artifact = next(tmp_path.glob("linear_*"))
    snapshot = json.loads((artifact / "metadata.json").read_text())
    assert [stage["qualname"] for stage in snapshot.values()] == [
        "MLScaler",
        "LinearModelExecutor",
    ]
    assert all(stage["has_model"] for stage in snapshot.values())
    assert {path.suffix for path in (artifact / "models").iterdir()} == {
        ".pkl",
        ".json",
    }

    restored = MLExperiment(
        [], mode="predict", key="linear", artifact_path=str(tmp_path)
    )
    restored.load()
    assert [type(stage) for stage in restored.executors] == [
        MLScaler,
        LinearModelExecutor,
    ]
    assert restored.executors[1].model_params == {"fit_intercept": True}
    encoded = DummyEncoder(target_roles=FeatureRole(str), key="linear").execute(
        ExperimentData(scoring)
    )
    predicted = restored.execute(encoded)
    values = predicted.ds.to_backend(BackendsEnum.pandas).data.sort_index()
    assert values["outcome_predicted"].to_list() == pytest.approx(expected_score)
    assert values["x"].mean() > 0
