from __future__ import annotations

import pickle
from unittest.mock import PropertyMock, patch

import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole, InfoRole, TargetRole
from hypex.ml.models import LinearModel
from hypex.utils import BackendsEnum
from hypex.utils.registry import BackendFactory


def _data(factory=Dataset) -> Dataset:
    return factory(
        data=pd.DataFrame(
            {
                "row_id": [20, 10, 20, 30, 40],
                "first": [1, 2, 3, 4, 5],
                "second": [3, 1, 2, 0, 4],
                "outcome": [0, 8, 8, 16, 8],
            }
        ),
        roles={
            "row_id": InfoRole(),
            "first": FeatureRole(),
            "second": FeatureRole(),
            "outcome": TargetRole(),
        },
    )


def _predictions(model, data):
    result = model.predict(data)
    assert result.backend_type is data.backend_type
    frame = result.to_backend(BackendsEnum.pandas).raw_data
    return frame.sort_values("outcome_predicted").sort_index(kind="stable")


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({}, id="ols"),
        pytest.param({"reg_param": 0.2, "reg_type": "l1"}, id="lasso"),
        pytest.param({"reg_param": 0.2, "reg_type": "l2"}, id="ridge"),
        pytest.param({"reg_param": 0.2, "elastic_net_param": 0.4}, id="elastic-net"),
    ],
)
def test_linear_model_native_round_trip(make_dataset, tmp_path, params):
    training = _data(make_dataset).set_index("row_id")
    scoring = training[["second", "first"]]
    model = LinearModel(**params)
    assert model.fit(training) is model
    assert isinstance(model.model, LinearModel)
    assert model.fitted
    reference = _predictions(model, scoring)

    assert reference.index.name == "row_id"
    assert reference.index.to_list() == [10, 20, 20, 30, 40]
    assert reference["outcome_predicted"].notna().all()
    if not params:
        assert reference["outcome_predicted"].to_list() == pytest.approx(
            [8, 0, 8, 16, 8], abs=1e-10
        )

    path = tmp_path / "linear.pkl"
    model.save(str(path))
    assert path.is_file()
    assert list(tmp_path.iterdir()) == [path]
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    assert isinstance(artifact["model"], bytes) and artifact["model"]
    assert artifact["state"]["backend"] is type(model.model)
    assert list(artifact["state"]["features"]) == ["first", "second"]
    with patch.object(
        BackendFactory,
        "registry",
        new_callable=PropertyMock,
        side_effect=AssertionError("load accessed the backend registry"),
    ):
        loaded = LinearModel.load(str(path))

    assert type(loaded) is LinearModel
    assert loaded.fitted
    assert type(loaded.model) is type(model.model)
    assert loaded.model_params == model.model_params
    pd.testing.assert_frame_equal(_predictions(loaded, scoring), reference)
    with pytest.raises(ValueError, match="first"):
        loaded.predict(scoring.rename({"first": "different_feature"}))


def test_save_keeps_existing_artifact(tmp_path):
    model = LinearModel().fit(_data())
    path = tmp_path / "linear.pkl"
    model.save(str(path))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        model.save(str(path))
    assert path.read_bytes() == original


def test_spark_assembler_uses_original_feature_names(spark_session):
    from pyspark.ml.feature import VectorAssembler

    data = (
        _data()
        .to_backend(BackendsEnum.spark, session=spark_session)
        .set_index("row_id")
    )
    with patch(
        "hypex.ml.models.backends.linear.spark.VectorAssembler", wraps=VectorAssembler
    ) as assembler:
        model = LinearModel().fit(data)
        assert assembler.call_args.kwargs["inputCols"] == ["first", "second"]
        assert model.model._model.getLabelCol() == "outcome"
        predictions = model.predict(data[["second", "first"]])
        assert assembler.call_args.kwargs["inputCols"] == ["first", "second"]

    assert predictions.backend_type is BackendsEnum.spark
    assert predictions.columns == ["outcome_predicted"]
    with pytest.raises(ValueError, match="fitted model's backend"):
        model.predict(_data()[["first", "second"]])
