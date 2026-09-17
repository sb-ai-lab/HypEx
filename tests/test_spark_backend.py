import pandas as pd
import pytest

from hypex.dataset import Dataset
from hypex.dataset.dataset import SmallDataset
from hypex.dataset.backends.spark_backend import SparkDataset
from hypex.dataset.roles import DefaultRole
from hypex.utils import ExperimentDataEnum


def _spark_backend(data):
    return SparkDataset(pd.DataFrame(data))


def test_small_dataset_from_dict_returns_instance():
    ds = SmallDataset.from_dict(
        {"a": [1, 2]}, roles={"a": DefaultRole()}
    )
    assert isinstance(ds, SmallDataset)
    assert list(ds.columns) == ["a"]


def test_experiment_analysis_tables_accepts_dict():
    ds = Dataset(data=pd.DataFrame({"x": [1]}), roles={"x": DefaultRole()})
    from hypex.dataset import ExperimentData
    exp = ExperimentData(ds)

    exp.set_value(
        space=ExperimentDataEnum.analysis_tables,
        executor_id="test_executor",
        value={"x": [1, 2]},
        role={"x": DefaultRole()},
    )

    assert "test_executor" in exp.analysis_tables
    assert isinstance(exp.analysis_tables["test_executor"], SmallDataset)


@pytest.mark.parametrize("how", ["any", "all"])
def test_spark_dropna_rows(how):
    backend = _spark_backend({"a": [1, None], "b": [2, None]})
    result = backend.dropna(how=how, axis=0)
    if how == "any":
        assert result.count() == 1
    else:
        assert result.count() == 1


def test_spark_sort_values_and_reindex():
    backend = _spark_backend({"a": [2, 1], "b": ["x", "y"]})
    sorted_df = backend.sort_values(by="a", ascending=True)
    assert [r["a"] for r in sorted_df.select("a").collect()] == [1, 2]

    reindexed = backend.reindex(["b", "missing"], fill_value="na")
    assert reindexed.columns == ["b", "missing"]
