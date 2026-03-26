import pytest
from hypex.dataset import Dataset

class TestDatasetAccess:
    def test_getitem_column(self, spark_dataset):
        result = spark_dataset["value"]
        assert isinstance(result, Dataset)
        assert "value" in result.columns

    def test_getitem_row_int(self, spark_dataset):
        row = spark_dataset[0]
        assert isinstance(row, Dataset)
        assert row.shape[0] == 5
        assert row.shape[1] == 1

    def test_getitem_list_columns(self, spark_dataset):
        subset = spark_dataset[["id", "score"]]
        assert set(subset.columns) == {"id", "score"}

    def test_get_values_scalar(self, spark_dataset):
        val = spark_dataset.get_values(row=0, column="id")
        assert val == 1

    def test_iget_values_scalar(self, spark_dataset):
        val = spark_dataset.iget_values(row=0, column=0)
        assert val == 1

    def test_filter_items(self, spark_dataset):
        filtered = spark_dataset.filter(items=["id", "name", "value"])
        assert set(filtered.columns) == {"id", "name", "value"}

    def test_select_columns(self, spark_dataset):
        selected = spark_dataset.select(["score", "value"])
        assert set(selected.columns) == {"score", "value"}

    def test_iselect_columns(self, spark_dataset):
        selected = spark_dataset.iselect([0, 4])
        assert set(selected.columns) == {"id", "score"}