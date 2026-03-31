import pytest
import numpy as np

from hypex.dataset import Dataset


class TestDatasetOperators:
    def test_comparison_ops(self, spark_dataset):
        eq_result = spark_dataset["score"] == 100
        gt_result = spark_dataset["value"] > 20
        assert isinstance(eq_result, Dataset)
        assert isinstance(gt_result, Dataset)

    def test_unary_ops(self, spark_dataset):
        neg = -spark_dataset[["score"]]
        abs_val = abs(spark_dataset[["value"]])
        assert isinstance(neg, Dataset)
        assert isinstance(abs_val, Dataset)

    def test_arithmetic_ops(self, spark_dataset):
        add = spark_dataset[["score"]] + 10
        mul = spark_dataset[["score"]] * 2
        assert isinstance(add, Dataset)
        assert isinstance(mul, Dataset)

    def test_right_ops(self, spark_dataset):
        radd = 10 + spark_dataset[["score"]]
        assert isinstance(radd, Dataset)

    def test_dot_product_single(self, spark_dataset):
        single_col = spark_dataset[["score"]]
        vec1 = np.array([2])
        dot_result1 = single_col.dot(vec1)
        assert dot_result1 is not None

    def test_dot_product_multi(self, spark_dataset):
        multi_col = spark_dataset[["id", "score"]]
        vec2 = np.array([1, 2])
        dot_result2 = multi_col.dot(vec2)
        assert dot_result2 is not None