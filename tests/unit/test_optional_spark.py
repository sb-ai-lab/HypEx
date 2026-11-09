"""Behavior of the base installation when the optional Spark extra is absent."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

_BLOCK_PYSPARK = """
        import builtins

        _real_import = builtins.__import__


        def _without_pyspark(name, *args, **kwargs):
            if name == "pyspark" or name.startswith("pyspark."):
                raise ModuleNotFoundError("No module named 'pyspark'", name="pyspark")
            return _real_import(name, *args, **kwargs)


        builtins.__import__ = _without_pyspark
"""


def _run_python(source: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result


def test_base_import_and_pandas_workflow_without_pyspark() -> None:
    _run_python(
        _BLOCK_PYSPARK
        + """
        import pandas as pd
        import hypex
        from hypex.dataset import Dataset, FeatureRole
        from hypex.encoders.encoders import DummyEncoder

        frame = pd.DataFrame({"color": ["red", "green", "blue", "red"]})
        dataset = Dataset(roles={"color": FeatureRole(str)}, data=frame)
        result = DummyEncoder._inner_function(dataset, target_cols=["color"])

        assert hypex.__version__
        assert dataset.backend_type.value == "pandas"
        assert sorted(result.columns) == ["color_green", "color_red"]
        assert result.raw_data["color_red"].tolist() == [1, 0, 0, 1]
        """
    )


def test_spark_entry_points_explain_missing_extra() -> None:
    _run_python(
        _BLOCK_PYSPARK
        + """
        import pandas as pd
        from hypex.dataset import Dataset, FeatureRole
        from hypex.dataset.backends import SparkDataset
        from hypex.utils import BackendsEnum, FaissIndexStorage, SparkSessionCalculator
        from hypex.utils.spark_config import SparkSettings

        dataset = Dataset(
            roles={"x": FeatureRole()}, data=pd.DataFrame({"x": [1, 2, 3]})
        )
        calls = [
            lambda: Dataset(
                roles={"x": FeatureRole()},
                data=pd.DataFrame({"x": [1]}),
                backend=BackendsEnum.spark,
            ),
            lambda: dataset.to_backend(BackendsEnum.spark),
            lambda: SparkDataset(pd.DataFrame({"x": [1]}), session=object()),
            lambda: SparkSessionCalculator.create_optimal_session(SparkSettings()),
            lambda: SparkSessionCalculator().optimize_config(),
            lambda: FaissIndexStorage(object()),
        ]

        for call in calls:
            try:
                call()
            except ImportError as exc:
                assert str(exc) == (
                    "Spark support requires PySpark. "
                    'Install it with pip install "hypex[spark]".'
                )
            else:
                raise AssertionError("Spark call succeeded without PySpark")
        """
    )


def test_unrelated_pyspark_import_failure_is_not_hidden() -> None:
    _run_python(
        """
        import builtins

        real_import = builtins.__import__

        def broken_pyspark(name, *args, **kwargs):
            if name == "pyspark" or name.startswith("pyspark."):
                raise ModuleNotFoundError(
                    "No module named 'broken_dependency'", name="broken_dependency"
                )
            return real_import(name, *args, **kwargs)

        builtins.__import__ = broken_pyspark

        try:
            import hypex
        except ModuleNotFoundError as exc:
            assert exc.name == "broken_dependency"
        else:
            raise AssertionError("A broken PySpark installation was silently ignored")
        """
    )


def test_no_spark_tox_environment_really_omits_pyspark() -> None:
    if os.environ.get("HYPEX_REQUIRE_NO_PYSPARK") != "1":
        pytest.skip("Only required in the dedicated no-Spark environment")
    assert importlib.util.find_spec("pyspark") is None
