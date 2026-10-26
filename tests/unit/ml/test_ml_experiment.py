"""Unit tests for hypex.experiments.ml.MLExperiment."""
from __future__ import annotations

import pandas as pd
import pytest

from hypex.config import MLConfig
from hypex.dataset import Dataset, ExperimentData, InfoRole, MLExperimentData
from hypex.executor import MLExecutor
from hypex.experiments.ml import MLExperiment
from hypex.ml.artifact import MLArtifact
from hypex.utils.enums import MLModeEnum


# ── Test doubles ────────────────────────────────────────────────────────
class _StubMLExecutor(MLExecutor):
    def __init__(self, key: str = "", value: float = 0.5):
        super().__init__(key=key)
        self.value = value

    def fit(self, data):
        self._model = {"value": self.value}
        self.is_fitted = True
        return data

    def predict(self, data):
        return data


# ── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def sample_dataset():
    return Dataset(
        data=pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}),
        roles={"x": InfoRole(), "y": InfoRole()},
    )


@pytest.fixture
def ml_data(sample_dataset):
    return MLExperimentData(sample_dataset)


# ── __init__ / mode coercion ────────────────────────────────────────────
class TestInit:
    def test_mode_string_coerced_to_enum(self):
        exp = MLExperiment(executors=[], mode="fit", key="k")
        assert exp.mode is MLModeEnum.FIT

    def test_mode_enum_preserved(self):
        exp = MLExperiment(executors=[], mode=MLModeEnum.FIT_PREDICT, key="k")
        assert exp.mode is MLModeEnum.FIT_PREDICT

    def test_invalid_mode_string_raises(self):
        with pytest.raises(ValueError):
            MLExperiment(executors=[], mode="predict_all", key="k")

    def test_predict_without_key_raises(self):
        with pytest.raises(ValueError, match="key is required for PREDICT"):
            MLExperiment(executors=[], mode=MLModeEnum.PREDICT)

    def test_predict_with_key_ok(self):
        MLExperiment(executors=[], mode=MLModeEnum.PREDICT, key="exp1")

    def test_fit_and_fit_predict_do_not_require_key(self):
        MLExperiment(executors=[], mode=MLModeEnum.FIT)
        MLExperiment(executors=[], mode=MLModeEnum.FIT_PREDICT)

    def test_artifact_path_defaults_to_config(self):
        exp = MLExperiment(executors=[], key="k")
        assert exp.artifact_path == MLConfig.DEFAULT_ARTIFACT_PATH

    def test_artifact_path_override(self, tmp_path):
        exp = MLExperiment(executors=[], artifact_path=str(tmp_path), key="k")
        assert exp.artifact_path == str(tmp_path)

    def test_save_models_flag_stored(self):
        exp = MLExperiment(executors=[], save_models=False, key="k")
        assert exp.save_models is False


# ── _validate_key ───────────────────────────────────────────────────────
class TestValidateKey:
    def test_matrix(self):
        MLExperiment._validate_key(MLModeEnum.FIT, "")
        MLExperiment._validate_key(MLModeEnum.FIT_PREDICT, "")
        MLExperiment._validate_key(MLModeEnum.PREDICT, "k")
        with pytest.raises(ValueError):
            MLExperiment._validate_key(MLModeEnum.PREDICT, "")


# ── execute ─────────────────────────────────────────────────────────────
class TestExecute:
    def test_execute_runs_pipeline(self, ml_data):
        ex = _StubMLExecutor(key="e1", value=0.7)
        exp = MLExperiment(
            executors=[ex],
            mode=MLModeEnum.FIT,
            save_models=False,
            key="exp1",
        )
        exp.execute(ml_data)
        assert ex.is_fitted is True

    def test_execute_fit_predict_saves_when_enabled(self, ml_data, tmp_path):
        exp = MLExperiment(
            executors=[_StubMLExecutor(key="e1")],
            mode=MLModeEnum.FIT_PREDICT,
            save_models=True,
            key="exp1",
            artifact_path=str(tmp_path),
        )
        exp.execute(ml_data)
        assert any(p.name.startswith("exp1_") for p in tmp_path.iterdir())

    def test_execute_fit_does_not_save_when_disabled(self, ml_data, tmp_path):
        exp = MLExperiment(
            executors=[_StubMLExecutor()],
            mode=MLModeEnum.FIT,
            save_models=False,
            key="exp1",
            artifact_path=str(tmp_path),
        )
        exp.execute(ml_data)
        assert not any(p.name.startswith("exp1_") for p in tmp_path.iterdir())

    @pytest.mark.xfail(
        strict=True, reason="requires bugs A and D fixed (save then auto-load)"
    )
    def test_predict_mode_loads_artifact_before_run(self, ml_data, tmp_path):
        fit_exp = MLExperiment(
            executors=[_StubMLExecutor(key="e1", value=0.4)],
            mode=MLModeEnum.FIT,
            save_models=True,
            key="exp1",
            artifact_path=str(tmp_path),
        )
        fit_exp.execute(ml_data)

        predict_exp = MLExperiment(
            executors=[],
            mode=MLModeEnum.PREDICT,
            key="exp1",
            artifact_path=str(tmp_path),
        )
        predict_exp.execute(ml_data)
        assert len(predict_exp.executors) == 1

    @pytest.mark.xfail(strict=True, reason="bug F: _ensure_ml_data is dead code")
    def test_execute_requires_ml_experiment_data(self, sample_dataset):
        exp = MLExperiment(
            executors=[_StubMLExecutor()],
            mode=MLModeEnum.FIT,
            save_models=False,
            key="exp1",
        )
        plain = ExperimentData(sample_dataset)
        exp.execute(plain)  # should be wrapped by _ensure_ml_data


# ── save ────────────────────────────────────────────────────────────────
class TestSave:
    def test_save_returns_artifact(self, tmp_path):
        exp = MLExperiment(
            executors=[_StubMLExecutor(key="e1")],
            key="exp1",
            artifact_path=str(tmp_path),
        )
        art = exp.save()
        assert isinstance(art, MLArtifact)
        assert art.experiment_id == "exp1"

    def test_save_uses_explicit_path_override(self, tmp_path):
        exp = MLExperiment(
            executors=[_StubMLExecutor()],
            key="exp1",
            artifact_path="/should/not/matter",
        )
        alt = tmp_path / "alt"
        alt.mkdir()
        art = exp.save(path=str(alt))
        assert art.base_dir == str(alt)


# ── load ────────────────────────────────────────────────────────────────
class TestLoad:
    def _make_artifact(self, tmp_path, n_executors=1):
        exp = MLExperiment(
            executors=[_StubMLExecutor(key=f"e{i}") for i in range(n_executors)],
            key="exp1",
            artifact_path=str(tmp_path),
        )
        exp.save()

    def test_load_replaces_executors(self, tmp_path):
        self._make_artifact(tmp_path, n_executors=1)
        fresh = MLExperiment(executors=[], key="exp1", artifact_path=str(tmp_path))
        fresh.load()
        assert len(fresh.executors) == 1
        assert isinstance(fresh.executors[0], _StubMLExecutor)

    def test_load_with_mismatching_executors_raises(self, tmp_path):
        self._make_artifact(tmp_path, n_executors=2)
        conflicting = MLExperiment(
            executors=[_StubMLExecutor()], key="exp1", artifact_path=str(tmp_path)
        )
        with pytest.raises(ValueError, match="Incompatible artifact"):
            conflicting.load()

    def test_load_missing_artifact_raises(self, tmp_path):
        exp = MLExperiment(executors=[], key="ghost", artifact_path=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            exp.load()

    def test_load_is_idempotent(self, tmp_path):
        self._make_artifact(tmp_path, n_executors=1)
        exp = MLExperiment(executors=[], key="exp1", artifact_path=str(tmp_path))
        exp.load()
        first = list(exp.executors)
        exp.load()
        assert exp.executors == first


# ── _ensure_ml_data ─────────────────────────────────────────────────────
class TestEnsureMlData:
    def test_passthrough_ml_experiment_data(self, ml_data):
        exp = MLExperiment(executors=[], key="k")
        assert exp._ensure_ml_data(ml_data) is ml_data

    def test_wraps_plain_experiment_data_preserving_state(self, sample_dataset):
        exp = MLExperiment(executors=[], key="k")
        plain = ExperimentData(sample_dataset)
        plain.variables = {"v": {"x": 1}}
        plain.groups = {"g": {}}
        plain.analysis_tables = {"a": None}
        plain.id_name_mapping = {"id": "name"}

        wrapped = exp._ensure_ml_data(plain)

        assert isinstance(wrapped, MLExperimentData)
        assert wrapped.variables == {"v": {"x": 1}}
        assert wrapped.groups == {"g": {}}
        assert wrapped.analysis_tables == {"a": None}
        assert wrapped.id_name_mapping == {"id": "name"}
        assert wrapped.ds is plain.ds
