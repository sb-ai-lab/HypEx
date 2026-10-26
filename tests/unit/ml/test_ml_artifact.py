"""Unit tests for hypex.ml.artifact.MLArtifact."""
from __future__ import annotations

import json
import os
from typing import ClassVar

import pytest

from hypex.config import MLConfig
from hypex.executor import MLExecutor
from hypex.ml.artifact import MLArtifact


# ── Test doubles ────────────────────────────────────────────────────────
class _StubMLExecutor(MLExecutor):
    """MLExecutor without external deps; picklable model payload."""

    def __init__(self, key: str = "", value: float = 0.5):
        super().__init__(key=key)
        self.value = value

    def fit(self, data):
        self._model = {"value": self.value}
        self.is_fitted = True
        return data

    def predict(self, data):
        return data


class _OtherStubMLExecutor(MLExecutor):
    def __init__(self, key: str = "", beta: float = 0.1):
        super().__init__(key=key)
        self.beta = beta

    def fit(self, data):
        return data

    def predict(self, data):
        return data


class _ExperimentLike:
    """Minimal stand-in for MLExperiment — only .executors is used."""

    def __init__(self, executors):
        self.executors = executors


# ── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def stub_executor():
    return _StubMLExecutor(key="e1", value=0.7)


@pytest.fixture
def artifact(tmp_path):
    return MLArtifact(
        experiment_id="exp1",
        base_dir=str(tmp_path),
        created_at="2026-01-01T00:00:00",
    )


# ── Construction ────────────────────────────────────────────────────────
class TestConstruction:
    def test_paths_derived_from_id_and_timestamp(self, tmp_path):
        art = MLArtifact("exp1", str(tmp_path), created_at="2026-01-01T00:00:00")
        assert art.experiment_dir == str(tmp_path / "exp1_2026-01-01T00-00-00")
        assert art.metadata_file == os.path.join(art.experiment_dir, "metadata.json")
        assert art.models_dir == os.path.join(art.experiment_dir, "models")

    def test_base_dir_defaults_to_config(self):
        art = MLArtifact("exp1", base_dir=None, created_at="2026-01-01T00:00:00")
        assert art.base_dir == MLConfig.DEFAULT_ARTIFACT_PATH

    def test_created_at_defaults_to_now(self):
        from datetime import datetime

        art = MLArtifact("exp1", base_dir=".")
        datetime.fromisoformat(art.created_at)  # no exception == sane ISO

    def test_created_at_sanitised_for_filesystem(self):
        art = MLArtifact("exp1", ".", created_at="2026-01-01T12:34:56.789")
        name = os.path.basename(art.experiment_dir)
        assert ":" not in name
        assert name == "exp1_2026-01-01T12-34-56-789"

    def test_fresh_state_is_empty(self):
        art = MLArtifact("exp1", ".")
        assert art._executors == []
        assert art.pipeline_config in ([], None)


# ── _build_pipeline_config ──────────────────────────────────────────────
class TestBuildPipelineConfig:
    def test_structure(self, stub_executor):
        cfg = MLArtifact._build_pipeline_config([stub_executor])
        assert len(cfg) == 1
        assert set(cfg[0]) == {"class", "module", "id", "key", "params"}
        assert cfg[0]["class"] == "_StubMLExecutor"
        assert cfg[0]["module"] == __name__
        assert cfg[0]["id"] == stub_executor.id
        assert cfg[0]["key"] == stub_executor.key

    def test_preserves_order(self):
        e1 = _StubMLExecutor(key="a")
        e2 = _OtherStubMLExecutor(key="b")
        cfg = MLArtifact._build_pipeline_config([e1, e2])
        assert [c["class"] for c in cfg] == ["_StubMLExecutor", "_OtherStubMLExecutor"]

    def test_empty_pipeline(self):
        assert MLArtifact._build_pipeline_config([]) == []


# ── _validate_config ────────────────────────────────────────────────────
class TestValidateConfig:
    _A: ClassVar[dict] = {"class": "A", "module": "m", "id": "1", "key": "", "params": {}}
    _B: ClassVar[dict] = {"class": "B", "module": "m", "id": "2", "key": "", "params": {}}

    def test_identical_is_compatible(self):
        ok, msg = MLArtifact._validate_config([self._A], [self._A])
        assert ok is True
        assert msg == "Compatible"

    def test_length_mismatch(self):
        ok, msg = MLArtifact._validate_config([self._A], [self._A, self._B])
        assert ok is False
        assert "Incorrect number of executors" in msg

    def test_class_mismatch(self):
        ok, msg = MLArtifact._validate_config([self._A], [self._B])
        assert ok is False
        assert "A" in msg and "B" in msg

    def test_module_and_params_not_compared(self):
        a = {**self._A, "module": "m1", "params": {"x": 1}}
        b = {**self._A, "module": "m2", "params": {"x": 2}}
        ok, _ = MLArtifact._validate_config([a], [b])
        assert ok is True


# ── _get_path ───────────────────────────────────────────────────────────
class TestGetPath:
    def test_missing_artifact_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MLArtifact._get_path("exp", str(tmp_path / "nope"), None)

    def test_no_matching_experiment(self, tmp_path):
        (tmp_path / "other_1").mkdir()
        with pytest.raises(FileNotFoundError):
            MLArtifact._get_path("exp", str(tmp_path), None)

    def test_returns_lexicographically_last(self, tmp_path):
        for name in ["exp1_old", "exp1_new", "exp1_mid"]:
            (tmp_path / name).mkdir()
        got = MLArtifact._get_path("exp1", str(tmp_path), None)
        # sorted last because ISO timestamps sort lexicographically
        assert os.path.basename(got) == "exp1_old"

    def test_full_path_returned_as_is(self, tmp_path):
        d = tmp_path / "custom"
        d.mkdir()
        assert MLArtifact._get_path("exp1", str(tmp_path), str(d)) == str(d)

    def test_full_path_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MLArtifact._get_path("exp1", str(tmp_path), str(tmp_path / "nope"))

    def test_artifact_path_none_defaults_to_config(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            MLConfig, "DEFAULT_ARTIFACT_PATH", str(tmp_path / "missing"), raising=False
        )
        with pytest.raises(FileNotFoundError):
            MLArtifact._get_path("exp", None, None)


# ── attach_executors ────────────────────────────────────────────────────
class TestAttachExecutors:
    def test_first_attach_sets_state(self, artifact, stub_executor):
        artifact.attach_executors(_ExperimentLike([stub_executor]))
        assert artifact._executors == [stub_executor]
        assert len(artifact.pipeline_config) == 1
        assert artifact.pipeline_config[0]["class"] == "_StubMLExecutor"

    def test_second_attach_with_same_pipeline_is_ok(self, artifact, stub_executor):
        exp = _ExperimentLike([stub_executor])
        artifact.attach_executors(exp)
        artifact.attach_executors(exp)

    def test_second_attach_with_different_pipeline_raises(self, artifact, stub_executor):
        artifact.attach_executors(_ExperimentLike([stub_executor]))
        with pytest.raises(ValueError):
            artifact.attach_executors(_ExperimentLike([_OtherStubMLExecutor()]))


# ── save_experiment ─────────────────────────────────────────────────────
class TestSaveExperiment:
    def test_no_executors_raises(self, artifact):
        with pytest.raises(ValueError):
            artifact.save_experiment()

    def test_writes_metadata_and_dirs(self, artifact, stub_executor):
        artifact.attach_executors(_ExperimentLike([stub_executor]))
        artifact.save_experiment()

        assert os.path.isdir(artifact.experiment_dir)
        assert os.path.isdir(artifact.models_dir)
        with open(artifact.metadata_file) as f:
            meta = json.load(f)
        assert meta["0"]["qualname"] == "_StubMLExecutor"
        assert meta["0"]["has_model"] is False
        assert meta["0"]["id"] == stub_executor.id

    def test_saves_model_when_fitted(self, artifact):
        ex = _StubMLExecutor(key="e1", value=0.7)
        ex.fit(None)
        artifact.attach_executors(_ExperimentLike([ex]))
        artifact.save_experiment()

        pkls = [f for f in os.listdir(artifact.models_dir) if f.endswith(".pkl")]
        assert len(pkls) == 1
        assert pkls[0].startswith(ex.id)

    def test_skips_model_when_not_fitted(self, artifact, stub_executor):
        artifact.attach_executors(_ExperimentLike([stub_executor]))
        artifact.save_experiment()
        pkls = [f for f in os.listdir(artifact.models_dir) if f.endswith(".pkl")]
        assert pkls == []


# ── _create_executor ────────────────────────────────────────────────────
class TestCreateExecutor:
    def test_rebuilds_class_and_metadata(self, tmp_path, stub_executor):
        models = tmp_path / "models"
        models.mkdir()
        state = stub_executor.get_artifact().json

        obj = MLArtifact._create_executor(state, str(models), "0")
        assert isinstance(obj, _StubMLExecutor)
        assert obj.key == stub_executor.key
        assert obj.id == stub_executor.id
        assert obj.is_fitted is True
        assert obj.has_model is False

    def test_loads_model_when_has_model(self, tmp_path):
        ex = _StubMLExecutor(key="e1", value=0.9)
        ex.fit(None)
        models = tmp_path / "models"
        models.mkdir()
        ex.save_model(str(models), postfix="0")
        state = ex.get_artifact().json

        obj = MLArtifact._create_executor(state, str(models), "0")
        assert obj.has_model is True
        assert obj.model == {"value": 0.9}

    def test_missing_models_dir_when_model_expected(self, tmp_path):
        ex = _StubMLExecutor(key="e1", value=0.9)
        ex.fit(None)
        state = ex.get_artifact().json
        with pytest.raises(FileNotFoundError):
            MLArtifact._create_executor(state, str(tmp_path / "nope"), "0")


# ── load_experiment / round-trip ────────────────────────────────────────
class TestLoadExperiment:
    def test_missing_artifact_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MLArtifact.load_experiment("exp1", artifact_path=str(tmp_path / "nope"))

    def test_no_artifacts_for_id(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MLArtifact.load_experiment("exp1", artifact_path=str(tmp_path))

    def test_missing_metadata_file(self, tmp_path):
        (tmp_path / "exp1_x").mkdir()
        with pytest.raises(FileNotFoundError):
            MLArtifact.load_experiment("exp1", artifact_path=str(tmp_path))

    def test_round_trip_preserves_pipeline(self, tmp_path):
        ex = _StubMLExecutor(key="e1", value=0.7)
        ex.fit(None)
        save_art = MLArtifact("exp1", str(tmp_path), created_at="2026-01-01T00:00:00")
        save_art.attach_executors(_ExperimentLike([ex]))
        save_art.save_experiment()

        loaded = MLArtifact.load_experiment("exp1", artifact_path=str(tmp_path))
        assert len(loaded._executors) == 1
        rebuilt = loaded._executors[0]
        assert isinstance(rebuilt, _StubMLExecutor)
        assert rebuilt.key == ex.key
        assert rebuilt.id == ex.id
        assert rebuilt.has_model is True
        assert rebuilt.model == {"value": 0.7}
        assert loaded.pipeline_config[0]["class"] == "_StubMLExecutor"
