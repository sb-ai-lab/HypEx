"""Artifact management for ML experiments: save and load entire pipelines.

This module provides the :class:`MLArtifact` class, which acts as an
**orchestrator** for persisting and restoring the state of an
:class:`~hypex.experiments.ml.MLExperiment`.

The core design principle is **separation of concerns**: ``MLArtifact``
knows *what* to save (the pipeline structure and the list of executors)
but delegates *how* to serialize each executor to the executors themselves
via the ``MLArtifactMixin`` protocol.

Disk Structure
--------------
::

    .hypex_experiments/                 # DEFAULT_ARTIFACT_PATH
    +-- my_experiment/                  # <experiment_id>
        +-- experiment.json             # manifest
        +-- ml_executors/               # lightweight states (JSON)
        +-- models/                     # heavy models (binary)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Iterable

from ..config import MLConfig
from ..executor import MLExecutor


class MLArtifact:
    """Orchestrator for persisting and restoring ML experiment pipelines.

    ``MLArtifact`` manages the complete lifecycle of an ML experiment:
    saving the pipeline configuration, executor states, and trained models
    to disk, and restoring them for subsequent inference runs.

    The class follows the **orchestrator pattern**: it does not know how to
    serialize specific executor types or model formats.  Instead, it delegates
    serialization to the executors themselves via the ``MLArtifactMixin``
    protocol (``get_artifact``, ``set_artifact``, ``save_model``, ``load_model``).

    Attributes
    ----------
    experiment_id : str
        Unique identifier for the experiment.
    experiment_class : str
        Name of the ``MLExperiment`` class.
    base_dir : str
        Full path to the artifact directory on disk.
    pipeline_config : dict[str, Any]
        Serialized pipeline structure containing executor metadata.
    created_at : str
        ISO-8601 timestamp of artifact creation.
    """

    def __init__(
        self,
        experiment_id: str,
        experiment_class: str,
        base_dir: str,
        pipeline_config: dict[str, Any],
        created_at: str | None = None,
    ) -> None:
        """Initialize an MLArtifact instance.

        Parameters
        ----------
        experiment_id : str
            Unique identifier for the experiment.
        experiment_class : str
            Name of the experiment class.
        base_dir : str
            Full path to the artifact directory on disk.
        pipeline_config : dict[str, Any]
            Serialized pipeline configuration.
        created_at : str | None, optional
            ISO-8601 timestamp of creation.  If ``None``, uses current time.
        """
        self.experiment_id = experiment_id
        self.experiment_class = experiment_class
        self.base_dir = base_dir
        self.pipeline_config = pipeline_config
        self.created_at = created_at or datetime.now().isoformat()

        self._executors: list[MLExecutor] = []

        self.metadata_file = os.path.join(base_dir, "experiment.json")
        self.ml_executors_dir = os.path.join(base_dir, "ml_executors")
        self.models_dir = os.path.join(base_dir, "models")

    # -- attachment of executors --

    def attach_executors(self, executors: Iterable[MLExecutor]) -> None:
        """Bind live executor instances to this artifact.

        Parameters
        ----------
        executors : Iterable[MLExecutor]
            Iterable of executor instances to bind.
        """
        self._executors = list(executors)

    # -- factory --

    @classmethod
    def from_experiment(
        cls,
        ml_experiment: Any,
        artifact_path: str | None = None,
        experiment_id: str | None = None,
    ) -> MLArtifact:
        """Create an artifact from a live MLExperiment instance.

        Parameters
        ----------
        ml_experiment : MLExperiment
            The experiment instance to serialize.
        artifact_path : str | None, optional
            Root directory for storing artifacts.  If ``None``, uses
            ``MLConfig.DEFAULT_ARTIFACT_PATH``.
        experiment_id : str | None, optional
            Custom experiment identifier.  If ``None``, generated from
            ``ml_experiment.key`` or current timestamp.

        Returns
        -------
        MLArtifact
            A new artifact instance ready for :meth:`save`.
        """
        if experiment_id is None:
            experiment_id = str(ml_experiment.key) or (
                f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        if artifact_path is None:
            artifact_path = MLConfig.DEFAULT_ARTIFACT_PATH

        full_path = os.path.join(artifact_path, experiment_id)

        artifact = cls(
            experiment_id=experiment_id,
            experiment_class=type(ml_experiment).__name__,
            base_dir=full_path,
            pipeline_config=cls._build_pipeline_config(ml_experiment),
        )

        artifact.attach_executors(ml_experiment.all_executors)

        return artifact

    @classmethod
    def load_from_directory(cls, base_dir: str) -> MLArtifact:
        """Restore an artifact from a directory on disk.

        Reads only the ``experiment.json`` manifest and reconstructs the
        artifact metadata.  Does **not** load executor states or models.

        Parameters
        ----------
        base_dir : str
            Path to the artifact directory (must contain ``experiment.json``).

        Returns
        -------
        MLArtifact
            A new artifact instance with metadata restored.

        Raises
        ------
        FileNotFoundError
            If ``experiment.json`` does not exist in the directory.
        """
        metadata_file = os.path.join(base_dir, "experiment.json")

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Experiment metadata not found: {metadata_file}")

        with open(metadata_file) as f:
            data = json.load(f)

        return cls(
            experiment_id=data["experiment_id"],
            experiment_class=data["experiment_class"],
            base_dir=base_dir,
            pipeline_config=data["pipeline_config"],
            created_at=data.get("created_at"),
        )

    # -- discovery & parsing by experiment_id --

    @staticmethod
    def list_experiments(artifact_path: str | None = None) -> list[str]:
        """Discover available experiment IDs in an artifact directory.

        Parameters
        ----------
        artifact_path : str | None, optional
            Root artifact directory.  If ``None``, uses
            ``MLConfig.DEFAULT_ARTIFACT_PATH``.

        Returns
        -------
        list[str]
            Sorted list of experiment IDs found in the directory.
        """
        if artifact_path is None:
            artifact_path = MLConfig.DEFAULT_ARTIFACT_PATH

        if not os.path.isdir(artifact_path):
            return []

        result: list[str] = []
        for entry in sorted(os.listdir(artifact_path)):
            candidate = os.path.join(artifact_path, entry)
            manifest = os.path.join(candidate, "experiment.json")
            if os.path.isdir(candidate) and os.path.exists(manifest):
                try:
                    with open(manifest) as f:
                        data = json.load(f)
                    exp_id = data.get("experiment_id")
                    if exp_id:
                        result.append(str(exp_id))
                except (json.JSONDecodeError, OSError):
                    continue
        return result

    @classmethod
    def load_experiment(
        cls,
        experiment_id: str,
        artifact_path: str | None = None,
    ) -> MLArtifact:
        """Load an artifact by experiment ID.

        Searches inside ``artifact_path`` for a subdirectory matching
        ``experiment_id`` and loads the artifact from there.

        Parameters
        ----------
        experiment_id : str
            The experiment identifier to load.
        artifact_path : str | None, optional
            Root artifact directory.  If ``None``, uses
            ``MLConfig.DEFAULT_ARTIFACT_PATH``.

        Returns
        -------
        MLArtifact
            A new artifact instance with metadata restored.

        Raises
        ------
        FileNotFoundError
            If no artifact matching ``experiment_id`` is found.
        """
        if artifact_path is None:
            artifact_path = MLConfig.DEFAULT_ARTIFACT_PATH

        if not os.path.isdir(artifact_path):
            raise FileNotFoundError(
                f"Artifact path does not exist: {artifact_path}"
            )

        # Fast path: direct subdirectory match.
        candidate_dir = os.path.join(artifact_path, experiment_id)
        manifest = os.path.join(candidate_dir, "experiment.json")
        if os.path.isdir(candidate_dir) and os.path.exists(manifest):
            return cls.load_from_directory(candidate_dir)

        # Slow path: scan all subdirectories for a matching experiment_id
        available = cls.list_experiments(artifact_path)
        for exp_id in available:
            sub_dir = os.path.join(artifact_path, exp_id)
            sub_manifest = os.path.join(sub_dir, "experiment.json")
            if not os.path.exists(sub_manifest):
                continue
            try:
                with open(sub_manifest) as f:
                    data = json.load(f)
                if data.get("experiment_id") == experiment_id:
                    return cls.load_from_directory(sub_dir)
            except (json.JSONDecodeError, OSError):
                continue

        raise FileNotFoundError(
            f"No artifact found for experiment_id={experiment_id!r} "
            f"in {artifact_path!r}. "
            f"Available experiments: {available or '(none)'}"
        )

    @staticmethod
    def _build_pipeline_config(ml_experiment: Any) -> dict[str, Any]:
        """Serialize the pipeline structure from an experiment.

        Parameters
        ----------
        ml_experiment : MLExperiment
            The experiment instance to serialize.

        Returns
        -------
        dict[str, Any]
            Pipeline configuration dictionary.
        """

        def _ser(ex: MLExecutor) -> dict[str, Any]:
            return {
                "class": type(ex).__name__,
                "module": type(ex).__module__,
                "id": ex.id,
                "key": ex.key,
                "params": ex.get_params(),
            }

        return {
            "splitters": [_ser(e) for e in ml_experiment.splitters],
            "transformers": [_ser(e) for e in ml_experiment.transformers],
            "ml_executors": [_ser(e) for e in ml_experiment.ml_executors],
        }

    # -- Save --

    def save(self) -> None:
        """Persist the artifact to disk."""
        self._create_directory_structure()
        self._save_metadata()

        for executor in self._executors:
            self._save_executor(executor)

    def _create_directory_structure(self) -> None:
        """Create the artifact directory structure on disk."""
        os.makedirs(self.ml_executors_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def _save_metadata(self) -> None:
        """Save the experiment manifest to ``experiment.json``."""
        metadata = {
            "experiment_id": self.experiment_id,
            "experiment_class": self.experiment_class,
            "created_at": self.created_at,
            "pipeline_config": self.pipeline_config,
        }

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _save_executor(self, executor: MLExecutor) -> None:
        """Save a single executor's state and model to disk.

        Parameters
        ----------
        executor : MLExecutor
            The executor instance to save.
        """
        # 1) the lightweight part -> json
        state_file = os.path.join(
            self.ml_executors_dir, f"{executor.id_for_name}.json"
        )

        with open(state_file, "w") as f:
            json.dump(executor.get_artifact(), f, indent=2, default=str)

        # 2) the heavy part -> binary file (delegate to executor)
        if executor.has_model:
            model_dir = os.path.join(self.models_dir, executor.id_for_name)
            executor.save_model(model_dir)

    # -- Load --

    def load(self) -> None:
        """Restore executor states and models from disk."""
        for executor in self._executors:
            self._load_executor(executor)

    def _load_executor(self, executor: MLExecutor) -> None:
        """Load a single executor's state and model from disk.

        Parameters
        ----------
        executor : MLExecutor
            The executor instance to restore.
        """
        # 1) state
        state_file = os.path.join(
            self.ml_executors_dir, f"{executor.id_for_name}.json"
        )

        if os.path.exists(state_file):
            with open(state_file) as f:
                executor.set_artifact(json.load(f))

        # 2) model
        model_dir = os.path.join(self.models_dir, executor.id_for_name)

        if os.path.exists(model_dir):
            executor.load_model(model_dir)

    # -- validation --

    def validate_compatibility(self, ml_experiment: Any) -> tuple[bool, str]:
        """Check if the saved pipeline is compatible with the current experiment.

        Parameters
        ----------
        ml_experiment : MLExperiment
            The current experiment instance to validate against.

        Returns
        -------
        tuple[bool, str]
            A tuple of ``(is_compatible, message)``.
        """
        saved = self.pipeline_config

        for stage in ("splitters", "transformers", "ml_executors"):
            saved_items = saved.get(stage, [])
            current_items = getattr(ml_experiment, stage, [])

            if len(saved_items) != len(current_items):
                return False, (
                    f"{stage}: saved {len(saved_items)}, current {len(current_items)}"
                )

            for i, (s, c) in enumerate(zip(saved_items, current_items)):
                if s["class"] != type(c).__name__:
                    return False, (
                        f"{stage}[{i}]: expected {s['class']}, got {type(c).__name__}"
                    )

        return True, "Compatible"
