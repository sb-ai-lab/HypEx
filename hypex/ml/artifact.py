"""Artifact management for ML experiments: save and load entire pipelines.

This module provides the :class:`MLArtifact` class, which acts as an
**orchestrator** for persisting and restoring the state of an
:class:`~hypex.experiments.ml.MLExperiment`.

The core design principle is **separation of concerns**: ``MLArtifact``
knows *what* to save (the pipeline structure and the list of executors)
but delegates *how* to serialize each executor to the executors themselves
via the ``MLArtifactMixin`` protocol.

Key design decisions:
- All executor states are stored in a **single** ``metadata.json`` file,
  making it easy to inspect the pipeline visually.
- Executors are stored as a **flat sequential list** (no hardcoded
  splitters/transformers/ml_executors stages).
- Models are saved as separate binary files alongside the metadata.
- Executors are **reconstructed** from the saved state via
  ``importlib``, so no pre-built experiment is needed for loading.

Disk Structure
--------------
::

    .hypex_experiments/                 # DEFAULT_ARTIFACT_PATH
    +-- my_experiment_20261023_120000/  # <experiment_id>_<created_at>
        +-- metadata.json               # all executor states in one file
        +-- models/                     # heavy models (binary)
            +-- 0_CUPACExecutor_abc123.pkl
            +-- 1_MLFillna_def456.pkl
"""

from __future__ import annotations

import importlib
import json
import os
from datetime import datetime
from typing import Any

from ..config import MLConfig
from ..executor import Executor, MLExecutor


class MLArtifact:
    """Orchestrates saving and loading of ML experiment pipelines.

    This class manages the persistence and restoration of ML experiment state,
    including pipeline structure, executor configurations, and trained models.

    The class follows a separation of concerns principle: it knows *what* to save
    (the pipeline structure and list of executors) but delegates *how* to serialize
    each executor to the executors themselves via the ``MLArtifactMixin`` protocol.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment.
    base_dir : str, optional
        Base directory for storing artifacts. Defaults to ``MLConfig.DEFAULT_ARTIFACT_PATH``.
    created_at : str, optional
        Creation timestamp in ISO format. Defaults to current datetime.

    Attributes
    ----------
    experiment_id : str
        The unique identifier for this experiment.
    base_dir : str
        The base directory where artifacts are stored.
    created_at : str
        The creation timestamp in ISO format.
    experiment_dir : str
        Full path to the experiment's artifact directory.
    metadata_file : str
        Path to the metadata JSON file.
    models_dir : str
        Path to the directory containing saved models.
    _executors : list[MLExecutor]
        Internal list of executor instances.
    pipeline_config : list[dict[str, Any]]
        Configuration of the pipeline structure.
    """

    def __init__(
        self,
        experiment_id: str,
        base_dir: str | None,
        created_at: str | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.base_dir = base_dir or MLConfig.DEFAULT_ARTIFACT_PATH
        self.created_at = created_at or datetime.now().isoformat()

        self._executors: list[MLExecutor] = []
        self.pipeline_config: list[dict[str, Any]] = []

        safe_created_at = self.created_at.replace(":", "-").replace(".", "-")
        normalized_name = f"{experiment_id}_{safe_created_at}"
        self.experiment_dir = os.path.join(self.base_dir, normalized_name)
        self.metadata_file = os.path.join(self.experiment_dir, "metadata.json")
        self.models_dir = os.path.join(self.experiment_dir, "models")

    # ── attachment of executors ────────────────────────────────────────────

    def attach_executors(self, experiment: Any) -> None:
        """Bind live executor instances to this artifact.

        Parameters
        ----------
        experiment : MLExperiment
            The ML experiment instance containing executors to bind.

        Raises
        ------
        ValueError
            If the provided executor configuration is incompatible with the
            existing pipeline configuration.
        """
        self._executors = experiment.executors
        config = self._build_pipeline_config(self._executors)

        if self.pipeline_config is None:
            self.pipeline_config = config
        else:
            valid, massage = self._validate_config(self.pipeline_config, config)
            if not valid:
                raise ValueError(massage)

    @staticmethod
    def _build_pipeline_config(executors: list[Executor | MLExecutor]) -> list[dict[str, Any]]:
        """Serialize the pipeline structure from a list of executors.

        Parameters
        ----------
        executors : list[Executor | MLExecutor]
            The list of executors to serialize into a pipeline configuration.

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries, each containing serialized executor metadata
            including class name, module, ID, key, and parameters.
        """

        def _ser(ex: MLExecutor) -> dict[str, Any]:
            return {
                "class": type(ex).__name__,
                "module": type(ex).__module__,
                "id": ex.id,
                "key": ex.key,
                "params": ex.get_params(),
            }

        return [_ser(e) for e in executors]

    @staticmethod
    def _validate_config(
        saved_config: list[dict[str, Any]],
        current_config: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """Validate compatibility between saved and current pipeline configurations.

        Parameters
        ----------
        saved_config : list[dict[str, Any]]
            The saved pipeline configuration (list of serialized executor dicts).
        current_config : list[dict[str, Any]]
            The current pipeline configuration.

        Returns
        -------
        tuple[bool, str]
            A tuple of (is_valid, message) where is_valid is True if configurations
            are compatible, and message provides details if they are not.
        """
        if len(saved_config) != len(current_config):
            return False, (
                f"Incorrect number of executors: saved {len(saved_config)}, current {len(current_config)}"
            )
        for idx in range(len(saved_config)):
            saved_items = saved_config[idx]
            current_items = current_config[idx]

            for i, (s, c) in enumerate(zip(saved_items, current_items)):
                if s["class"] != type(c).__name__:
                    return False, (
                        f"stage №{i}: expected {s['class']}, got {type(c).__name__}"
                    )

        return True, "Compatible"

    # ── save experiment ────────────────────────────────────────────

    @staticmethod
    def _state_to_json(executor: MLExecutor) -> dict[str, Any]:
        """Convert an executor's state to a JSON-serializable dictionary.

        Parameters
        ----------
        executor : MLExecutor
            The executor whose state should be serialized.

        Returns
        -------
        dict[str, Any]
            JSON-serializable dictionary representation of the executor's state.
        """
        executor_state = executor.get_artifact()
        return executor_state.json

    def save_experiment(self) -> None:
        """Save the experiment's state to disk.

        Persists all executor states and trained models to the artifact directory.
        Creates the necessary directory structure if it doesn't exist.

        Raises
        ------
        ValueError
            If the base directory doesn't exist or no executors are attached.
        """
        if len(self._executors) == 0:
            raise ValueError("First attach executors to MLArtifact.")

        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        output_json: dict[str, Any] = {}

        for idx, executor in enumerate(self._executors):
            output_json[str(idx)] = self._state_to_json(executor)
            if isinstance(executor, MLExecutor):
                executor.save_model(self.models_dir, prefix=str(idx))

        with open(self.metadata_file, "w") as f:
            json.dump(output_json, f, indent=4)

    # ── load experiment ────────────────────────────────────────────

    @staticmethod
    def _get_path(experiment_id: str, artifact_path: str | None, full_path: str | None) -> str:
        """Resolve the artifact directory path for an experiment.

        Parameters
        ----------
        experiment_id : str
            The unique identifier of the experiment.
        artifact_path : str, optional
            Base path where artifacts are stored. Defaults to
            ``MLConfig.DEFAULT_ARTIFACT_PATH``.
        full_path : str, optional
            If provided, use this as the exact path instead of searching.

        Returns
        -------
        str
            The resolved path to the experiment's artifact directory.

        Raises
        ------
        FileNotFoundError
            If the artifact path doesn't exist or no matching experiment is found.
        FileExistsError
            If no experiment artifacts exist for the given ID.
        """
        if artifact_path is None:
            artifact_path = MLConfig.DEFAULT_ARTIFACT_PATH

        if not os.path.isdir(artifact_path):
            raise FileNotFoundError(
                f"Artifact path does not exist: {artifact_path}"
            )

        if full_path is None:
            relevant_files_list = sorted(
                element
                for element in os.listdir(artifact_path)
                if element.startswith(experiment_id)
                and os.path.isdir(os.path.join(artifact_path, element))
            )
            # take most recent artifact
            if len(relevant_files_list) == 0:
                raise FileNotFoundError(
                    f"No experiment artifacts found for "
                    f"experiment_id={experiment_id!r} in {artifact_path!r}"
                )
            return os.path.join(artifact_path, relevant_files_list[-1])
        else:
            if not os.path.isdir(full_path):
                raise FileNotFoundError(
                    f"You wrote `full_path`. Full path does not exist: {full_path}"
                )
            return full_path

    @staticmethod
    def _create_executor(
        state: dict[str, Any],
        models_dir: str,
        prefix: str
    ) -> MLExecutor:
        """Reconstruct an executor instance from its saved state.

        Dynamically imports the executor's class and instantiates it
        with the saved parameters, then restores its ID, key, and
        fitted status.  If the executor has a model, loads it from disk.

        Parameters
        ----------
        state : dict[str, Any]
            The saved state of the executor.
        models_dir : str
            Directory where model files are stored.
        prefix : str
            Prefix used for the model filename.

        Returns
        -------
        MLExecutor | Executor
            A reconstructed executor instance with restored state.
        """
        module_name = state["module"]
        qual_name = state["qualname"]

        module = importlib.import_module(module_name)
        cls = module
        for part in qual_name.split('.'):
            cls = getattr(cls, part)

        obj = cls(**state["params"])
        obj.id = state["id"]
        obj.key = state["key"]
        obj.is_fitted = state.get("fitted", False)

        if state.get("has_model", False):
            if not os.path.exists(models_dir):
                raise FileNotFoundError(
                    f"Models directory does not exist: {models_dir}"
                )
            obj.load_model(models_dir, prefix)

        return obj

    @classmethod
    def load_experiment(
        cls,
        experiment_id: str,
        artifact_path: str | None = None,
        full_path: str | None = None
    ) -> MLArtifact:
        """Load an experiment from its saved artifacts.

        Reconstructs the experiment's pipeline by loading executor states
        from the metadata file and trained models from the models directory.

        Parameters
        ----------
        experiment_id : str
            The unique identifier of the experiment to load.
        artifact_path : str, optional
            Base directory where artifacts are stored. Defaults to
            ``MLConfig.DEFAULT_ARTIFACT_PATH``.
        full_path : str, optional
            If provided, use this exact path instead of searching for the
            experiment in artifact_path.

        Returns
        -------
        MLArtifact
            A new MLArtifact instance with the loaded experiment's state,
            including reconstructed executors and pipeline configuration.

        Raises
        ------
        FileNotFoundError
            If the artifact path doesn't exist, no matching experiment is found,
            or the metadata file is missing.
        FileExistsError
            If no experiment artifacts exist for the given ID.
        """
        path = cls._get_path(experiment_id, artifact_path, full_path)
        metadata_path = os.path.join(path, "metadata.json")
        models_dir = os.path.join(path, "models")

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"No metadata file found at: {metadata_path}"
            )

        metadata = {}
        with open(metadata_path) as f:
            metadata: dict[str, Any] = json.load(f)

        executors = []
        for idx, state in sorted(metadata.items(), key=lambda x: x[0]):
            executors.append(
                cls._create_executor(state, models_dir, str(idx))
            )

        artifact =  cls(
            experiment_id=experiment_id,
            base_dir=artifact_path or MLConfig.DEFAULT_ARTIFACT_PATH
        )
        artifact._executors = executors
        artifact.pipeline_config = cls._build_pipeline_config(executors)
        return artifact

