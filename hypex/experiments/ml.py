"""ML experiment pipeline with artifact management.

This module provides the :class:`MLExperiment` class, which extends the base
:class:`~hypex.experiments.base.Experiment` with machine learning capabilities:
model training, prediction, and artifact persistence.

Key Features
------------
- **Mode-based execution**: Run experiments in ``FIT``, ``PREDICT``, or
  ``FIT_PREDICT`` mode via the ``mode`` parameter.
- **Artifact management**: Automatically save and load executor states and
  trained models using :class:`~hypex.ml.artifact.MLArtifact`.
- **Default artifact path**: Uses ``.hypex_experiments`` by default.
  Users only need to specify ``key`` (experiment ID).
"""

from __future__ import annotations

from typing import Sequence

from ..config import MLConfig
from ..dataset import ExperimentData, MLExperimentData
from ..executor import Executor
from ..ml import MLArtifact
from ..utils import MLModeEnum
from .base import Experiment


class MLExperiment(Experiment):
    """Machine learning experiment pipeline with artifact management.

    ``MLExperiment`` extends the base :class:`~hypex.experiments.base.Experiment`
    to support ML-specific workflows: model training, prediction, and automatic
    persistence of executor states and trained models.

    The experiment operates in three modes:

    - **FIT**: Train all ML executors on the data.
    - **PREDICT**: Load trained models and perform inference.
    - **FIT_PREDICT**: Train models, then immediately predict (default).

    Artifacts are stored in ``<artifact_path>/<key>/``.  By default,
    ``artifact_path`` is ``".hypex_experiments"``.

    Parameters
    ----------
    executors : Sequence[Executor]
        List of executors to run in the pipeline.
    mode : MLModeEnum | str, optional
        Execution mode.  Defaults to ``MLModeEnum.FIT_PREDICT``.
    save_models : bool, optional
        Whether to save artifacts after execution.  Defaults to ``True``.
    artifact_path : str | None, optional
        Root path to the artifact directory.  If ``None``, uses
        ``MLConfig.DEFAULT_ARTIFACT_PATH``.
    key : str, optional
        Identifier for the experiment.  **Required** when ``mode`` is
        ``PREDICT``.
    """

    def __init__(
        self,
        executors: Sequence[Executor],
        mode: MLModeEnum | str = MLModeEnum.FIT_PREDICT,
        save_models: bool = True,
        artifact_path: str | None = None,
        key: str = "",
    ) -> None:
        """Initialize an MLExperiment instance.

        Parameters
        ----------
        executors : Sequence[Executor]
            List of executors to run in the pipeline.
        mode : MLModeEnum | str, optional
            Execution mode.  Defaults to ``MLModeEnum.FIT_PREDICT``.
        save_models : bool, optional
            Whether to save artifacts after execution.  Defaults to ``True``.
        artifact_path : str | None, optional
            Root path to the artifact directory.  If ``None``, uses
            ``MLConfig.DEFAULT_ARTIFACT_PATH``.
        key : str, optional
            Identifier for the experiment.  Defaults to ``""``.

        Raises
        ------
        ValueError
            If ``mode`` is ``PREDICT`` and ``key`` is empty.
        """
        # Convert string mode to enum
        if isinstance(mode, str):
            mode = MLModeEnum(mode)

        self._validate_key(mode, key)

        self.mode = mode
        self.save_models = save_models
        self.artifact_path = artifact_path or MLConfig.DEFAULT_ARTIFACT_PATH

        super().__init__(executors=executors, key=key)

    @staticmethod
    def _validate_key(mode: MLModeEnum, key: str) -> None:
        """Validate that ``key`` is provided when required.

        Parameters
        ----------
        mode : MLModeEnum
            Execution mode.
        key : str
            Experiment identifier.

        Raises
        ------
        ValueError
            If ``mode`` is ``PREDICT`` and ``key`` is empty.
        """
        if mode == MLModeEnum.PREDICT and not key:
            raise ValueError(
                "key is required for PREDICT mode. "
                "Provide an experiment identifier so that the artifact "
                "can be located in the artifact directory."
            )

    @property
    def all_executors(self) -> list[Executor]:
        """Get all executors in the pipeline.

        Returns
        -------
        list[Executor]
            List of all executors in execution order.
        """
        return list(self.executors)

    @property
    def splitters(self) -> list[Executor]:
        """Get all splitter executors in the pipeline.

        Returns
        -------
        list[Executor]
            List of splitter executors (currently empty for MLExperiment).
        """
        return []

    @property
    def transformers(self) -> list[Executor]:
        """Get all transformer executors in the pipeline.

        Returns
        -------
        list[Executor]
            List of transformer executors.
        """
        return [ex for ex in self.executors if getattr(ex, "_is_transformer", False)]

    @property
    def ml_executors(self) -> list[Executor]:
        """Get all ML executors in the pipeline.

        Returns
        -------
        list[Executor]
            List of ML executors (instances of MLExecutor).
        """
        from ..executor import MLExecutor

        return [ex for ex in self.executors if isinstance(ex, MLExecutor)]

    def execute(self, data: ExperimentData) -> ExperimentData:
        """Execute the ML experiment pipeline.

        Runs the executor pipeline in the configured mode:

        - In ``PREDICT`` mode: loads artifacts before execution.
        - In ``FIT`` / ``FIT_PREDICT`` modes: saves artifacts after execution.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to process.

        Returns
        -------
        ExperimentData
            The processed experiment data with results from all executors.

        Raises
        ------
        ValueError
            If ``mode`` is ``PREDICT`` and ``key`` is empty.
        FileNotFoundError
            If the artifact directory or experiment does not exist.
        """
        # Re-validate in case mode was changed after construction.
        self._validate_key(self.mode, self.key)

        # Load artifacts if in PREDICT mode
        if self.mode == MLModeEnum.PREDICT:
            self._load_artifacts()

        # Execute the pipeline
        result = super().execute(data)

        # Save artifacts if configured
        if self.save_models:
            self._save_artifacts()

        return result

    def _save_artifacts(self) -> None:
        """Save executor states and models to disk."""
        artifact = MLArtifact.from_experiment(
            ml_experiment=self,
            artifact_path=self.artifact_path,
        )
        artifact.save()

    def _load_artifacts(self) -> None:
        """Load executor states and models from disk.

        Raises
        ------
        FileNotFoundError
            If the artifact directory or experiment does not exist.
        ValueError
            If the saved pipeline is incompatible with the current experiment.
        """
        artifact = MLArtifact.load_experiment(
            experiment_id=str(self.key),
            artifact_path=self.artifact_path,
        )

        # Validate compatibility
        is_compatible, message = artifact.validate_compatibility(self)
        if not is_compatible:
            raise ValueError(f"Incompatible artifact: {message}")

        # Attach executors and load
        artifact.attach_executors(self.all_executors)
        artifact.load()

    def save(self, path: str | None = None) -> MLArtifact:
        """Manually save the experiment to an artifact.

        Parameters
        ----------
        path : str | None, optional
            Root path to save the artifact.  If ``None``, uses
            ``artifact_path``.

        Returns
        -------
        MLArtifact
            The saved artifact instance.
        """
        save_path = path or self.artifact_path

        artifact = MLArtifact.from_experiment(
            ml_experiment=self,
            artifact_path=save_path,
        )
        artifact.save()

        return artifact

    @classmethod
    def load(
        cls,
        experiment_id: str,
        executors: Sequence[Executor] | None = None,
        mode: MLModeEnum | str = MLModeEnum.PREDICT,
        artifact_path: str | None = None,
    ) -> MLExperiment:
        """Load an experiment from an artifact on disk.

        Parameters
        ----------
        experiment_id : str
            The experiment identifier to load.
        executors : Sequence[Executor] | None, optional
            List of executors to use.  If ``None``, the experiment is created
            with an empty executor list.
        mode : MLModeEnum | str, optional
            Execution mode.  Defaults to ``MLModeEnum.PREDICT``.
        artifact_path : str | None, optional
            Root artifact directory.  If ``None``, uses
            ``MLConfig.DEFAULT_ARTIFACT_PATH``.

        Returns
        -------
        MLExperiment
            A new experiment instance with loaded artifacts.

        Raises
        ------
        FileNotFoundError
            If no artifact matching ``experiment_id`` is found.
        ValueError
            If the saved pipeline is incompatible with the provided executors.
        """
        if artifact_path is None:
            artifact_path = MLConfig.DEFAULT_ARTIFACT_PATH

        artifact = MLArtifact.load_experiment(
            experiment_id=experiment_id,
            artifact_path=artifact_path,
        )

        if executors is None:
            executors = []

        exp = cls(
            executors=executors,
            mode=mode,
            artifact_path=artifact_path,
            key=experiment_id,
        )

        is_compatible, message = artifact.validate_compatibility(exp)
        if not is_compatible:
            raise ValueError(f"Incompatible artifact: {message}")

        artifact.attach_executors(exp.all_executors)
        artifact.load()

        return exp

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
        return MLArtifact.list_experiments(artifact_path)

    def _ensure_ml_data(self, data: ExperimentData) -> MLExperimentData:
        """Ensure the data is an MLExperimentData instance.

        Parameters
        ----------
        data : ExperimentData
            The input data.

        Returns
        -------
        MLExperimentData
            The data as an MLExperimentData instance.
        """
        if isinstance(data, MLExperimentData):
            return data

        ml_data = MLExperimentData(data=data.ds)
        ml_data.variables = data.variables
        ml_data.groups = data.groups
        ml_data.analysis_tables = data.analysis_tables
        ml_data.id_name_mapping = data.id_name_mapping

        return ml_data

    def _configure_for_inference(self, ml: MLExperimentData) -> MLExperimentData:
        """Configure the data for inference mode.

        Parameters
        ----------
        ml : MLExperimentData
            The ML experiment data.

        Returns
        -------
        MLExperimentData
            The configured data for inference.
        """
        return ml
