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
from ..ml.artifact import MLArtifact
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

        self._artifacts_loaded = False

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
        self._validate_key(self.mode, self.key)

        result = super().execute(data)

        if self.save_models and self.mode in [MLModeEnum.FIT, MLModeEnum.FIT_PREDICT]:
            self.save()

        return result

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
        artifact = MLArtifact(
            experiment_id=str(self.key),
            base_dir=save_path,
        )
        artifact.attach_executors(self)
        artifact.save_experiment()
        return artifact

    def load(self) -> None:
        """Load executors and models from a saved artifact.

        Restores the pipeline by loading executor states and models
        from the artifact directory identified by ``key`` and
        ``artifact_path``.  If executors were provided at construction,
        validates compatibility before replacing them.  If no executors
        were provided, the loaded executors are used as-is.

        This method is **idempotent**: calling it multiple times will
        only load once.

        Raises
        ------
        FileNotFoundError
            If no artifact matching ``key`` is found.
        ValueError
            If the saved pipeline is incompatible with the current
            executors.

        Examples
        --------
        **Load after construction**::

            exp = MLExperiment(
                executors=[CUPACExecutor(...)],
                key="my_cupac",
            )
            exp.load()
            result = exp.execute(new_data)

        **Load without pre-built executors**::

            exp = MLExperiment(executors=[], key="my_cupac")
            exp.load()
            result = exp.execute(new_data)
        """
        if self._artifacts_loaded:
            return

        artifact = MLArtifact.load_experiment(
            experiment_id=str(self.key),
            artifact_path=self.artifact_path,
        )

        # Validate compatibility only if executors were provided
        if self.executors:
            current_config = MLArtifact._build_pipeline_config(
                list(self.executors)
            )
            is_valid, message = MLArtifact._validate_config(
                artifact.pipeline_config, current_config
            )
            if not is_valid:
                raise ValueError(f"Incompatible artifact: {message}")

        # Replace executors with loaded ones
        self.executors = artifact._executors
        self._artifacts_loaded = True

    def reload(self) -> None:
        """Force re-reading of the artifact, bypassing the idempotency cache."""
        self._artifacts_loaded = False
        self.load()

    # ── data helpers ────────────────────────────────────────────────

    def _ensure_ml_data(self, data: ExperimentData) -> MLExperimentData:
        """Ensure the data is an MLExperimentData instance."""
        if isinstance(data, MLExperimentData):
            return data
        ml_data = MLExperimentData(data=data.ds)
        ml_data.variables = data.variables
        ml_data.groups = data.groups
        ml_data.analysis_tables = data.analysis_tables
        ml_data.id_name_mapping = data.id_name_mapping
        return ml_data

    def _configure_for_inference(
        self, ml: MLExperimentData
    ) -> MLExperimentData:
        """Configure the data for inference mode (no-op for now)."""
        return ml
