from __future__ import annotations

import pickle
from typing import Any

from .dataset import Dataset, ExperimentData


class MLExperimentData(ExperimentData):
    """ExperimentData with ML pipeline support.

    Extends ExperimentData with:
    - pipeline: ordered list of executor IDs that have been applied
    - artifacts: fitted objects (models, scalers, etc.) keyed by executor ID
    """

    def __init__(self, data: Dataset):
        """Initialise with a Dataset, empty pipeline, and empty artifacts.

        Args:
            data: Source dataset wrapped in the experiment container.
        """
        super().__init__(data)
        self.pipeline: list[str] = []
        self.artifacts: dict[str, Any] = {}

    @classmethod
    def from_experiment_data(cls, data: ExperimentData) -> MLExperimentData:
        """Create MLExperimentData from an existing ExperimentData."""
        result = cls(data.ds)
        result.additional_fields = data.additional_fields
        result.variables = data.variables
        result.groups = data.groups
        result.analysis_tables = data.analysis_tables
        result.id_name_mapping = data.id_name_mapping
        if isinstance(data, MLExperimentData):
            result.pipeline = list(data.pipeline)
            result.artifacts = dict(data.artifacts)
        return result

    def copy(self, data: Dataset | None = None) -> MLExperimentData:
        """Deep copy with optional data replacement."""
        from copy import deepcopy

        result = deepcopy(self)
        if data is not None:
            result._data = data
        return result

    def save(self, path: str) -> None:
        """Save pipeline state and artifacts to disk."""
        state = {
            "pipeline": self.pipeline,
            "artifacts": self.artifacts,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_artifacts(self, path: str) -> None:
        """Load pipeline state and artifacts from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        self.pipeline = state["pipeline"]
        self.artifacts = state["artifacts"]
