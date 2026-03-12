from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


class MLExecutorState:
    """Serializable fitted state for MLExecutor instances."""

    def __init__(
        self,
        executor_id: str,
        executor_class: str,
        fitted_params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.executor_id = executor_id
        self.executor_class = executor_class
        self.fitted_params = fitted_params
        self.metadata = metadata or {}
        self.fitted_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executor_id": self.executor_id,
            "executor_class": self.executor_class,
            "fitted_params": self.fitted_params,
            "metadata": self.metadata,
            "fitted_at": self.fitted_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MLExecutorState:
        # Backward compatibility with legacy TransformerParams keys.
        executor_id = data.get("executor_id", data.get("transformer_id"))
        executor_class = data.get("executor_class", data.get("transformer_class"))

        state = cls(
            executor_id=executor_id,
            executor_class=executor_class,
            fitted_params=data["fitted_params"],
            metadata=data.get("metadata"),
        )
        state.fitted_at = data.get("fitted_at")
        return state

    def __repr__(self) -> str:
        return (
            f"MLExecutorState(id={self.executor_id}, "
            f"class={self.executor_class}, "
            f"params={list(self.fitted_params.keys())})"
        )
