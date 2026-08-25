"""
Base abstract class for experiment trackers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTracker(ABC):
    """
    Abstract base class for experiment tracking.
    
    All tracker implementations must inherit from this class
    and implement the abstract methods.
    
    Example:
        >>> tracker = SQLiteTracker()
        >>> tracker.start_run("my_experiment")
        >>> tracker.log_params({"n_iterations": 100})
        >>> tracker.log_metrics({"p-value": 0.032})
        >>> tracker.end_run()
    """
    
    @abstractmethod
    def start_run(self, 
                  run_name: str | None = None, 
                  tags: dict[str, str] | None = None) -> None:
        """
        Start a new experiment run.
        
        Args:
            run_name: Optional name for the run. If None, auto-generated.
            tags: Optional dictionary of tags for categorization.
        """
        pass
    
    @abstractmethod
    def end_run(self) -> None:
        """End the current experiment run."""
        pass
    
    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log experiment parameters.
        
        Args:
            params: Dictionary of parameter name-value pairs.
        """
        pass
    
    @abstractmethod
    def log_metrics(self, 
                    metrics: dict[str, float], 
                    step: int | None = None) -> None:
        """
        Log experiment metrics.
        
        Args:
            metrics: Dictionary of metric name-value pairs.
            step: Optional step number for iterative metrics.
        """
        pass
    
    @abstractmethod
    def log_artifact(self, 
                     file_path: str, 
                     artifact_path: str | None = None) -> None:
        """
        Log a file artifact.
        
        Args:
            file_path: Path to the file to log.
            artifact_path: Optional path within artifact storage.
        """
        pass
    
    @abstractmethod
    def log_dataset(self, 
                    dataset: "Dataset",  # type: ignore
                    name: str) -> None:
        """
        Log a HypEx Dataset as an artifact.
        
        Args:
            dataset: HypEx Dataset object to log.
            name: Name for the dataset artifact.
        """
        pass
    
    @abstractmethod
    def log_error(self, error: Exception) -> None:
        """
        Log an error that occurred during experiment.
        
        Args:
            error: Exception object to log.
        """
        pass
    
    def log_system_info(self) -> None:
        """
        Log system information (git commit, versions, etc.).
        Optional method with default implementation.
        """
        pass