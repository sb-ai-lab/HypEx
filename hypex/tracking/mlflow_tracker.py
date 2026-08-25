"""
MLflow-based experiment tracker.
Production-ready tracking with full MLflow ecosystem integration.
"""
from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTracker

# Настройка логирования
logger = logging.getLogger(__name__)


class MLflowTracker(BaseTracker):
    """
    MLflow experiment tracker for HypEx.
    
    Provides full integration with MLflow ecosystem including:
    - Experiment organization
    - Run comparison
    - Artifact storage
    - Model registry integration
    - Distributed tracking server support
    
    Metrics are logged with step support for consistent tracking
    across all HypEx trackers.
    
    Example:
        >>> tracker = MLflowTracker(experiment_name="Production_AB_Tests")
        >>> tracker.start_run("ab_test_v1")
        >>> tracker.log_params({"alpha": 0.05})
        >>> tracker.log_metrics({"p-value": 0.032})
        >>> tracker.end_run()
    """
    
    def __init__(
        self,
        experiment_name: str = "HypEx",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: URI of MLflow tracking server.
                         If None, uses local file system or MLFLOW_TRACKING_URI env.
            artifact_location: Location for artifact storage.
        
        Raises:
            ImportError: If mlflow is not installed.
        """
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "MLflow is not installed. Install it with: pip install mlflow"
            ) from e
        
        self.mlflow = mlflow
        
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        
        # Set or create experiment
        experiment = self.mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = self.mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
        
        self.mlflow.set_experiment(experiment_name)
        self.run_id: Optional[str] = None
        self._run_started: bool = False
    
    def _ensure_run_started(self, method_name: str) -> None:
        """Helper to check if run was started before logging."""
        if not self._run_started:
            raise RuntimeError(
                f"Call start_run() before {method_name}(). "
                f"Current run_id: {self.run_id}"
            )
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Start a new MLflow run."""
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run = self.mlflow.start_run(
            run_name=run_name,
            tags=tags
        )
        self.run_id = self.run.info.run_id
        self._run_started = True
        logger.info("Started MLflow run: %s", self.run_id)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        self._ensure_run_started("end_run")
        
        self.mlflow.end_run()
        logger.info("✅ MLflow run completed: %s", self.run_id)
        self._run_started = False
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log experiment parameters to MLflow.
        
        Note: MLflow has param value length limit (250 chars).
        Long values will be truncated with a warning.
        
        Args:
            params: Dictionary of parameter name-value pairs.
        """
        self._ensure_run_started("log_params")
        
        for key, value in params.items():
            str_value = str(value)
            if len(str_value) > 250:
                logger.warning(
                    "Param '%s' value exceeds 250 chars (%d), truncating",
                    key, len(str_value)
                )
                str_value = str_value[:247] + "..."
            try:
                self.mlflow.log_param(key, str_value)
            except Exception as e:
                logger.warning("Could not log param '%s': %s", key, e)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log experiment metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric name-value pairs.
            step: Optional step number for iterative metrics.
        """
        self._ensure_run_started("log_metrics")
        
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                logger.warning(
                    "Metric '%s' has non-numeric value %r (type: %s), skipping",
                    key, value, type(value).__name__
                )
                continue
            
            try:
                if step is not None:
                    self.mlflow.log_metric(key, float(value), step=step)
                else:
                    self.mlflow.log_metric(key, float(value))
            except Exception as e:
                logger.warning("Could not log metric '%s': %s", key, e)
    
    def log_artifact(
        self,
        file_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """
        Log a file artifact to MLflow.
        
        Args:
            file_path: Path to the file to log.
            artifact_path: Optional path within artifact storage.
        """
        self._ensure_run_started("log_artifact")
        
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.warning("Artifact file not found: %s", file_path)
            return
        
        try:
            self.mlflow.log_artifact(file_path, artifact_path)
            logger.debug("Logged artifact: %s", file_path)
        except Exception as e:
            logger.error("Could not log artifact '%s': %s", file_path, e)
            raise
    
    def log_dataset(
        self,
        dataset: "Dataset",  # type: ignore
        name: str
    ) -> None:
        """
        Log a HypEx Dataset as Parquet artifact.
        
        Args:
            dataset: HypEx Dataset object to log.
            name: Name for the dataset artifact.
        """
        self._ensure_run_started("log_dataset")
        
        temp_path = None
        try:
            # Create temp file
            temp_dir = Path(tempfile.mkdtemp(prefix="mlflow_artifacts_"))
            temp_path = temp_dir / f"{name}.parquet"
            
            # Convert and save
            if hasattr(dataset.data, 'to_pandas'):
                df = dataset.data.to_pandas()
            else:
                df = dataset.data
            
            df.to_parquet(temp_path, index=False)
            
            # Log to MLflow
            self.log_artifact(str(temp_path), f"datasets/{name}")
            logger.info("Logged dataset '%s' as artifact", name)
            
        except Exception as e:
            logger.error("Could not log dataset '%s': %s", name, e)
            raise
        finally:
            # Cleanup temp file
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    temp_path.parent.rmdir()
                except Exception as e:
                    logger.debug("Could not clean up temp file: %s", e)
    
    def log_error(self, error: Exception) -> None:
        """
        Log an error to MLflow.
        
        Args:
            error: Exception object to log.
        """
        # Don't require start_run() to allow logging initialization errors
        if not self._run_started:
            logger.warning("Cannot log error: no active run")
            return
        
        self.mlflow.set_tag("status", "FAILED")
        self.mlflow.set_tag("error_type", type(error).__name__)
        self.mlflow.set_tag("error_message", str(error))
        logger.warning("Logged error for run %s: %s", self.run_id, error)
    
    def log_system_info(self) -> None:
        """Log system information to MLflow."""
        if not self._run_started:
            return
        
        # Python version
        self.mlflow.set_tag("python_version", sys.version)
        
        # Git commit
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            self.mlflow.set_tag("git_commit", commit)
        except Exception as e:
            logger.debug("Could not get git commit: %s", e)
            self.mlflow.set_tag("git_commit", "unknown")
        
        # HypEx version
        try:
            from hypex import __version__
            self.mlflow.set_tag("hypex_version", __version__)
        except ImportError:
            self.mlflow.set_tag("hypex_version", "unknown")
        
        # Hostname
        self.mlflow.set_tag("hostname", socket.gethostname())
        
        # Timestamp
        self.mlflow.set_tag("start_timestamp", datetime.now().isoformat())
    
    # ==================== Query Helpers ====================
    
    def get_run_history(
        self,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get experiment run history from MLflow.
        
        Args:
            limit: Maximum number of runs to return.
            status: Filter by status ('FINISHED', 'FAILED', etc.).
            
        Returns:
            List of run dictionaries.
        """
        runs = self.mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string=f"tags.status = '{status}'" if status else None,
            max_results=limit,
            order_by=["start_time DESC"]
        )
        return runs.to_dict('records') if runs is not None and len(runs) > 0 else []
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare metrics across multiple runs.
        
        Args:
            run_ids: List of run IDs to compare.
            
        Returns:
            Dictionary with run_name -> {metrics, params, tags} mapping.
        """
        comparison = {}
        for run_id in run_ids:
            try:
                run = self.mlflow.get_run(run_id)
                comparison[run.info.run_name or run_id] = {
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
            except Exception as e:
                logger.warning("Could not get run '%s': %s", run_id, e)
        return comparison
    
    def get_best_run(
        self,
        metric_name: str = "p-value",
        mode: str = "min"
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize.
            mode: 'min' or 'max'.
            
        Returns:
            Best run information or None.
        """
        runs = self.mlflow.search_runs(
            experiment_names=[self.experiment_name],
            order_by=[f"metrics.{metric_name} {'ASC' if mode == 'min' else 'DESC'}"],
            max_results=1
        )
        
        if runs is not None and len(runs) > 0:
            return runs.iloc[0].to_dict()
        return None
    
    def get_metrics(
        self,
        run_id: Optional[str] = None,
        key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query logged metrics with optional filters.
        
        Args:
            run_id: Filter by run ID (defaults to current run).
            key: Filter by metric name.
            
        Returns:
            List of metric records.
        """
        run_id = run_id or self.run_id
        if run_id is None:
            return []
        
        try:
            run = self.mlflow.get_run(run_id)
            metrics = run.data.metrics
            
            if key:
                if key in metrics:
                    return [{"key": key, "value": metrics[key], "step": None}]
                return []
            
            return [{"key": k, "value": v, "step": None} for k, v in metrics.items()]
        except Exception as e:
            logger.warning("Could not get metrics for run '%s': %s", run_id, e)
            return []
    
    def get_metric_values(self, key: str, run_id: Optional[str] = None) -> list[tuple[int | None, float]]:
        """
        Get (step, value) pairs for a specific metric.
        
        Note: MLflow's Python API doesn't expose step history easily.
        This returns the latest value with step=None.
        
        Returns:
            List with single (step, value) tuple.
        """
        metrics = self.get_metrics(run_id=run_id, key=key)
        if metrics:
            return [(None, metrics[0]["value"])]
        return []
    
    def get_latest_metric(self, key: str, run_id: Optional[str] = None) -> float | None:
        """
        Get the most recent value for a metric.
        
        Returns:
            Latest value or None if metric not found.
        """
        values = self.get_metric_values(key, run_id=run_id)
        return values[-1][1] if values else None