"""
JSON-based experiment tracker.
Simple file-based tracking without external dependencies.
"""
from __future__ import annotations

import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseTracker
from .json_encoder import CustomJSONEncoder



class JsonTracker(BaseTracker):
    """
    JSON file-based experiment tracker.
    
    Each experiment run is saved as a separate JSON file
    with timestamp in the filename.
    
    Example:
        >>> tracker = JsonTracker(log_dir="./experiments")
        >>> tracker.start_run("ab_test_v1")
        >>> tracker.log_params({"alpha": 0.05})
        >>> tracker.log_metrics({"p-value": 0.032})
        >>> tracker.end_run()
    """
    
    def __init__(self, log_dir: str = "./hypex_logs"):
        """
        Initialize JSON tracker.
        
        Args:
            log_dir: Directory to store experiment logs.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_data: dict[str, Any] = {}
        self.run_name: str | None = None
    
    def start_run(self, 
                  run_name: str | None = None, 
                  tags: dict[str, str] | None = None
    ) -> None:
        """Start a new experiment run."""
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_run_data = {
            "run_name": self.run_name,
            "start_time": datetime.now().isoformat(),
            "tags": tags or {},
            "params": {},
            "metrics": {},
            "artifacts": [],
            "errors": [],
            "system_info": {}
        }
    
    def end_run(self) -> None:
        """End the current run and save to file."""
        self.current_run_data["end_time"] = datetime.now().isoformat()
        self.current_run_data["status"] = "completed"
        
        file_path = self.log_dir / f"{self.run_name}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(
                self.current_run_data, 
                f, 
                indent=2, 
                ensure_ascii=False, 
                default=lambda o: str(o),
                cls=CustomJSONEncoder 
            )
        print(f"💾 Experiment log saved to: {file_path}")
    
    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters."""
        self.current_run_data["params"].update(params)
    
    def log_metrics(self, 
                    metrics: dict[str, float], 
                    step: int | None = None
    ) -> None:
        """Log experiment metrics."""
        if step is not None:
            metrics = {f"{k}_step{step}": v for k, v in metrics.items()}
        self.current_run_data["metrics"].update(metrics)
    
    def log_artifact(self, 
                     file_path: str, 
                     artifact_path: str | None = None) -> None:
        """Log a file artifact."""
        self.current_run_data["artifacts"].append({
            "path": str(file_path),
            "artifact_path": artifact_path,
            "logged_at": datetime.now().isoformat()
        })
    
    def log_dataset(self, 
                    dataset: "Dataset",  # type: ignore
                    name: str) -> None:
        """Log a HypEx Dataset as JSON artifact."""
        run_dir = self.log_dir / self.run_name
        run_dir.mkdir(exist_ok=True)
        
        file_path = run_dir / f"{name}.json"
        dataset.to_json(str(file_path))
        self.log_artifact(str(file_path), f"datasets/{name}")
    
    def log_error(self, error: Exception) -> None:
        """Log an error."""
        self.current_run_data["errors"].append({
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        })
        self.current_run_data["status"] = "failed"
    
    def log_system_info(self) -> None:
        """Log system information."""
        info = {
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to get git commit
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            info["git_commit"] = commit
        except Exception:
            info["git_commit"] = "unknown"
        
        # Try to get HypEx version
        try:
            from hypex import __version__
            info["hypex_version"] = __version__
        except Exception:
            info["hypex_version"] = "unknown"
        
        self.current_run_data["system_info"] = info