"""
SQLite-based experiment tracker.
Local database tracking with query capabilities.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseTracker
from .json_encoder import CustomJSONEncoder


logger = logging.getLogger(__name__)


class SQLiteTracker(BaseTracker):
    """
    SQLite database experiment tracker.
    
    All experiment runs are stored in a single SQLite database file,
    enabling easy querying and comparison of experiments.
    
    Metrics are stored in a separate table with step support for
    consistent querying across all trackers.
    
    Example:
        >>> tracker = SQLiteTracker(db_path="./hypex_history.db")
        >>> tracker.start_run("ab_test_v1")
        >>> tracker.log_params({"alpha": 0.05})
        >>> tracker.log_metrics({"p-value": 0.032})
        >>> tracker.end_run()
        
        # Query experiments later
        >>> history = tracker.get_run_history(limit=10)
    """
    
    def __init__(self, db_path: str = "./hypex_history.db"):
        """
        Initialize SQLite tracker.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id: int | None = None
        self.run_name: str | None = None
        self._run_started: bool = False
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            # Runs table
            c.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name TEXT UNIQUE,
                    start_time TEXT,
                    end_time TEXT,
                    status TEXT DEFAULT 'RUNNING',
                    params TEXT,
                    tags TEXT,
                    system_info TEXT,
                    error_message TEXT
                )
            ''')
            
            # Metrics table (consistent format with JsonTracker)
            c.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    key TEXT,
                    value REAL,
                    step INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            ''')
            
            # Artifacts table
            c.execute('''
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    name TEXT,
                    path TEXT,
                    artifact_path TEXT,
                    timestamp TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            ''')
            
            # Datasets table
            c.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    name TEXT,
                    path TEXT,
                    timestamp TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            ''')
            
            # Create indexes for faster queries
            c.execute('CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(key)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(run_name)')
            
            conn.commit()
    
    def _ensure_run_started(self, method_name: str) -> None:
        """Helper to check if run was started before logging."""
        if not self._run_started:
            raise RuntimeError(
                f"Call start_run() before {method_name}(). "
                f"Current run_id: {self.run_id}"
            )
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def start_run(self, 
                  run_name: str | None = None, 
                  tags: dict[str, str] | None = None
    ) -> None:
        """Start a new experiment run."""
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._get_connection() as conn:
            c = conn.cursor()
            
            try:
                c.execute('''
                    INSERT INTO runs (run_name, start_time, status, params, tags)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    self.run_name,
                    datetime.now().isoformat(),
                    'RUNNING',
                    '{}',
                    json.dumps(tags or {}, cls=CustomJSONEncoder)
                ))
                self.run_id = c.lastrowid
                conn.commit()
                self._run_started = True
            except sqlite3.IntegrityError:
                # Run name already exists, add timestamp suffix
                self.run_name = f"{self.run_name}_{datetime.now().strftime('%H%M%S%f')}"
                c.execute('''
                    INSERT INTO runs (run_name, start_time, status, params, tags)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    self.run_name,
                    datetime.now().isoformat(),
                    'RUNNING',
                    '{}',
                    json.dumps(tags or {}, cls=CustomJSONEncoder)
                ))
                self.run_id = c.lastrowid
                conn.commit()
                self._run_started = True
    
    def end_run(self) -> None:
        """End the current run."""
        self._ensure_run_started("end_run")
        
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute('''
                UPDATE runs 
                SET status = ?, end_time = ?
                WHERE id = ?
            ''', ('COMPLETED', datetime.now().isoformat(), self.run_id))
            conn.commit()
        
        logger.info("Experiment logged to SQLite: %s (ID: %s)", self.db_path, self.run_id)
        self._run_started = False
    
    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters."""
        self._ensure_run_started("log_params")
        
        with self._get_connection() as conn:
            c = conn.cursor()
            
            # Get current params
            c.execute("SELECT params FROM runs WHERE id=?", (self.run_id,))
            row = c.fetchone()
            current_params = json.loads(row[0] or '{}') if row else {}
            
            # Update params
            current_params.update(params)
            
            c.execute(
                "UPDATE runs SET params=? WHERE id=?",
                (json.dumps(current_params, cls=CustomJSONEncoder), self.run_id)
            )
            conn.commit()
    
    def log_metrics(self, 
                    metrics: dict[str, float], 
                    step: int | None = None) -> None:
        """
        Log experiment metrics.
        
        Metrics are stored as separate rows to support multiple values
        per key with different steps (consistent with JsonTracker).
        
        Args:
            metrics: Dictionary of metric name-value pairs.
            step: Optional step number for iterative metrics.
        """
        self._ensure_run_started("log_metrics")
        timestamp = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            c = conn.cursor()
            
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    logger.warning(
                        "Metric '%s' has non-numeric value %r (type: %s), skipping",
                        key, value, type(value).__name__
                    )
                    continue
                    
                c.execute('''
                    INSERT INTO metrics (run_id, key, value, step, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    self.run_id, 
                    key, 
                    float(value), 
                    step,
                    timestamp
                ))
            
            conn.commit()
    
    def log_artifact(self, 
                     file_path: str, 
                     artifact_path: str | None = None) -> None:
        """Log a file artifact."""
        self._ensure_run_started("log_artifact")
        
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.warning("Artifact file not found: %s", file_path)
        
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO artifacts (run_id, name, path, artifact_path, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                self.run_id,
                path_obj.name,
                str(file_path),
                artifact_path,
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def log_dataset(self, 
                    dataset: "Dataset",  # type: ignore
                    name: str) -> None:
        """Log a HypEx Dataset as Parquet artifact."""
        self._ensure_run_started("log_dataset")
        
        # Create artifacts directory
        artifacts_dir = Path("./hypex_artifacts") / str(self.run_id)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        file_path = artifacts_dir / f"{name}.parquet"
        try:
            df = dataset.data.to_pandas() if hasattr(dataset.data, 'to_pandas') else dataset.data
            df.to_parquet(file_path, index=False)
            
            # Log artifact reference
            self.log_artifact(str(file_path), f"datasets/{name}")
            
            # Log dataset record
            with self._get_connection() as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO datasets (run_id, name, path, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (
                    self.run_id,
                    name,
                    str(file_path),
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error("Failed to log dataset '%s': %s", name, e)
            raise
    
    def log_error(self, error: Exception) -> None:
        """Log an error."""
        # Don't require start_run() to allow logging initialization errors
        if self.run_id is None:
            logger.warning("Cannot log error: no active run")
            return
        
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute('''
                UPDATE runs 
                SET status = ?, error_message = ?
                WHERE id = ?
            ''', ('FAILED', f"{type(error).__name__}: {str(error)}", self.run_id))
            conn.commit()
    
    def log_system_info(self) -> None:
        """Log system information."""
        if not self._run_started or self.run_id is None:
            return
        
        info = {
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to get git commit
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            info["git_commit"] = commit
        except Exception as e:
            logger.debug("Could not get git commit: %s", e)
            info["git_commit"] = "unknown"
        
        # Try to get HypEx version
        try:
            from hypex import __version__
            info["hypex_version"] = __version__
        except ImportError:
            info["hypex_version"] = "unknown"
        
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "UPDATE runs SET system_info=? WHERE id=?",
                (json.dumps(info, cls=CustomJSONEncoder), self.run_id)
            )
            conn.commit()
    
    # ==================== Query Helpers ====================
    
    def get_run_history(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        """
        Get experiment run history.
        
        Args:
            limit: Maximum number of runs to return.
            status: Filter by status ('COMPLETED', 'FAILED', 'RUNNING').
            
        Returns:
            List of run dictionaries.
        """
        with self._get_connection() as conn:
            c = conn.cursor()
            
            if status:
                c.execute('''
                    SELECT * FROM runs 
                    WHERE status = ?
                    ORDER BY start_time DESC 
                    LIMIT ?
                ''', (status, limit))
            else:
                c.execute('''
                    SELECT * FROM runs 
                    ORDER BY start_time DESC 
                    LIMIT ?
                ''', (limit,))
            
            runs = [dict(row) for row in c.fetchall()]
        
        # Parse JSON fields
        for run in runs:
            if run.get('params'):
                run['params'] = json.loads(run['params'])
            if run.get('tags'):
                run['tags'] = json.loads(run['tags'])
            if run.get('system_info'):
                run['system_info'] = json.loads(run['system_info'])
        
        return runs
    
    def get_metrics(
        self, 
        run_id: int | None = None,
        key: str | None = None, 
        step: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Query logged metrics with optional filters.
        
        Args:
            run_id: Filter by run ID (defaults to current run).
            key: Filter by metric name.
            step: Filter by step number.
            
        Returns:
            List of metric records matching the filters.
        """
        run_id = run_id or self.run_id
        if run_id is None:
            return []
        
        with self._get_connection() as conn:
            c = conn.cursor()
            
            query = "SELECT key, value, step, timestamp FROM metrics WHERE run_id = ?"
            params: list[Any] = [run_id]
            
            if key:
                query += " AND key = ?"
                params.append(key)
            if step is not None:
                query += " AND step = ?"
                params.append(step)
            
            query += " ORDER BY step, timestamp"
            
            c.execute(query, params)
            metrics = [dict(row) for row in c.fetchall()]
        
        return metrics
    
    def get_metric_values(self, key: str, run_id: int | None = None) -> list[tuple[int | None, float]]:
        """
        Get (step, value) pairs for a specific metric.
        
        Returns:
            List of (step, value) tuples, sorted by step.
        """
        metrics = self.get_metrics(run_id=run_id, key=key)
        values = [(m["step"], m["value"]) for m in metrics]
        return sorted(values, key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))
    
    def get_latest_metric(self, key: str, run_id: int | None = None) -> float | None:
        """
        Get the most recent value for a metric.
        
        Returns:
            Latest value or None if metric not found.
        """
        values = self.get_metric_values(key, run_id=run_id)
        return values[-1][1] if values else None
    
    def compare_runs(self, run_ids: list[int]) -> dict[str, dict[str, float]]:
        """
        Compare latest metrics across multiple runs.
        
        Args:
            run_ids: List of run IDs to compare.
            
        Returns:
            Dictionary with run_name -> {metric_key: latest_value} mapping.
        """
        if not run_ids:
            return {}
        
        placeholders = ','.join('?' * len(run_ids))
        
        with self._get_connection() as conn:
            c = conn.cursor()
            
            # Get latest metric per key per run
            c.execute(f'''
                SELECT r.run_name, m.key, m.value, m.timestamp
                FROM runs r
                JOIN metrics m ON r.id = m.run_id
                WHERE r.id IN ({placeholders})
                ORDER BY r.run_name, m.key, m.timestamp DESC
            ''', run_ids)
            
            comparison: dict[str, dict[str, float]] = {}
            seen: set[tuple[str, str]] = set()
            
            for row in c.fetchall():
                run_name, key, value, timestamp = row
                if (run_name, key) not in seen:
                    if run_name not in comparison:
                        comparison[run_name] = {}
                    comparison[run_name][key] = value
                    seen.add((run_name, key))
        
        return comparison
    
    def get_metrics_summary(self, run_id: int | None = None) -> dict[str, Any]:
        """
        Compute summary statistics for metrics.
        
        Args:
            run_id: Run ID to summarize (defaults to current run).
            
        Returns:
            Dictionary with metric_key -> {count, min, max, mean, last}
        """
        from statistics import mean
        
        run_id = run_id or self.run_id
        if run_id is None:
            return {}
        
        metrics = self.get_metrics(run_id=run_id)
        
        # Group by key
        by_key: dict[str, list[float]] = {}
        for m in metrics:
            key = m["key"]
            by_key.setdefault(key, []).append(m["value"])
        
        summary = {}
        for key, values in by_key.items():
            if not values:
                continue
            summary[key] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
                "last": values[-1]
            }
        
        return summary