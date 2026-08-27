"""
Profiling utilities for HypEx library.

This module provides decorators and utilities for measuring execution time
of methods and functions, particularly useful for identifying performance
bottlenecks in Spark-based operations.
"""
from __future__ import annotations

import os
import time
import functools
import logging
from typing import Callable, Any

# Configure logger for profiling output
logger = logging.getLogger(__name__)

# Global flag to enable/disable profiling
# Can be controlled via environment variable HYPEX_PROFILING_ENABLED
_PROFILING_ENABLED = os.getenv("HYPEX_PROFILING_ENABLED", "false").lower() == "true"

# Thresholds for performance warnings (in seconds)
SLOW_THRESHOLD = 300.0
WARN_THRESHOLD = 100.0


def timeit(
    level: str = "INFO",
    prefix: str = "",
    enabled: bool | None = None,
    log_to_console: bool = True,
    log_to_logger: bool = False,
) -> Callable:
    """
    Decorator for profiling method execution time.
    
    Measures the execution time of decorated functions/methods and outputs
    timing information with performance indicators. Useful for identifying
    slow operations in data processing pipelines, especially in distributed
    computing environments like Apache Spark.
    
    Parameters
    ----------
    level : str, optional
        Logging level indicator (e.g., "INFO", "SPARK", "AGG", "SPLIT").
        Used for categorizing and filtering profiling output.
        Default is "INFO".
    
    prefix : str, optional
        Additional prefix string for grouping related operations.
        Useful for visual organization of profiling output.
        Default is "".
    
    enabled : bool, optional
        Explicit override for enabling/disabling this specific decorator.
        If None, uses the global _PROFILING_ENABLED flag.
        Default is None.
    
    log_to_console : bool, optional
        If True, prints timing information to stdout.
        Default is True.
    
    log_to_logger : bool, optional
        If True, logs timing information using Python's logging module.
        Default is False.
    
    Returns
    -------
    Callable
        Decorated function that measures and reports execution time.
    
    Notes
    -----
    Performance indicators:
    - "[SLOW]": Execution time > 10 seconds
    - "[WARN]": Execution time > 1 second
    - "[OK]": Execution time <= 1 second
    - "[FAIL]": Execution failed with exception
    
    The decorator automatically detects if it's applied to a method
    (by checking if the first argument is an instance) and includes
    the class name in the output.
    
    Examples
    --------
    Basic usage with default settings:
    
    >>> @timeit()
    ... def my_function():
    ...     time.sleep(1)
    ...     return "done"
    [OK] [INFO] my_function: 1.0023s
    
    With custom level and prefix:
    
    >>> @timeit(level="SPARK", prefix="COMPUTE")
    ... def spark_operation():
    ...     # Some Spark operation
    ...     pass
    [OK] COMPUTE[SPARK] spark_operation: 0.5432s
    
    Disable profiling for production:
    
    >>> @timeit(enabled=False)
    ... def production_function():
    ...     # No timing output
    ...     pass
    
    Control via environment variable:
    
    >>> # Set HYPEX_PROFILING_ENABLED=true in environment
    >>> @timeit()
    ... def conditional_function():
    ...     # Will be profiled if env var is set
    ...     pass
    
    Profile class methods:
    
    >>> class DataProcessor:
    ...     @timeit(level="PROCESS")
    ...     def transform(self, data):
    ...         # Processing logic
    ...         return data
    [OK] [PROCESS] DataProcessor.transform: 2.3456s
    
    Enable logging instead of console output:
    
    >>> @timeit(log_to_console=False, log_to_logger=True)
    ... def logged_function():
    ...     pass
    # Output goes to logger instead of stdout
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if profiling is enabled
            should_profile = enabled if enabled is not None else _PROFILING_ENABLED
            if not should_profile:
                return func(*args, **kwargs)
            
            # Determine function/method name
            class_name = ""
            if args and hasattr(args[0], '__class__'):
                class_name = f"{args[0].__class__.__name__}."
            
            full_name = f"{prefix}[{level}] {class_name}{func.__name__}"
            
            # Measure execution time
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                # Determine performance indicator
                if elapsed > SLOW_THRESHOLD:
                    marker = "[SLOW]"
                elif elapsed > WARN_THRESHOLD:
                    marker = "[WARN]"
                else:
                    marker = "[OK]"
                
                message = f"{marker} {full_name}: {elapsed:.4f}s"
                
                # Output timing information
                if log_to_console:
                    print(message)
                
                if log_to_logger:
                    if elapsed > SLOW_THRESHOLD:
                        logger.warning(message)
                    elif elapsed > WARN_THRESHOLD:
                        logger.info(message)
                    else:
                        logger.debug(message)
                
                return result
                
            except Exception as e:
                elapsed = time.perf_counter() - start
                error_message = f"[FAIL] {full_name}: FAILED after {elapsed:.4f}s - {type(e).__name__}: {e}"
                
                if log_to_console:
                    print(error_message)
                
                if log_to_logger:
                    logger.error(error_message)
                
                raise
        
        return wrapper
    return decorator


def enable_profiling():
    """
    Enable profiling globally.
    
    Sets the global _PROFILING_ENABLED flag to True, enabling all
    @timeit decorators that don't have explicit enabled=False parameter.
    
    Examples
    --------
    >>> from hypex.utils.profiling import enable_profiling
    >>> enable_profiling()
    >>> # All @timeit() decorators will now output timing information
    """
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = True


def disable_profiling():
    """
    Disable profiling globally.
    
    Sets the global _PROFILING_ENABLED flag to False, disabling all
    @timeit decorators that don't have explicit enabled=True parameter.
    
    This is useful for production environments where profiling overhead
    should be eliminated.
    
    Examples
    --------
    >>> from hypex.utils.profiling import disable_profiling
    >>> disable_profiling()
    >>> # All @timeit() decorators will now be silent (unless enabled=True)
    """
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = False


def is_profiling_enabled() -> bool:
    """
    Check if profiling is currently enabled.
    
    Returns
    -------
    bool
        True if profiling is enabled, False otherwise.
    
    Examples
    --------
    >>> from hypex.utils.profiling import is_profiling_enabled
    >>> if is_profiling_enabled():
    ...     print("Profiling is active")
    """
    return _PROFILING_ENABLED


class ProfilingContext:
    """
    Context manager for temporarily enabling/disabling profiling.
    
    Useful for enabling profiling only for specific code sections
    without affecting the global state.
    
    Examples
    --------
    >>> from hypex.utils.profiling import ProfilingContext
    >>> with ProfilingContext(enabled=True):
    ...     # Profiling is enabled in this block
    ...     slow_operation()
    >>> # Profiling returns to previous state
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize profiling context.
        
        Parameters
        ----------
        enabled : bool, optional
            Whether to enable profiling in this context.
            Default is True.
        """
        self.enabled = enabled
        self.previous_state = None
    
    def __enter__(self):
        """Enter the context, saving previous profiling state."""
        global _PROFILING_ENABLED
        self.previous_state = _PROFILING_ENABLED
        _PROFILING_ENABLED = self.enabled
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, restoring previous profiling state."""
        global _PROFILING_ENABLED
        _PROFILING_ENABLED = self.previous_state
        return False