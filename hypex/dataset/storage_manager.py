"""Compatibility imports for Spark cache management.

Implementation lives in the Spark backend; use Dataset.persist/unpersist in
backend-independent code.
"""

from .backends.spark_backend import PersistEntry, StorageManager

__all__ = ["PersistEntry", "StorageManager"]
