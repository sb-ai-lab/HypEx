from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class Adapter:
    @staticmethod
    def to_list(data: Any) -> list:
        """Convert any iterable / array-like / scalar to a plain list.

        Handles:
        - ``None`` → ``[]``
        - ``str`` → ``[str]`` (avoids iterating over characters)
        - objects with ``.to_list()`` (pyspark.pandas Index/Series)
        - objects with ``.tolist()`` (pandas Index/Series, numpy arrays)
        - objects with ``.to_array()`` (legacy arrays)
        - ``Sequence`` subclasses → ``list(data)``
        - scalars (int, float, bool, np.integer, np.floating) → ``[data]``
        - anything else → ``[data]``
        """
        if data is None:
            return []
        if isinstance(data, str):
            return [data]
        # Scalars: numpy scalars (.tolist() returns a plain scalar, not list)
        if isinstance(data, (int, float, bool)):
            return [data]
        if hasattr(data, "to_list"):
            return data.to_list()
        if hasattr(data, "tolist"):
            result = data.tolist()
            # Guard: numpy scalar .tolist() returns non-list
            if not isinstance(result, (list, tuple)):
                return [result]
            return list(result)
        if hasattr(data, "to_array"):
            return data.to_array()
        return list(data) if isinstance(data, Sequence) else [data]

    @staticmethod
    def list_to_single(data: list) -> Any:
        if isinstance(data, list):
            if len(data) == 0:
                return None
            elif len(data) == 1:
                return data[0]
            else:
                raise ValueError("Only a list of a single item can be accepted")
        return None
