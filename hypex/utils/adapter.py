from typing import Any, List, Sequence


class Adapter:
    @staticmethod
    def to_list(data: Any) -> List:
        if data is None:
            return []
        if isinstance(data, str):
            return [data]
        return list(data) if isinstance(data, Sequence) else [data]
