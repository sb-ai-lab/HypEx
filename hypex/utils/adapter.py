from typing import Any, List, Sequence


class Adapter:
    @staticmethod
    def to_list(data: Any) -> List:
        if data is None:
            return []
        if isinstance(data, str):
            return [data]
        return list(data) if isinstance(data, Sequence) else [data]

    @staticmethod
    def list_to_single(data: List) -> Any:
        if isinstance(data, List):
            if len(data) == 0:
                return None
            elif len(data) == 1:
                return data[0]
            else:
                raise ValueError("Only a list of a single item can be accepted")
