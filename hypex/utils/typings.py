import typing as tp
from hypex.pipelines.base import BaseExecutor

ExecutorSequence = tp.Union[BaseExecutor, tp.Sequence[BaseExecutor]]


