from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Any

from ..comparators.abstract import StatsComparator, StatsHypothesisTesting
from ..comparators.comparators import StatTestMasterAbstract
from ..dataset import (
    ABCRole,
    AdditionalTargetRole,
    Dataset,
    ExperimentData,
    TempTargetRole,
)
from ..executor import Executor
from ..utils import BackendsEnum, ExperimentDataEnum, timeit
from ..utils import ExperimentDataEnum, HypExLogger
from ..utils.registry import backend_factory


class Experiment(Executor):
    """Base pipeline that sequentially executes a list of executors on experiment data.

    An ``Experiment`` is the fundamental building block of HypEx pipelines.
    It wraps a sequence of :class:`~hypex.executor.Executor` instances and
    runs them one after another, threading the same :class:`ExperimentData`
    container through each step.

    If any executor in the pipeline is a transformer (i.e. modifies the
    underlying dataset), the input data is deep-copied before execution so
    that the caller's original data remains untouched.

    Attributes:
        executors: The ordered sequence of executors to run.
        transformer: Whether the pipeline mutates the dataset. When ``True``,
            ``execute`` deep-copies the input ``ExperimentData`` before
            running the executors.
    """
    def _detect_transformer(self) -> bool:
        """Detect whether any executor in the pipeline is a transformer.

        Returns:
            ``True`` if at least one executor has ``_is_transformer`` set to
            ``True``, ``False`` otherwise.
        """
        return any(executor._is_transformer for executor in self.executors)

    def get_executor_ids(
        self, searched_classes: type | Iterable[type] | None = None
    ) -> dict[type, list[str]]:
        """Collect executor IDs filtered by class type.

        Args:
            searched_classes: A single class or an iterable of classes to
                match against. If ``None`` or empty, returns an empty dict.

        Returns:
            A dictionary mapping each searched class to the list of executor
            IDs in the pipeline that are instances of that class.
        """
        if not searched_classes:
            return {}

        searched_classes = (
            searched_classes
            if isinstance(searched_classes, Iterable)
            else [searched_classes]
        )
        return {
            searched_class: [
                executor.id
                for executor in self.executors
                if isinstance(executor, searched_class)
            ]
            for searched_class in searched_classes
        }

    def __init__(
        self,
        executors: Sequence[Executor],
        transformer: bool | None = None,
        key: Any = "",
    ):
        """Initialize the experiment pipeline.

        Args:
            executors: Ordered sequence of executors to run sequentially.
            transformer: Explicit flag indicating whether the pipeline
                mutates the dataset. If ``None``, the flag is inferred by
                scanning ``executors`` for any transformer.
            key: Optional identifier key forwarded to the base
                :class:`~hypex.executor.Executor`.
        """
        self.executors: Sequence[Executor] = executors
        self.transformer: bool = (
            transformer if transformer is not None else self._detect_transformer()
        )
        super().__init__(key)

        # Создаем логгер для эксперимента
        self.logger = HypExLogger(
            name="hypex.experiment",
            level="INFO",
            log_file="experiment.log",
            console_out=False
        )

    def set_params(self, params: dict[str, Any] | dict[type, dict[str, Any]]) -> None:
        """Propagate parameters to the executors in the pipeline.

        Supports two parameter formats:

        * ``{str: dict}`` — parameters keyed by executor attribute name,
          forwarded to the base :class:`~hypex.executor.Executor.set_params`.
        * ``{type: dict}`` — parameters keyed by executor class. Each
          executor in the pipeline that is an instance of the given class
          receives the corresponding parameter dict.

        Args:
            params: Parameter mapping in one of the two supported formats.

        Raises:
            ValueError: If ``params`` is neither ``{str: dict}`` nor
                ``{type: dict}``.
        """
        if isinstance(next(iter(params)), str):
            super().set_params(params)
        elif isinstance(next(iter(params)), type):
            for executor in self.executors:
                executor.set_params(params)
        else:
            raise ValueError(
                "params must be a dict of str to dict or a dict of class to dict"
            )

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        """Store a value in the experiment data's analysis tables.

        Args:
            data: The experiment data container to update.
            value: The value to store.
            key: Optional sub-key (unused here, kept for API compatibility).

        Returns:
            The updated experiment data container.
        """
        return data.set_value(ExperimentDataEnum.analysis_tables, self.id, value)

    @staticmethod
    def _get_executor_backend(executor: Executor, ds: Dataset):
        """
        Class for selecting backend-dependent realization for direct executor
        """
        executor_cls = type(executor)
        backend_cls = backend_factory.resolve_backend(executor_cls, ds)
        if backend_cls is None:
             return executor

        sig = inspect.signature(backend_cls.__init__)
        expected_params = {p.name for p in sig.parameters.values() if p.name != 'self'}

        init_kwargs = {k: getattr(executor, k) for k in expected_params if hasattr(executor, k)}

        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_keyword and hasattr(executor, 'calc_kwargs'):
            init_kwargs['calc_kwargs'] = executor.calc_kwargs

        new_executor = backend_cls(**init_kwargs)

        if hasattr(executor, 'key'):
            new_executor.key = executor.key

        return new_executor


    @timeit(level="PIPELINE", prefix="EXPERIMENT")
    def execute(self, data: ExperimentData) -> ExperimentData:
        """Run the full executor pipeline on the given experiment data.

        Each executor is invoked in order. The executor's ``key`` is
        synchronized with the experiment's key before each call. If the
        pipeline is flagged as a transformer, the input data is deep-copied
        first to preserve the caller's original dataset.

        Args:
            data: The experiment data container to process.

        Returns:
            The experiment data after all executors have been applied.
        """
        # Логируем информацию о Spark-сессии
        self.logger.log_spark_info()

        experiment_data = deepcopy(data) if self.transformer else data
        for executor in self.executors:
            with self.logger.process(
                name=executor.__class__.__name__,
                backend=experiment_data.ds.backend_type.value,
                log_spark=False  # можно включить для детального логирования Spark-процессов
            ):
                cur_executor = self._get_executor_backend(executor, experiment_data.ds)
                cur_executor.key = self.key
                experiment_data = cur_executor.execute(experiment_data)
        return experiment_data


class OnRoleExperiment(Experiment):
    """Experiment that runs its executors once per target role/column.

    ``OnRoleExperiment`` iterates over every column matching the configured
    ``role`` (e.g. :class:`~hypex.dataset.TargetRole`) and executes the
    pipeline for each column individually.

    Executors are split into two groups for performance:

    * **Vector executors** (subclasses of :class:`StatsComparator`,
      :class:`StatsHypothesisTesting`, or :class:`AdaptiveHypothesisTest`)
      are capable of processing all target columns in a single pass. They
      are executed once with ``tmp_roles`` set to all target columns.
    * **Iterative executors** (everything else, e.g.
      :class:`GroupDifference`) operate on a single column at a time and
      are executed in a loop, once per target column.

    This split minimizes the number of Spark jobs while preserving
    correctness for executors that expect a single-column input.
    """
    def __init__(
        self,
        executors: list[Executor],
        role: ABCRole | Sequence[ABCRole],
        transformer: bool | None = None,
        key: Any = "",
    ):
        """Initialize the per-role experiment.

        Args:
            executors: List of executors to run for each target column.
            role: A single role or a sequence of roles identifying the
                target columns to iterate over.
            transformer: Explicit transformer flag forwarded to the base
                :class:`Experiment`. If ``None``, inferred automatically.
            key: Optional identifier key forwarded to the base
                :class:`Experiment`.
        """
        self.role: list[ABCRole] = [role] if isinstance(role, ABCRole) else list(role)
        super().__init__(executors, transformer, key)

    @timeit(level="PIPELINE", prefix="ON_ROLE")
    def execute(self, data: ExperimentData) -> ExperimentData:
        """Execute the pipeline for every column matching the configured role.

        This method iterates over all columns in the dataset that match the
        roles specified in ``self.role`` (e.g., :class:`~hypex.dataset.TargetRole`).
        To optimize performance, especially on Spark backends, executors are
        split into two groups:

        1. **Vector executors**: Subclasses of :class:`StatsComparator` or
           :class:`StatsHypothesisTesting`. These can process all target columns
           in a single pass by setting ``tmp_roles`` for all targets simultaneously.

        2. **Iterative executors**: All other executors (including
           :class:`AdaptiveHypothesisTest` like TTest/KSTest on Pandas backend,
           which delegate to non-vectorized group tests). These are executed
           in a loop, once per target column, to ensure compatibility with
           single-column input requirements.

        The method ensures that ``tmp_roles`` are correctly set and cleared
        for each execution phase to isolate column processing.

        Args:
            data: The experiment data container to process.

        Returns:
            The experiment data after all per-role executions have completed.
            If no columns match the specified roles, the input data is returned
            unchanged.

        Example:
            .. code-block:: python

                # Execute TTest and GroupDifference for all TargetRole columns
                exp = OnRoleExperiment(
                    executors=[TTest(), GroupDifference()],
                    role=TargetRole()
                )
                result_data = exp.execute(experiment_data)
        """
        target_fields = data.field_search(self.role)
        if not target_fields:
            return data


        vector_executors = []
        iterative_executors = []
        _backend = data.ds.backend_type

        for ex in self.executors:
            # StatsComparator / StatsHypothesisTesting — истинно векторные
            if isinstance(ex, (StatsComparator, StatsHypothesisTesting)):
                vector_executors.append(ex)
            # Мастер-классы (TTest, KSTest, Chi2Test…):
            #   Spark  → Stats* (векторный)
            #   Pandas → Group* (итеративный)
            elif isinstance(ex, StatTestMasterAbstract):
                if _backend == BackendsEnum.spark:
                    vector_executors.append(ex)
                else:
                    iterative_executors.append(ex)
            else:
                iterative_executors.append(ex)

        original_executors = self.executors

        # Vector executors execution (only true StatsComparators)
        if vector_executors:
            tmp_roles_dict = {}
            for field in target_fields:
                if field in data.ds.columns:
                    tmp_roles_dict[field] = TempTargetRole()
                elif data.additional_fields and field in data.additional_fields.columns:
                    tmp_roles_dict[field] = AdditionalTargetRole()

            if tmp_roles_dict:
                data.ds.tmp_roles = tmp_roles_dict
                self.executors = vector_executors
                data = super().execute(data)
                data.ds.tmp_roles = {}

        # Iterative executors execution (one by one)
        if iterative_executors:
            self.executors = iterative_executors
            for field in target_fields:
                if field in data.ds.columns:
                    data.ds.tmp_roles = {field: TempTargetRole()}
                elif data.additional_fields and field in data.additional_fields.columns:
                    data.additional_fields.tmp_roles = {field: AdditionalTargetRole()}

                data = super().execute(data)

                data.ds.tmp_roles = {}
                if data.additional_fields:
                    data.additional_fields.tmp_roles = {}

        self.executors = original_executors
        return data
