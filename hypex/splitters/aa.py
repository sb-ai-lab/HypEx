from __future__ import annotations

from typing import Any

from ..dataset import (
    AdditionalTreatmentRole,
    Dataset,
    ExperimentData,
    StatisticRole,
    StratificationRole,
)
from ..dataset.roles import ConstGroupRole
from ..executor import Calculator
from ..utils import Adapter, BackendsEnum, ExperimentDataEnum, timeit

MISSING_CONST_LABELS = frozenset({"", "nan", "none", "nat", "<na>"})

# A value that cannot collide with a split label: rows carrying it in the
# tagged const column are the ones that take part in the random split.
_FREE_CONST_SENTINEL = "__hypex_free_const_group__"

# Safe column name for value_counts: prevents collision when the
# user-supplied const-group column is literally named "count".
_CONST_LABEL_COL = "__hypex_const_label__"


class AASplitter(Calculator):
    """Splits data into control/test groups with optional pinned const groups.

    When a ``ConstGroupRole`` column is present, rows with non-missing values
    in that column are **pinned** to their respective groups and do NOT
    participate in random splitting. Only rows with missing values (NaN, None,
    empty string, etc.) are split randomly.

    The effective ``control_size`` is adjusted so that the TOTAL control
    fraction (pinned + random) matches the requested value.

    Args:
        control_size: Desired fraction of data in the control group.
            Must be between 0 and 1. Defaults to 0.5.
        random_state: Seed for reproducibility. Defaults to None.
        sample_size: Fraction of data to sample. Defaults to None (full data).
        constant_key: Whether to keep the key constant across iterations.
            Defaults to True.
        save_groups: Whether to store group subsets in ``ExperimentData.groups``.
            Defaults to True.
        groups_sizes: Custom group size proportions. Defaults to None.
        key: Optional identifier key. Defaults to "".
    """

    def __init__(
        self,
        control_size: float = 0.5,
        random_state: int | None = None,
        sample_size: float | None = None,
        constant_key: bool = True,
        save_groups: bool = True,
        groups_sizes: list[float] | None = None,
        key: Any = "",
    ):
        self.control_size = control_size
        self.random_state = random_state
        self._key = key
        self.constant_key = constant_key
        self.save_groups = save_groups
        self.sample_size = sample_size
        self.groups_sizes = groups_sizes
        super().__init__(key)

    def _generate_params_hash(self):
        hash_parts: list[str] = []
        if self.control_size != 0.5:
            hash_parts.append(f"cs {self.control_size}")
        if self.random_state is not None:
            hash_parts.append(f"rs {self.random_state}")
        if self.groups_sizes is not None:
            hash_parts.append(f"gs {self.groups_sizes}")
        self._params_hash = "|".join(hash_parts)

    def init_from_hash(self, params_hash: str):
        hash_parts: list[str] = params_hash.split("|")
        for hash_part in hash_parts:
            if hash_part.startswith("cs"):
                self.control_size = float(hash_part[hash_part.rfind(" ") + 1 :])
            elif hash_part.startswith("rs"):
                self.random_state = int(hash_part[hash_part.rfind(" ") + 1 :])
            elif hash_part.startswith("gs"):
                self.groups_sizes = []
                groups_sizes = (
                    hash_part[hash_part.find(" ") + 1 :].strip("[]").split(",")
                )
                self.groups_sizes = [float(gs) for gs in groups_sizes]
        self._generate_id()

    @property
    def key(self) -> Any:
        return self._key

    @key.setter
    def key(self, value: Any):
        if not self.constant_key:
            self._key = value
            self._generate_id()

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        data = data.set_value(
            ExperimentDataEnum.additional_fields,
            self._id,
            value,
            role=AdditionalTreatmentRole(),
        )
        if self.save_groups:
            splitter_col = self._id
            unique_vals = data.ds[splitter_col].unique()
            group_keys = list(unique_vals[splitter_col].to_dict().values())
            for group_key in group_keys:
                if group_key is None:
                    continue
                mask = data.ds[splitter_col] == group_key
                group_data = data.ds[mask]
                data.set_value(
                    space=ExperimentDataEnum.groups,
                    executor_id=self._id,
                    value=group_data,
                    key=str(group_key),
                )
        return data

    @staticmethod
    def _const_group_plan(
        data: Dataset,
        const_group_field: str,
        label_map: dict[int, str],
        control_size: float,
        sample_size: float = 1.0,
    ) -> tuple[dict[Any, str], int, float]:
        """Resolve the pinned constant groups in one aggregate pass.

        ``control`` and ``test_N`` are the labels of the split itself, ``test``
        is the documented alias for ``test_1`` of a two-group split. A value
        that is missing, or whose stripped lower-case form is in
        :data:`MISSING_CONST_LABELS`, means the row takes part in the split.
        Anything else is a typo or a group that was not requested.

        Args:
            data: The dataset being split.
            const_group_field: Column holding the pinned group labels.
            label_map: Codes of the split itself, e.g.
                ``{0: "control", 1: "test_1"}``.
            control_size: The requested TOTAL control share.
            sample_size: Fraction of data that will actually be sampled.
                Used to rescale ``control_size`` correctly when
                ``sample_size < 1``.

        Returns:
            A tuple of ``(translation, free_size, control_size)``:

            * ``translation`` maps every distinct value of the const column
            (as its ``str`` representation) to either its split label
            (pinned) or :data:`_FREE_CONST_SENTINEL` (takes part in the
            split);
            * ``free_size`` is the number of rows left to split;
            * ``control_size`` is rescaled so that the TOTAL control share
            (pinned + random) matches the requested one, accounting for
            ``sample_size``.

        Raises:
            ValueError: If a pinned label is not one of the split's groups.
        """
        import warnings

        # Rename to a safe name so that a column literally called "count"
        # does not collide with the value_counts() result column.
        vc = (
            data.select(const_group_field)
            .rename({const_group_field: _CONST_LABEL_COL})
            .value_counts(dropna=False, sort=False)
        )
        # to_records() is a stable public API (unlike .to_dict()["data"]["data"]).
        records = vc.to_records()

        codes = {label: code for code, label in label_map.items()}
        codes.setdefault("test", 1)
        translation: dict[Any, str] = {}
        n_total = 0
        n_pinned = 0
        n_pinned_control = 0

        for rec in records:
            label = rec[_CONST_LABEL_COL]
            count = int(rec["count"])
            n_total += count

            # Missing value (None, NaN): after astype(str) it becomes
            # "None" / "nan", both covered by MISSING_CONST_LABELS.
            if label is None or (isinstance(label, float) and label != label):
                translation[str(label)] = _FREE_CONST_SENTINEL
                continue

            key = str(label)
            if key.strip().lower() in MISSING_CONST_LABELS:
                translation[key] = _FREE_CONST_SENTINEL
                continue

            code = codes.get(key)
            if code is None or code not in label_map:
                raise ValueError(
                    f"Unknown constant group {key!r} in column "
                    f"'{const_group_field}'. Expected one of {sorted(codes)}, "
                    f"or a missing value (None / np.nan / 'nan') for a row "
                    f"that takes part in the split."
                )
            translation[key] = label_map[code]
            n_pinned += count
            if key == "control":
                n_pinned_control += count

        free_size = n_total - n_pinned

        # Warn when pinned controls already exceed the requested quota.
        if n_pinned_control > n_total * control_size:
            warnings.warn(
                f"Pinned control rows ({n_pinned_control}) exceed the "
                f"requested control quota ({n_total * control_size:.0f}). "
                f"The overall control share will be higher than "
                f"{control_size}.",
                UserWarning,
                stacklevel=3,
            )

        # Rescale control_size accounting for sample_size < 1.
        # Pinned rows are never sampled; only free rows are sampled with
        # probability `sample_size`. The target total control count among
        # sampled rows is:
        #   (n_pinned + free_size * sample_size) * control_size
        n_sampled_total = n_pinned + free_size * sample_size
        target_control_total = n_sampled_total * control_size
        free_control_needed = max(0.0, target_control_total - n_pinned_control)
        free_sampled = free_size * sample_size

        adjusted_control_size = (
            free_control_needed / free_sampled if free_sampled > 0 else 0.0
        )
        return translation, free_size, adjusted_control_size

    @staticmethod
    def _inner_function(
        data: Dataset,
        random_state: int | None = None,
        control_size: float = 0.5,
        groups_sizes: list[float] | None = None,
        sample_size: float | None = 1.0,
        const_group_field: str | None = None,
        **kwargs,
    ) -> Dataset:
        """Split data into control/test groups using distributed labeling.

        When *const_group_field* is provided, rows with non-missing values
        in that column are **pinned** to their respective groups and do
        NOT participate in random splitting. Only rows with missing
        values (NaN, None, empty string, etc.) are split randomly.

        The effective ``control_size`` is adjusted so that the TOTAL
        control fraction (pinned + random) matches the requested value.

        Args:
            data: Input dataset to split.
            random_state: Seed for reproducibility.
            control_size: Desired fraction of data in control group.
            groups_sizes: Custom group size proportions.
            sample_size: Fraction of data to sample.
            const_group_field: Column with pinned group assignments.
            **kwargs: Additional arguments (unused).

        Returns:
            Dataset with a ``split`` column assigning each row to a group.

        Raises:
            ValueError: If an unknown constant group label is encountered.
        """
        # ── 1. labels of the split itself ───────────────────────────
        sample_size = sample_size if sample_size is not None else 1.0
        frac = sample_size

        if groups_sizes:
            labels = ["control"] + [
                f"test_{i+1}" for i in range(len(groups_sizes) - 1)
            ]
        else:
            labels = ["control", "test_1"]
        label_map = {i: label for i, label in enumerate(labels)}

        # ── 2. pinned groups: one aggregate pass, on the driver ─────
        translation: dict[Any, str] = {}
        if const_group_field:
            translation, free_size, control_size = (
                AASplitter._const_group_plan(
                    data, const_group_field, label_map, control_size,
                    sample_size=frac,
                )
            )
        else:
            free_size = len(data)

        # ── 3. bucket edges (always in MOD scale, frac handled separately)
        MOD = 10_000_000
        frac_limit = int(frac * MOD)

        if groups_sizes:
            edges: list[int] = []
            cumulative = 0.0
            for size_prop in groups_sizes:
                cumulative += size_prop
                edges.append(int(cumulative * frac_limit))
            edges[-1] = frac_limit
        else:
            edges = [
                int(control_size * frac_limit),
                frac_limit,
            ]

        # ── 4. free rows are split, pinned rows keep their label ────
        parts: list[Dataset] = []
        if const_group_field:
            tagged = data.select(const_group_field).fillna(
                values={const_group_field: _FREE_CONST_SENTINEL}
            )
            if translation:
                tagged = tagged.replace(to_replace=translation)
            if free_size > 0:
                parts.append(
                    tagged[tagged == _FREE_CONST_SENTINEL].random_split_labels(
                        edges=edges,
                        labels=labels,
                        random_state=random_state,
                        frac=frac,
                        name="split",
                    )
                )
            if any(
                label != _FREE_CONST_SENTINEL for label in translation.values()
            ):
                pinned = tagged[tagged != _FREE_CONST_SENTINEL].rename(
                    {const_group_field: "split"}
                )
                pinned.roles = {"split": StatisticRole()}
                parts.append(pinned)
        elif frac > 0:
            parts.append(
                data.random_split_labels(
                    edges=edges,
                    labels=labels,
                    random_state=random_state,
                    frac=frac,
                    name="split",
                )
            )

        if not parts:
            return Dataset.create_empty(
                roles={"split": StatisticRole()},
                backend=data.backend_type,
                session=data.session,
            )

        index_names = data.data.index.names
        for part in parts:
            part.data = part.data.rename_axis(index_names)

        split_ds = parts[0] if len(parts) == 1 else parts[0].append(parts[1:])
        split_ds.roles["split"] = StatisticRole()
        return split_ds

    @timeit(level="SPLIT", prefix="SPLITTER")
    def execute(self, data: ExperimentData) -> ExperimentData:
        const_group_fields = data.ds.search_columns(ConstGroupRole())
        const_group_fields = (
            const_group_fields[0] if len(const_group_fields) > 0 else None
        )
        result = self.calc(
            data.ds,
            random_state=self.random_state,
            control_size=self.control_size,
            sample_size=self.sample_size,
            const_group_field=const_group_fields,
            groups_sizes=self.groups_sizes,
        )
        data = self._set_value(data, result)
        return data


class AASplitterWithStratification(AASplitter):
    """Stratified variant of :class:`AASplitter`.

    Applies the split logic within each stratum defined by
    ``StratificationRole`` columns, ensuring balanced group representation
    across strata.

    Args:
        control_size: Desired fraction of data in the control group.
        random_state: Seed for reproducibility.
        sample_size: Fraction of data to sample.
        constant_key: Whether to keep the key constant across iterations.
        save_groups: Whether to store group subsets.
        groups_sizes: Custom group size proportions.
        key: Optional identifier key.
    """

    @staticmethod
    def _inner_function(
        data: Dataset,
        random_state: int | None = None,
        control_size: float = 0.5,
        grouping_fields=None,
        groups_sizes: list[float] | None = None,
        sample_size: float | None = 1.0,
        const_group_field: str | None = None,
        **kwargs,
    ) -> Dataset:
        """Split data within each stratum, respecting pinned const groups.

        Args:
            data: Input dataset to split.
            random_state: Seed for reproducibility.
            control_size: Desired fraction of data in control group.
            grouping_fields: Stratification column(s) to group by.
            groups_sizes: Custom group size proportions.
            sample_size: Fraction of data to sample.
            const_group_field: Column with pinned group assignments.
            **kwargs: Additional arguments (unused).

        Returns:
            Dataset with a ``split`` column assigning each row to a group.
        """
        if not grouping_fields:
            return AASplitter._inner_function(
                data,
                random_state,
                control_size,
                groups_sizes=groups_sizes,
                sample_size=sample_size,
                const_group_field=const_group_field,
                **kwargs,
            )

        result_splits = []
        for _, group_data in data.groupby(grouping_fields):
            group_split = AASplitter._inner_function(
                group_data,
                random_state,
                control_size,
                groups_sizes=groups_sizes,
                sample_size=sample_size,
                const_group_field=const_group_field,
                **kwargs,
            )
            result_splits.append(group_split)

        if not result_splits:
            return Dataset.create_empty(
                roles={"split": StatisticRole()},
                backend=data.backend_type,
                session=data.session,
            )

        # One concat over all strata: chaining append() per stratum builds an
        # N-deep union tree in the Spark plan.
        return (
            result_splits[0]
            if len(result_splits) == 1
            else result_splits[0].append(result_splits[1:])
        )

    @timeit(level="SPLIT", prefix="SPLITTER_STRAT")
    def execute(self, data: ExperimentData) -> ExperimentData:
        grouping_fields = data.ds.search_columns(StratificationRole())
        const_group_fields = data.ds.search_columns(ConstGroupRole())
        const_group_field = (
            const_group_fields[0] if len(const_group_fields) > 0 else None
        )
        if data.ds.backend_type == BackendsEnum.spark and not data.ds.is_persisted:
            data.ds.persist(storage_level="MEMORY_AND_DISK", action="count")
        result = self.calc(
            data.ds,
            random_state=self.random_state,
            control_size=self.control_size,
            grouping_fields=grouping_fields,
            groups_sizes=self.groups_sizes,
            const_group_field=const_group_field,
        )
        if isinstance(result, Dataset):
            result = result.replace_roles({"split": AdditionalTreatmentRole()})
        data = self._set_value(data, result)
        return data