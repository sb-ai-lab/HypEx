from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..dataset import (
    AdditionalTreatmentRole,
    Dataset,
    ExperimentData,
    StatisticRole,
    StratificationRole,
)
from ..dataset.roles import ConstGroupRole
from ..executor import Calculator
from ..utils import BackendsEnum, ExperimentDataEnum, timeit

MISSING_CONST_LABELS = frozenset({"", "nan", "none", "nat", "<na>"})

# A value that cannot collide with a split label: rows carrying it in the
# tagged const column are the ones that take part in the random split.
_FREE_CONST_SENTINEL = "__hypex_free_const_group__"


class AASplitter(Calculator):
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
                    key=str(group_key)
                )
        return data

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
        NOT participate in random splitting.  Only rows with missing
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
            For ``sample_size`` of 1.0 the result holds exactly one row per
            input row, with no duplicate index entries.

        Raises:
            ValueError: If an unknown constant group label is encountered.
        """
        # ── 1. labels of the split itself ───────────────────────────
        sample_size = sample_size if sample_size is not None else 1.0
        frac = sample_size

        MOD = 10_000_000
        effective_mod = int(frac * MOD) if frac < 1.0 else MOD

        if groups_sizes:
            labels = ["control"] + [
                f"test_{i+1}" for i in range(len(groups_sizes) - 1)
            ]
        else:
            labels = ["control", "test_1"]
        label_map = {i: label for i, label in enumerate(labels)}

        # ── 2. pinned groups: one aggregate pass, resolved on the driver ──
        translation: dict[Any, str] = {}
        if const_group_field:
            translation, free_size, control_size = AASplitter._const_group_plan(
                data, const_group_field, label_map, control_size
            )
        else:
            free_size = len(data)

        # ── 3. bucket edges ─────────────────────────────────────────
        n_sampled = int(free_size * frac)
        if groups_sizes:
            edges = []
            cumulative = 0.0
            for size_prop in groups_sizes:
                cumulative += size_prop
                edges.append(int(cumulative * effective_mod))
            edges[-1] = effective_mod
        else:
            n_control = int(n_sampled * control_size) if n_sampled > 0 else 0
            edges = [
                int((n_control / n_sampled) * effective_mod)
                if n_sampled > 0
                else 0,
                effective_mod,
            ]

        # ── 4. free rows are split, pinned rows keep their label ────
        parts: list[Dataset] = []
        if const_group_field:
            # A single-column view of the const column in which every free row
            # holds the sentinel and every pinned row already holds its split
            # label. Both subsets are then plain row filters on the same frame,
            # which is what keeps the Spark plan join-free: a mask built with
            # Dataset.isin() would come from a different pyspark.pandas anchor
            # and turn each selection into a SortMergeJoin.
            tagged = data.select(const_group_field).fillna(
                values={const_group_field: _FREE_CONST_SENTINEL}
            )
            if translation:
                tagged = tagged.replace(to_replace=translation)
            if free_size > 0 and n_sampled > 0:
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
        elif n_sampled > 0:
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

        # random_split_labels names the index 'index' on the Spark backend
        # (reset_index of an unnamed index), which makes ps.concat refuse to
        # append the pinned part. Aligning every part with the source index is
        # a no-op on pandas.
        index_names = data.data.index.names
        for part in parts:
            part.data = part.data.rename_axis(index_names)  # type: ignore[operator]

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

        # if data.ds.backend_type == BackendsEnum.spark:
        #     data.ds.checkpoint(eager=True)

        return data

    @staticmethod
    def _const_group_plan(
        data: Dataset,
        const_group_field: str,
        label_map: dict[int, str],
        control_size: float,
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

        Returns:
            A tuple of (translation, free_size, control_size):

            * ``translation`` maps every non-null value of the const column to
              either its split label (pinned) or
              :data:`_FREE_CONST_SENTINEL` (takes part in the split);
            * ``free_size`` is the number of rows left to split;
            * ``control_size`` is rescaled so that the TOTAL control share
              (pinned + random) matches the requested one.

        Raises:
            ValueError: If a pinned label is not one of the split's groups.
        """
        # One aggregate pass. The frame has one row per distinct label,
        # including the null bucket, so it also gives the row total.
        # sort=False keeps the Spark plan at a single Exchange.
        counts = (
            data.select(const_group_field)
            .value_counts(dropna=False, sort=False)
            .to_dict()["data"]["data"]
        )
        codes = {label: code for code, label in label_map.items()}
        codes.setdefault("test", 1)

        translation: dict[Any, str] = {}
        n_total = 0
        n_pinned = 0
        n_pinned_control = 0
        for label, count in zip(counts[const_group_field], counts["count"]):
            count = int(count)
            n_total += count
            # a real missing value: fillna() turns it into the sentinel
            if label is None or (isinstance(label, float) and label != label):
                continue
            key = str(label)
            if key.strip().lower() in MISSING_CONST_LABELS:
                translation[label] = _FREE_CONST_SENTINEL
                continue
            code = codes.get(key)
            if code is None or code not in label_map:
                raise ValueError(
                    f"Unknown constant group {key!r} in column "
                    f"'{const_group_field}'. Expected one of {sorted(codes)}, "
                    f"or a missing value (None / np.nan / 'nan') for a row "
                    f"that takes part in the split."
                )
            translation[label] = label_map[code]
            n_pinned += count
            if key == "control":
                n_pinned_control += count

        free_size = n_total - n_pinned
        control_size = (
            0.0
            if free_size == 0
            else max(0.0, (n_total * control_size - n_pinned_control) / free_size)
        )
        return translation, free_size, control_size


class AASplitterWithStratification(AASplitter):
    @staticmethod
    def _inner_function(
        data: Dataset,
        random_state: int | None = None,
        control_size: float = 0.5,
        grouping_fields=None,
        groups_sizes: list[float] | None = None,
        sample_size: float | None = 1.0,
        **kwargs,
    ) -> Dataset:
        if not grouping_fields:
            return AASplitter._inner_function(
                data,
                random_state,
                control_size,
                groups_sizes=groups_sizes,
                sample_size=sample_size,
                **kwargs,
            )
        
        # For stratified split, we need to apply the split logic within each group.
        # However, doing len() per group is expensive.
        # Optimization: Use the global random_split_labels but include grouping fields in the hash?
        # No, stratification requires exact proportions PER GROUP.
        
        # We must iterate groups. To avoid OOM, we rely on the new random_split_labels 
        # being safe for each group partition.
        
        result_splits = []
        
        # GroupBy in Spark Dataset returns an iterator of (key, Dataset)
        # Note: This materializes groups if not careful, but with the new split method,
        # each group's split is a lightweight transformation.
        
        for _, group_data in data.groupby(grouping_fields):
            # group_data is a Dataset
            group_split = AASplitter._inner_function(
                group_data,
                random_state,
                control_size,
                groups_sizes=groups_sizes,
                sample_size=sample_size,
                **kwargs,
            )
            result_splits.append(group_split)
            
        if not result_splits:
            return Dataset.create_empty(roles={"split": StatisticRole()}, backend=data.backend_type)
            
        # Append all splits back together
        combined_split = result_splits[0]
        for i in range(1, len(result_splits)):
            combined_split = combined_split.append(result_splits[i])
            
        return combined_split

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

        # if data.ds.backend_type == BackendsEnum.spark:
        #     data.ds.checkpoint(eager=True)

        return data
