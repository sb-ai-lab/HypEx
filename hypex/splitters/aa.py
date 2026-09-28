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
        """
        Splits data into control/test groups using distributed labeling.
        Avoids iloc/sort-limit OOM issues on Spark.
        """
        # Handle const_group_field filtering
        if const_group_field:
            data_to_split = data.filter(data.select(const_group_field).isna())
        else:
            data_to_split = data
        # Determine fraction and total count
        # Note: len() on Spark Dataset triggers a count(), which is necessary 
        # to calculate exact edges for balanced splits.
        n_total = len(data_to_split)
        frac = sample_size if sample_size is not None else 1.0
        n_sampled = int(n_total * frac)

        if n_sampled == 0:
            # Return empty dataset with same structure if nothing to sample
            return Dataset.create_empty(
                roles={"split": StatisticRole()}, backend=data.backend_type
            )

        MOD = 10_000_000

        effective_mod = int(frac * MOD) if frac < 1.0 else MOD

        if groups_sizes:
            labels = ["control"] + [
                f"test_{i+1}" for i in range(len(groups_sizes) - 1)
            ]
            edges = []
            cumulative = 0.0
            for size_prop in groups_sizes:
                cumulative += size_prop
                edges.append(int(cumulative * effective_mod))
            edges[-1] = effective_mod
        else:
            n_control = int(n_sampled * control_size)
            edges = [
                int((n_control / n_sampled) * effective_mod)
                if n_sampled > 0
                else 0,
                effective_mod,
            ]
            labels = ["control", "test_1"]

        # Call the new backend method
        # This returns a Dataset with the original index and a new 'split' column
        split_ds = data_to_split.random_split_labels(
            edges=edges,
            labels=labels,
            random_state=random_state,
            frac=frac,
            name="split",
        )
        # Ensure roles are set correctly
        split_ds.roles["split"] = StatisticRole() # Or AdditionalTreatmentRole depending on downstream usage
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
    def _apply_const_groups(
        split_ds: Dataset,
        const_data: dict[str, Dataset],
        label_map: dict[int, str],
        const_group_field: str | None,
    ) -> Dataset:
        """Write pinned groups over the split of free rows.

        For each pinned group in *const_data*, creates a single-column
        Dataset with the appropriate split label and appends it to the
        random-split result.

        Args:
            split_ds: Dataset with split labels for free rows.
            const_data: Dict mapping group label → Dataset of pinned rows.
            label_map: Mapping from group codes to labels
                (e.g. ``{0: "control", 1: "test_1"}``).
            const_group_field: Name of the const group column.

        Returns:
            Dataset with both free and pinned rows labeled.

        Raises:
            ValueError: If an unknown constant group label is found.
        """
        # Build reverse mapping: label → code
        codes = {label: code for code, label in label_map.items()}
        codes.setdefault("test", 1)

        pinned_rows = []
        for group, group_data in const_data.items():
            group_str = str(group).strip().lower()
            code = codes.get(group_str)
            if code is None:
                raise ValueError(
                    f"Unknown constant group {str(group)!r} in column "
                    f"'{const_group_field}'. Expected one of "
                    f"{sorted(codes)}, or a missing value "
                    f"(None / np.nan / 'nan') for a row that takes "
                    f"part in the split."
                )
            label = label_map.get(code, str(group))
            # Create a single-column dataset with the split label
            # matching the index of the pinned group data.
            import pandas as pd
            pinned_ds = Dataset.create_empty(
                roles={"split": StatisticRole()},
                backend=group_data.backend_type,
                session=group_data.session if hasattr(group_data, "session") else None,
            )
            pinned_ds.data = pd.DataFrame(
                {"split": [label] * len(group_data)},
                index=group_data.index,
            )
            pinned_rows.append(pinned_ds)

        if pinned_rows:
            return split_ds.append(pinned_rows)
        return split_ds


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
            const_group_field=const_group_field,  # ── НОВОЕ ──
        )
        if isinstance(result, Dataset):
            result = result.replace_roles({"split": AdditionalTreatmentRole()})

        data = self._set_value(data, result)

        # if data.ds.backend_type == BackendsEnum.spark:
        #     data.ds.checkpoint(eager=True)

        return data
