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
    TreatmentRole,
)
from ..dataset.roles import ConstGroupRole, IndexRole
from ..executor import Calculator
from ..utils import ExperimentDataEnum, BackendsEnum, timeit


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
            return Dataset.create_empty(roles={"split": StatisticRole()}, backend=data.backend_type)

        # Calculate edges for labels
        labels = ["control"]
        edges = []

        MOD = 10_000_000

        if groups_sizes:
            labels = ["control"] + [f"test_{i+1}" for i in range(len(groups_sizes) - 1)]
            edges = []
            cumulative = 0.0
            for i, size_prop in enumerate(groups_sizes):
                cumulative += size_prop
                edges.append(int(cumulative * MOD))
            edges[-1] = MOD
        else:
            n_control = int(n_sampled * control_size)
            edges = [
                int((n_control / n_sampled) * MOD) if n_sampled > 0 else 0,
                MOD,
            ]
            labels = ["control", "test_1"]

        # Call the new backend method
        # This returns a Dataset with the original index and a new 'split' column
        split_ds = data_to_split.random_split_labels(
            edges=edges,
            labels=labels,
            random_state=random_state,
            frac=frac,
            name="split"
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
        
        if data.ds.backend_type == BackendsEnum.spark and not data.ds.is_persisted:
            data.ds.persist(storage_level="MEMORY_AND_DISK", action="count")

        result = self.calc(
            data.ds,
            random_state=self.random_state,
            control_size=self.control_size,
            grouping_fields=grouping_fields,
            groups_sizes=self.groups_sizes,
        )
        
        if isinstance(result, Dataset):
            result = result.replace_roles({"split": AdditionalTreatmentRole()})
        
        data = self._set_value(data, result)

        # if data.ds.backend_type == BackendsEnum.spark:
        #     data.ds.checkpoint(eager=True)

        return data
