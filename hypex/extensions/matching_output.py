from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

from ..dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    GroupingRole,
    InfoRole,
    SmallDataset,
    StatisticRole,
)
from ..dataset.backends import PandasDataset, SparkDataset
from ..utils import ID_SPLIT_SYMBOL, MATCHING_INDEXES_SPLITTER_SYMBOL
from ..utils.adapter import Adapter
from ..utils.registry import backend_factory


class MatchingOutputExtension(ABC):
    @abstractmethod
    def extract_indexes(
        self,
        experiment_data: ExperimentData,
        index_ids: list[str],
        summary: dict[str, Any],
        reformatted_summary: dict[str, Any],
    ) -> Dataset | SmallDataset:
        raise NotImplementedError

    @abstractmethod
    def extract_summary_indexes(
        self, experiment_data: ExperimentData, index_ids: list[str]
    ) -> dict[str, str]:
        raise NotImplementedError

    def build_full_data(
        self, experiment_data: ExperimentData, indexes: Dataset
    ) -> tuple[Dataset, Dataset]:
        if indexes.backend_type != experiment_data.ds.backend_type:
            indexes = indexes.to_backend(
                backend=experiment_data.ds.backend_type,
                session=experiment_data.ds.session,
            )

        context = self._prepare_full_data(experiment_data)
        output_indexes = Dataset.create_empty(
            roles={},
            backend=experiment_data.ds.backend_type,
            session=experiment_data.ds.session,
        )
        full_data = None

        for i in range(len(indexes.columns)):
            t_indexes = indexes.iloc[:, i]
            col_name = t_indexes.columns[0]
            matched_data = self._match(
                experiment_data, t_indexes, col_name, context=context
            )
            matched_data = matched_data.rename(
                {col: f"{col}_matched_{i}" for col in matched_data.columns}
            )

            reindexed_matched = experiment_data.ds.merge(
                matched_data,
                left_index=True,
                right_index=True,
                how="left",
            )
            reindexed_matched = reindexed_matched.drop(
                columns=list(experiment_data.ds.columns),
            )

            if output_indexes.is_empty():
                output_indexes = t_indexes
            else:
                output_indexes = output_indexes.add_column(
                    data=t_indexes.raw_data,
                    role={
                        col: t_indexes.roles.get(col, InfoRole())
                        for col in t_indexes.columns
                    },
                )

            if full_data is None:
                full_data = experiment_data.ds.merge(
                    reindexed_matched,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            else:
                full_data = full_data.merge(
                    reindexed_matched,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            full_data.checkpoint(eager=True)

        if full_data is None:
            full_data = experiment_data.ds
        return full_data, output_indexes

    def _prepare_full_data(self, experiment_data: ExperimentData) -> dict[str, Any]:
        if not experiment_data.ds.is_persisted:
            experiment_data.ds.checkpoint(eager=True)
        return {}

    @abstractmethod
    def _match(
        self,
        experiment_data: ExperimentData,
        t_indexes: Dataset,
        col_name: str,
        context: dict[str, Any],
    ) -> Dataset:
        raise NotImplementedError


@backend_factory.register(MatchingOutputExtension, PandasDataset)
class PandasMatchingOutputExtension(MatchingOutputExtension):
    def extract_summary_indexes(
        self, experiment_data: ExperimentData, index_ids: list[str]
    ) -> dict[str, str]:
        return {
            f"indexes{ID_SPLIT_SYMBOL}{col.split(ID_SPLIT_SYMBOL)[3]}": MATCHING_INDEXES_SPLITTER_SYMBOL.join(
                str(int(i)) if isinstance(i, float) and i.is_integer() else str(i)
                for i in experiment_data.additional_fields[col]._to_numpy()
            )
            for col in index_ids
        }

    def extract_indexes(
        self,
        experiment_data: ExperimentData,
        index_ids: list[str],
        summary: dict[str, Any],
        reformatted_summary: dict[str, Any],
    ) -> Dataset | SmallDataset:
        ds_len = len(experiment_data.ds)

        if "indexes" in reformatted_summary:
            indexes_items = reformatted_summary.pop("indexes")
            are_nested = all(isinstance(v, dict) for v in indexes_items.values())

            if are_nested:
                index_parts = [
                    self._collect_grouped_indexes(experiment_data, values).rename(
                        {"indexes": f"indexes_{group}"}
                    )
                    for group, values in indexes_items.items()
                ]
            else:
                index_parts = []
                for group, values in indexes_items.items():
                    idx_values = list(
                        map(int, values.split(MATCHING_INDEXES_SPLITTER_SYMBOL))
                    )
                    ds = SmallDataset.from_dict(
                        {f"indexes_{group}": idx_values},
                        roles={f"indexes_{group}": StatisticRole()},
                    )
                    if len(ds) == ds_len:
                        ds.index = Adapter.to_list(experiment_data.ds.index)
                    else:
                        warnings.warn(
                            f"Matched indexes length ({len(ds)}) != "
                            f"dataset length ({ds_len}) for group "
                            f"'{group}'. Alignment skipped.",
                            UserWarning,
                            stacklevel=2,
                        )
                    index_parts.append(ds)

            if index_parts:
                return index_parts[0].append(other=index_parts[1:], axis=1).sort()
            return SmallDataset.create_empty()

        indexes_data = summary.get("indexes", "").split(
            MATCHING_INDEXES_SPLITTER_SYMBOL
        )
        if indexes_data and indexes_data[0]:
            indexes = SmallDataset.from_dict(
                {"indexes": list(map(int, indexes_data))},
                roles={"indexes": AdditionalMatchingRole()},
            )
            if len(indexes) == ds_len:
                indexes.index = Adapter.to_list(experiment_data.ds.index)
            else:
                warnings.warn(
                    f"Matched indexes length ({len(indexes)}) != "
                    f"dataset length ({ds_len}). Alignment skipped.",
                    UserWarning,
                    stacklevel=2,
                )
            return indexes
        return SmallDataset.create_empty()

    @staticmethod
    def _collect_grouped_indexes(
        experiment_data: ExperimentData, group: dict
    ) -> Dataset:
        group_indexes_id = experiment_data.ds.search_columns(GroupingRole())
        indexes = []
        for group_name, values in group.items():
            ds = SmallDataset.from_dict(
                {
                    "indexes": list(
                        map(int, values.split(MATCHING_INDEXES_SPLITTER_SYMBOL))
                    )
                },
                roles={"indexes": StatisticRole()},
            )
            ds.index = Adapter.to_list(
                experiment_data.ds[
                    experiment_data.ds[group_indexes_id] == group_name
                ].index
            )
            indexes.append(ds)
        return indexes[0].append(indexes[1:]).sort()

    def _match(
        self,
        experiment_data: ExperimentData,
        t_indexes: Dataset,
        col_name: str,
        context: dict[str, Any],
    ) -> Dataset:
        index_values: list = Adapter.to_list(t_indexes.get_values(column=col_name))
        positional_indices: list = Adapter.to_list(t_indexes.raw_data.index)

        valid_positions: list = []
        valid_lookups: list = []
        for pos, idx in zip(positional_indices, index_values):
            if idx != -1:
                valid_positions.append(pos)
                valid_lookups.append(idx)

        if not valid_positions:
            return Dataset.create_empty(
                roles={},
                backend=experiment_data.ds.backend_type,
            )

        matched_data: Dataset = experiment_data.ds.loc[valid_lookups]
        matched_data.index = valid_positions
        return matched_data


@backend_factory.register(MatchingOutputExtension, SparkDataset)
class SparkMatchingOutputExtension(MatchingOutputExtension):
    def extract_summary_indexes(
        self, experiment_data: ExperimentData, index_ids: list[str]
    ) -> dict[str, str]:
        return {}

    def extract_indexes(
        self,
        experiment_data: ExperimentData,
        index_ids: list[str],
        summary: dict[str, Any],
        reformatted_summary: dict[str, Any],
    ) -> Dataset:
        reformatted_summary.pop("indexes", None)
        additional = experiment_data.additional_fields
        available = sorted(
            (col for col in index_ids if col in additional.columns),
            key=lambda c: int(str(c).split(ID_SPLIT_SYMBOL)[-1]),
        )
        if not available:
            return Dataset.create_empty(
                roles={},
                backend=experiment_data.ds.backend_type,
                session=experiment_data.ds.session,
            )
        indexes = additional[available]
        return indexes.rename(
            {
                col: f"indexes_{col.split(ID_SPLIT_SYMBOL)[-1]}"
                for col in indexes.columns
            }
        )

    def _prepare_full_data(self, experiment_data: ExperimentData) -> dict[str, Any]:
        super()._prepare_full_data(experiment_data)
        original_columns = set(experiment_data.ds.columns)
        ds_reset = experiment_data.ds.reset_index(drop=False)
        idx_col = next(c for c in ds_reset.columns if c not in original_columns)
        ds_reset.checkpoint(eager=True)
        return {"ds_reset": ds_reset, "idx_col": idx_col}

    def _match(
        self,
        experiment_data: ExperimentData,
        t_indexes: Dataset,
        col_name: str,
        context: dict[str, Any],
    ) -> Dataset:
        filtered = t_indexes[t_indexes[col_name] != -1]
        filtered_reset = filtered.reset_index(drop=False)
        pos_col = next(c for c in filtered_reset.columns if c not in filtered.columns)
        mapping_ds = filtered_reset.rename(
            {pos_col: "_hypex_pos", col_name: "_hypex_lookup"}
        )

        ds_reset = context.get("ds_reset")
        idx_col = context.get("idx_col")
        if ds_reset is None:
            original_columns = set(experiment_data.ds.columns)
            ds_reset = experiment_data.ds.reset_index(drop=False)
            idx_col = next(c for c in ds_reset.columns if c not in original_columns)

        matched_data = mapping_ds.merge(
            ds_reset,
            left_on="_hypex_lookup",
            right_on=idx_col,
            how="left",
        )
        matched_data = matched_data.set_index("_hypex_pos", drop=True)
        return matched_data.drop(columns=["_hypex_lookup", idx_col])
