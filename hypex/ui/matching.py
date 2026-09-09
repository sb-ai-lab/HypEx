# hypex/ui/matching.py
from __future__ import annotations

from typing import Any

import pandas as pd


from ..analyzers.matching import MatchingAnalyzer
from ..dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    GroupingRole,
    InfoRole,
    SmallDataset,
    StatisticRole,
    TargetRole,
)
from ..reporters.matching import MatchingDictReporter, MatchingQualityDatasetReporter
from ..utils import BackendsEnum, ID_SPLIT_SYMBOL, MATCHING_INDEXES_SPLITTER_SYMBOL
from ..utils.logger import logger
from .base import Output

@logger.log_methods(log_args=False, log_result=False, private=True, static=True)
class MatchingOutput(Output):
    """Output handler for matching experiment results."""

    resume: Dataset
    full_data: Dataset
    quality_results: Dataset

    def __init__(self, searching_class: type = MatchingAnalyzer):
        """Initialize matching output with resume and quality reporters.

        Args:
            searching_class: The analyzer class used to search for results.
        """
        super().__init__(
            resume_reporter=MatchingDictReporter(searching_class),
            additional_reporters={"quality_results": MatchingQualityDatasetReporter()},
        )

    def _extract_full_data(self, experiment_data: ExperimentData, indexes: Dataset):
        """Build the full matched dataset from original data and matched indexes.
        Materialises the dataset index as a plain Python list so that it can be
        safely assigned to Pandas-backed SmallDataset objects regardless of the
        backend used by ``experiment_data.ds`` (Pandas or Spark).
        Args:
            experiment_data: The experiment data container.
            indexes: Dataset containing matched neighbor indexes per group.
        """
        # ── Convert to list to avoid PySpark Index → Pandas assignment error ──
        ds_index = experiment_data.ds.index.to_numpy().tolist()

        self.indexes = SmallDataset.create_empty(roles={})

        for i in range(len(indexes.columns)):
            t_indexes = indexes.iloc[:, i]
            t_indexes.index = ds_index
            filtered_field = indexes.drop(
                indexes[indexes[t_indexes.columns[0]] == -1], axis=0
            )
            lookup_vals = list(map(lambda x: x[0], filtered_field.get_values()))
            filtered_index_list = filtered_field.index.to_numpy().tolist()
            if experiment_data.ds.backend_type == BackendsEnum.spark:
                mapping_ds = Dataset(
                    roles={
                        "_hypex_pos": InfoRole(),
                        "_hypex_lookup": InfoRole(),
                    },
                    data=pd.DataFrame({
                        "_hypex_pos": filtered_index_list,
                        "_hypex_lookup": lookup_vals,
                    }),
                    backend=BackendsEnum.spark,
                    session=experiment_data.ds.session,
                )
                orig_cols = set(experiment_data.ds.columns)
                ds_reset = experiment_data.ds.reset_index()
                idx_col = next(c for c in ds_reset.columns if c not in orig_cols)
                matched_data = mapping_ds.merge(
                    ds_reset,
                    left_on="_hypex_lookup",
                    right_on=idx_col,
                    how="left",
                )
                matched_data = matched_data.set_index("_hypex_pos", drop=True)
                matched_data = matched_data.drop(columns=["_hypex_lookup", idx_col])
            else:
                matched_data = experiment_data.ds.loc[lookup_vals]
                matched_data.index = filtered_index_list
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
                columns=list(experiment_data.ds.columns)
            )

            if self.indexes.is_empty():
                self.indexes = t_indexes
            else:
                self.indexes.add_column(
                    data=t_indexes.data,
                    role={col: t_indexes.roles.get(col, InfoRole()) for col in t_indexes.columns}
                )

            if hasattr(self, "full_data") and self.full_data is not None:
                self.full_data = self.full_data.merge(
                    reindexed_matched,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            else:
                self.full_data = experiment_data.ds.merge(
                    reindexed_matched,
                    left_index=True,
                    right_index=True,
                    how="left",
                )

    @staticmethod
    def _reformat_resume(resume: dict[str, Any]) -> dict[str, Any]:
        """Reformat a flat resume dictionary with composite keys into a nested structure.

        Args:
            resume: Flat dictionary with composite keys separated by ID_SPLIT_SYMBOL.

        Returns:
            Nested dictionary grouped by metric name and index.
        """
        reformatted_resume: dict[str, Any] = {}
        for key, value in resume.items():
            if ID_SPLIT_SYMBOL not in key:
                continue
            keys = key.split(ID_SPLIT_SYMBOL)
            if keys[0] == "indexes":
                if len(keys) > 2:
                    reformatted_resume.setdefault("indexes", {}).setdefault(
                        keys[1], {}
                    )[keys[2]] = value
                else:
                    reformatted_resume.setdefault("indexes", {})[keys[1]] = value
            else:
                l1_key = keys[0] if len(keys) < 3 else f"{keys[2]} {keys[0]}"
                reformatted_resume.setdefault(l1_key, {})[keys[1]] = value
        return reformatted_resume

    @staticmethod
    def _collect_grouped_indexes(experiment_data: ExperimentData, group: dict) -> Dataset:
        """Collect matched indexes for grouped matching results.

        Args:
            experiment_data: The experiment data container.
            group: Dictionary mapping group names to matched index strings.

        Returns:
            Dataset with collected indexes sorted by index.
        """
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
            ds.index = experiment_data.ds[
                experiment_data.ds[group_indexes_id] == group_name
            ].index
            indexes.append(ds)
        return indexes[0].append(indexes[1:]).sort()

    def extract(self, experiment_data: ExperimentData):
        """Extract and format all matching results from experiment data.

        Args:
            experiment_data: The experiment data container with matching results.
        """
        # Let the base class handle additional_reporters (like quality_results)
        super().extract(experiment_data)

        resume = self.resume_reporter.report(experiment_data)
        reformatted_resume = self._reformat_resume(resume)

        if "indexes" in reformatted_resume.keys():
            indexes_items = reformatted_resume.pop("indexes")
            are_nested = all(isinstance(v, dict) for v in indexes_items.values())
            if are_nested:
                indexes = [
                    self._collect_grouped_indexes(experiment_data, values).rename(
                        {"indexes": f"indexes_{group}"}
                    )
                    for group, values in indexes_items.items()
                ]
            else:
                indexes = [
                    SmallDataset.from_dict(
                        {
                            f"indexes_{group}": list(
                                map(int, values.split(MATCHING_INDEXES_SPLITTER_SYMBOL))
                            )
                        },
                        roles={f"indexes_{group}": StatisticRole()},
                    )
                    for group, values in indexes_items.items()
                ]
            indexes = indexes[0].append(other=indexes[1:], axis=1).sort()
        else:
            indexes_data = resume.get("indexes", "").split(MATCHING_INDEXES_SPLITTER_SYMBOL)
            if indexes_data and indexes_data[0]:
                indexes = SmallDataset.from_dict(
                    {"indexes": list(map(int, indexes_data))},
                    roles={"indexes": AdditionalMatchingRole()},
                )
            else:
                indexes = SmallDataset.create_empty()


        if reformatted_resume:
            first_key = next(iter(reformatted_resume.keys()))
            group_keys = list(reformatted_resume[first_key].keys())
            transposed_resume = {
                metric: [values[group] for group in group_keys]
                for metric, values in reformatted_resume.items()
            }
            self.resume = SmallDataset.from_dict(
                {"data": transposed_resume, "index": group_keys},
                roles={
                    column: StatisticRole()
                    for column in list(reformatted_resume.keys())
                },
            )
        else:
            self.resume = SmallDataset.create_empty()

        self._extract_full_data(
            experiment_data,
            indexes,
        )
        self.resume.data = self.resume.data.round(2)
