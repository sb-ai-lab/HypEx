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

    def _extract_full_data(
        self, experiment_data: ExperimentData, indexes: Dataset
    ) -> None:
        """Build the full matched dataset from original data and matched indexes.

        Constructs ``self.indexes`` (raw match-index columns) and
        ``self.full_data`` (the original dataset augmented with one
        ``*_matched_{i}`` column group per neighbor position).

        For the Spark backend every step is a lazy transformation — no data
        is collected to the driver.  For the Pandas backend the in-memory
        label lookup is preserved.

        Args:
            experiment_data: The experiment data container holding the
                original dataset in ``experiment_data.ds``.
            indexes: A ``Dataset`` whose columns contain matched neighbor
                indices.  Rows with value ``-1`` mark unmatched observations
                and are excluded from the matched portion.
        """
        backend = experiment_data.ds.backend_type

        # ── Ensure indexes uses the same backend as experiment_data.ds ──
        if indexes.backend_type != backend:
            indexes = indexes.to_backend(
                backend=backend,
                session=experiment_data.ds.session,
            )

        # ── Break the DAG once before the iterative merge loop ──────────
        if backend == BackendsEnum.spark and not experiment_data.ds.is_persisted:
            experiment_data.ds.checkpoint(eager=True)

        # ── Pre-compute ds_reset ONCE outside the loop ─────────────────
        # Previously ds_reset was created inside _match_spark on EVERY
        # iteration, duplicating the entire experiment_data.ds lineage
        # each time.  Hoisting it here ensures the reset_index node
        # appears exactly once in the DAG.
        ds_reset: Dataset | None = None
        idx_col: str | None = None
        if backend == BackendsEnum.spark:
            orig_cols = set(experiment_data.ds.columns)
            ds_reset = experiment_data.ds.reset_index(drop=False)
            idx_col = next(c for c in ds_reset.columns if c not in orig_cols)
            # Checkpoint ds_reset so downstream joins reference a
            # materialized node instead of re-expanding the full lineage.
            ds_reset.checkpoint(eager=True)

        # ── Container for raw index columns (same backend as input) ─────
        self.indexes = Dataset.create_empty(
            roles={},
            backend=backend,
            session=experiment_data.ds.session,
        )

        for i in range(len(indexes.columns)):
            t_indexes = indexes.iloc[:, i]
            col_name = t_indexes.columns[0]

            # ── Build the matched subset for this neighbor position ─────
            if backend == BackendsEnum.spark:
                matched_data = self._match_spark(
                    experiment_data, t_indexes, col_name,
                    ds_reset=ds_reset,
                    idx_col=idx_col,
                )
            else:
                matched_data = self._match_pandas(
                    experiment_data, t_indexes, col_name,
                )

            # ── Rename matched columns with a position suffix ───────────
            matched_data = matched_data.rename(
                {col: f"{col}_matched_{i}" for col in matched_data.columns}
            )

            # ── Left-join matched columns onto the full original index ──
            reindexed_matched = experiment_data.ds.merge(
                matched_data,
                left_index=True,
                right_index=True,
                how="left",
            )
            reindexed_matched = reindexed_matched.drop(
                columns=list(experiment_data.ds.columns),
            )

            # ── Accumulate raw index columns ────────────────────────────
            if self.indexes.is_empty():
                self.indexes = t_indexes
            else:
                self.indexes = self.indexes.add_column(
                    data=t_indexes.data,
                    role={
                        col: t_indexes.roles.get(col, InfoRole())
                        for col in t_indexes.columns
                    },
                )

            # ── Accumulate the full matched dataset ─────────────────────
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

            # ── Checkpoint after EVERY merge to truncate lineage ────────
            # Without this, each merge embeds the entire accumulated
            # lineage, producing O(N²) plan growth.  With checkpoint,
            # each iteration starts from a clean materialized snapshot.
            if backend == BackendsEnum.spark:
                self.full_data.checkpoint(eager=True)


    def _match_spark(
        self,
        experiment_data: ExperimentData,
        t_indexes: Dataset,
        col_name: str,
        ds_reset: Dataset | None = None,
        idx_col: str | None = None,
    ) -> Dataset:
        """Build matched data via a lazy Spark join — no driver collection.

        The match-index column serves as a lookup key against the original
        dataset's index.  Rows with value ``-1`` are filtered out before
        the join so they do not pollute the result.

        Args:
            experiment_data: The experiment data container.
            t_indexes: Single-column ``Dataset`` with match indices.
            col_name: Name of the match-index column in *t_indexes*.
            ds_reset: Pre-computed ``experiment_data.ds.reset_index()``.
                When provided, avoids re-creating this node on every call.
            idx_col: Name of the exposed index column in *ds_reset*.

        Returns:
            A ``Dataset`` containing the matched rows, indexed by the
            original observation positions.
        """
        # 1. Filter out unmatched rows (-1) — lazy transformation.
        filtered = t_indexes[t_indexes[col_name] != -1]

        # 2. Expose the row index as a column so it survives the join.
        filtered_reset = filtered.reset_index(drop=False)
        pos_col = next(
            c for c in filtered_reset.columns if c not in filtered.columns
        )

        # 3. Rename helper columns for the join.
        mapping_ds = filtered_reset.rename({
            pos_col: "_hypex_pos",
            col_name: "_hypex_lookup",
        })

        # 4. Use pre-computed ds_reset or fall back to computing it.
        if ds_reset is None:
            orig_cols = set(experiment_data.ds.columns)
            ds_reset = experiment_data.ds.reset_index(drop=False)
            idx_col = next(c for c in ds_reset.columns if c not in orig_cols)

        # 5. Lazy join: match-index value → original dataset row.
        matched_data = mapping_ds.merge(
            ds_reset,
            left_on="_hypex_lookup",
            right_on=idx_col,
            how="left",
        )

        # 6. Restore the positional index and drop helper columns.
        matched_data = matched_data.set_index("_hypex_pos", drop=True)
        matched_data = matched_data.drop(columns=["_hypex_lookup", idx_col])
        return matched_data
    
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
