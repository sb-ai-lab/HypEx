"""Reporters for CUPED variance reduction results."""

from __future__ import annotations

from typing import Any

from ..dataset import ExperimentData, SmallDataset
from ..dataset.roles import InfoRole, StatisticRole
from ..transformers.cuped import CUPEDTransformer
from ..utils import ExperimentDataEnum
from .abstract import Reporter


class CupedReporter(Reporter):
    """Extracts CUPED variance-reduction metrics from ``analysis_tables``.

    Produces a ``SmallDataset`` with one row per target feature:

    +------------------+---------------------------+
    | feature          | variance_reduction_pct    |
    +==================+===========================+
    | post_spends      | 42.7                      |
    +------------------+---------------------------+

    Args:
        output_format: ``"dataset"`` (default) or ``"dict"``.
    """

    def report(self, data: ExperimentData) -> SmallDataset | dict[str, Any]:
        """Generate the CUPED variance-reduction report.

        Args:
            data: The experiment data container.

        Returns:
            A ``SmallDataset`` or dict with variance-reduction metrics.
        """
        ids = data.get_ids(
            CUPEDTransformer,
            searched_space=ExperimentDataEnum.analysis_tables,
        )
        table_ids = ids.get(CUPEDTransformer.__name__, {}).get(
            ExperimentDataEnum.analysis_tables.value,
            [],
        )
        if not table_ids:
            return SmallDataset.create_empty()

        table = data.analysis_tables[table_ids[0]]
        records = table.to_records()
        if not records:
            return SmallDataset.create_empty()

        rows: list[dict[str, Any]] = []
        for record in records:
            rows.append(
                {
                    "feature": record.get("feature"),
                    "variance_reduction_pct": record.get("variance_reduction_pct"),
                }
            )

        return SmallDataset.from_dict(
            rows,
            roles={
                "feature": InfoRole(str),
                "variance_reduction_pct": StatisticRole(float),
            },
        )
