"""Reporters for CUPAC variance reduction results."""

from __future__ import annotations

from typing import Any

from ..dataset import ExperimentData, SmallDataset
from ..dataset.roles import InfoRole, StatisticRole
from ..ml.cupac import CUPACExecutor
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum
from .abstract import Reporter


class CupacReporter(Reporter):
    """Extracts CUPAC model-selection and variance-reduction metrics.

    Produces two datasets:

    **variance_reductions** — one row per target:

    +--------+------------+-------------------------+--------------------------+
    | target | best_model | variance_reduction_cv   | variance_reduction_real  |
    +========+============+=========================+==========================+
    | y      | ridge      | 38.2                    | 41.5                     |
    +--------+------------+-------------------------+--------------------------+

    **feature_importances** — one row per (target, feature):

    +--------+----------+------------+------------+
    | target | feature  | importance | model      |
    +========+==========+============+============+
    | y      | x1       | 0.85       | ridge      |
    +--------+----------+------------+------------+

    Args:
        output_format: ``"dataset"`` (default) or ``"dict"``.
    """

    def report(self, data: ExperimentData) -> dict[str, SmallDataset | None]:
        """Generate the CUPAC results report.

        Args:
            data: The experiment data container.

        Returns:
            A dict with ``"variance_reductions"`` and
            ``"feature_importances"`` keys.
        """
        ids = data.get_ids(
            CUPACExecutor,
            searched_space=ExperimentDataEnum.analysis_tables,
        )
        all_ids = ids.get(CUPACExecutor.__name__, {}).get(
            ExperimentDataEnum.analysis_tables.value,
            [],
        )

        # Separate main reports from importance reports
        main_ids = [i for i in all_ids if not i.endswith("importances")]
        imp_ids = [i for i in all_ids if i.endswith("importances")]

        vr_ds = self._extract_variance_reductions(data, main_ids)
        fi_ds = self._extract_feature_importances(data, imp_ids)

        return {
            "variance_reductions": vr_ds,
            "feature_importances": fi_ds,
        }

    @staticmethod
    def _extract_variance_reductions(
        data: ExperimentData,
        main_ids: list[str],
    ) -> SmallDataset | None:
        """Aggregate per-target CUPAC reports into a single table.

        Args:
            data: The experiment data container.
            main_ids: Analysis-table IDs for main CUPAC reports.

        Returns:
            A ``SmallDataset`` or ``None`` if no reports found.
        """
        if not main_ids:
            return None

        rows: list[dict[str, Any]] = []
        for aid in main_ids:
            table = data.analysis_tables.get(aid)
            if table is None or table.is_empty():
                continue
            records = table.to_records()
            if not records:
                continue
            row = records[0]
            # Extract target name from the composite ID
            target = aid.split(ID_SPLIT_SYMBOL)[-1]
            rows.append(
                {
                    "target": target,
                    "best_model": row.get("cupac_best_model"),
                    "variance_reduction_cv": row.get("cupac_variance_reduction_cv"),
                    "variance_reduction_real": row.get("cupac_variance_reduction_real"),
                }
            )

        if not rows:
            return None

        return SmallDataset.from_dict(
            rows,
            roles={
                "target": InfoRole(str),
                "best_model": InfoRole(str),
                "variance_reduction_cv": StatisticRole(float),
                "variance_reduction_real": StatisticRole(float),
            },
        )

    @staticmethod
    def _extract_feature_importances(
        data: ExperimentData,
        imp_ids: list[str],
    ) -> SmallDataset | None:
        """Aggregate per-target feature importances into a single table.

        Args:
            data: The experiment data container.
            imp_ids: Analysis-table IDs for importance reports.

        Returns:
            A ``SmallDataset`` or ``None`` if no importances found.
        """
        if not imp_ids:
            return None

        rows: list[dict[str, Any]] = []
        for aid in imp_ids:
            table = data.analysis_tables.get(aid)
            if table is None or table.is_empty():
                continue
            records = table.to_records()
            if not records:
                continue
            row = records[0]
            # Extract target from ID: CUPACExecutor┆hash┆target┆importances
            parts = aid.split(ID_SPLIT_SYMBOL)
            target = parts[-2] if len(parts) >= 2 else "unknown"
            # Find the corresponding main report to get the model name
            main_id = ID_SPLIT_SYMBOL.join(parts[:-1])
            main_table = data.analysis_tables.get(main_id)
            model = None
            if main_table and not main_table.is_empty():
                main_records = main_table.to_records()
                if main_records:
                    model = main_records[0].get("cupac_best_model")

            for feature, importance in row.items():
                rows.append(
                    {
                        "target": target,
                        "feature": feature,
                        "importance": importance,
                        "model": model,
                    }
                )

        if not rows:
            return None

        return SmallDataset.from_dict(
            rows,
            roles={
                "target": InfoRole(str),
                "feature": InfoRole(str),
                "importance": StatisticRole(float),
                "model": InfoRole(str),
            },
        )
