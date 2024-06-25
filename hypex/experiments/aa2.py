from typing import Optional, Dict

from hypex.comparators.abstract import GroupComparator
from hypex.dataset import TreatmentRole
from hypex.experiments.aa import ONE_AA_TEST, ONE_AA_TEST_WITH_STRATIFICATION
from hypex.experiments.base_complex import ParamsExperiment
from hypex.reporters import DatasetReporter, OneAADictReporter
from hypex.splitters import AASplitter
from hypex.utils import SpaceEnum


class AATest(ParamsExperiment):
    def __init__(
        self,
        stratification: bool = False,
        random_states=range(2000),
        control_size: float = 0.5,
        additional_params: Optional[Dict] = None,
    ):
        additional_params = additional_params or {}
        params = {
            AASplitter: {"random_state": random_states, "control_size": [control_size]},
            GroupComparator: {
                "grouping_role": [TreatmentRole()],
                "space": [SpaceEnum.additional],
            },
        }
        inner_test = (
            ONE_AA_TEST if not stratification else ONE_AA_TEST_WITH_STRATIFICATION
        )
        params.update(additional_params)
        super().__init__(
            executors=[inner_test],
            reporter=DatasetReporter(OneAADictReporter(front=False)),
            params=params,
        )
