from typing import Optional, Dict

from hypex.comparators.abstract import GroupComparator
from hypex.dataset import TreatmentRole
from hypex.experiments.aa import ONE_AA_TEST
from hypex.experiments.base_complex import ParamsExperiment
from hypex.reporters import DatasetReporter, OneAADictReporter
from hypex.splitters import AASplitter
from hypex.utils import SpaceEnum


class AATest(ParamsExperiment):
    def __init__(self, random_states=range(2000), additional_params: Optional[Dict] = None):
        additional_params = additional_params or {}
        params = {
            AASplitter: {"random_state": random_states},
            GroupComparator: {
                "grouping_role": [TreatmentRole()],
                "space": [SpaceEnum.additional],
            },
        }
        params.update(additional_params)
        super().__init__(
            executors=[ONE_AA_TEST],
            reporter=DatasetReporter(OneAADictReporter(front=False)),
            params=params,
        )
