from typing import Dict

import pandas as pd

from hypex.experiments.homogeneity import HOMOGENEITY_TEST
from hypex.dataset import Dataset, ExperimentData, TreatmentRole, TargetRole
from hypex.reporters.homo import HomoDatasetReporter

ROLE_MAPPING = {"treatment": TreatmentRole(), "target": TargetRole()}


def prepare_experiment_data(df: pd.DataFrame) -> ExperimentData:
    roles = {column: ROLE_MAPPING[role] for column, role in df.columns_metadata}
    return ExperimentData(Dataset(roles=roles, data=df))


def get_semaphore(result_df: pd.DataFrame):
    return "OK" if (result_df["pass"] == "OK").all() else "NOT OK"


def main(df: pd.DataFrame) -> Dict:
    ed = prepare_experiment_data(df)
    ed = HOMOGENEITY_TEST.execute(ed)
    result = HomoDatasetReporter().report(ed).data.drop(columns=["group"])
    return {"result": result, "semaphore": get_semaphore(result)}
