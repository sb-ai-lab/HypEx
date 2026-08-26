# starts with HYPEX-dir: PYTHONPATH=$(pwd) pytest
import numpy as np
import pandas as pd
import pytest

from hypex import Matching
from hypex.dataset import (
    Dataset,
    FeatureRole,
    GroupingRole,
    InfoRole,
    TargetRole,
    TreatmentRole,
)


@pytest.fixture
def grouped_data():
    rng = np.random.default_rng(11)
    n = 600
    df = pd.DataFrame(
        {
            "user_id": range(n),
            "accept_date": rng.choice(
                ["2025-07-31", "2025-08-31", "2025-09-30"], n
            ),
            "treat": rng.integers(0, 2, n),
            "age": rng.normal(40, 5, n),
            "income": rng.normal(100, 20, n),
            "target": rng.normal(size=n),
        }
    )
    return Dataset(
        roles={
            "user_id": InfoRole(int),
            "accept_date": GroupingRole(str),
            "treat": TreatmentRole(int),
            "age": FeatureRole(),
            "income": FeatureRole(),
            "target": TargetRole(),
        },
        data=df,
    )


def _groups_in_resume(resume):
    return {str(column).split(" ")[0] for column in resume.columns}


def test_group_matching_keeps_every_group(grouped_data):
    result = Matching(group_match=True).execute(grouped_data)
    groups = _groups_in_resume(result.resume)
    assert {"2025-07-31", "2025-08-31", "2025-09-30"} <= groups
    assert result.indexes.shape[0] == len(grouped_data)


def test_group_matching_with_scalar_groupby_keys(grouped_data, monkeypatch):
    """pandas < 2.0 yields a scalar group key when grouping by a 1-element list.

    Taking ``group[0]`` unconditionally then sliced the first character out of
    the key, collapsing every group into one ('2') and silently dropping all
    but the last group from the resume.
    """
    original_groupby = Dataset.groupby

    def scalar_key_groupby(self, by, *args, **kwargs):
        return [
            (group[0] if isinstance(group, tuple) and len(group) == 1 else group, data)
            for group, data in original_groupby(self, by, *args, **kwargs)
        ]

    monkeypatch.setattr(Dataset, "groupby", scalar_key_groupby)

    result = Matching(group_match=True).execute(grouped_data)
    groups = _groups_in_resume(result.resume)
    assert {"2025-07-31", "2025-08-31", "2025-09-30"} <= groups
    assert result.indexes.shape[0] == len(grouped_data)
