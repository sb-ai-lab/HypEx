"""Regression tests for the parameters AATest passes down to the splitter."""

import numpy as np
import pandas as pd

from hypex import AATest
from hypex.dataset import Dataset, InfoRole, TargetRole


def _make_data(n: int = 300, seed: int = 3) -> Dataset:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "pre_spends": rng.normal(100, 10, n),
        }
    )
    return Dataset(
        roles={"user_id": InfoRole(), "pre_spends": TargetRole()},
        data=df,
    )


def test_sample_size_together_with_groups_sizes():
    """The second pass added by sample_size used to get shifted arguments.

    ``groups_sizes`` landed in the ``additional_params`` slot of
    ``_prepare_params`` and blew up on ``params.update(list)``.
    """
    result = AATest(
        sample_size=0.3,
        groups_sizes=[0.5, 0.2, 0.3],
        random_states=range(3),
    ).execute(_make_data())

    groups = set(result.best_split["split"].get_values(column="split"))
    assert groups == {"control", "test_1", "test_2"}
