from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_csv() -> str:
    """Path to the committed synthetic dataset shared by the tests.

    The file is checked in on purpose: regenerating it would shift with the
    numpy version and make the golden A/A and A/B verdicts unstable.
    """
    return str(Path(__file__).parent / "data.csv")
