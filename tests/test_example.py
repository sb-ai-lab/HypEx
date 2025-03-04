from typing import Any

import pytest


@pytest.mark.parametrize("test_input, expected", [(1, 1), (2, 2), (3, 3)])
def test_example(test_input: Any, expected: Any) -> None:
    """
    Tests if the input values are equal to the expected values.

    This test uses parametrization to check multiple pairs of values.
    It ensures that each input argument is equal to its expected value.

    Args:
        test_input: The input value for the test.
        expected: The expected value to compare against the input.

    Returns:
        None. The test simply asserts the condition.
    """
    assert test_input == expected, f"Expected {expected}, got {test_input}"
    return
