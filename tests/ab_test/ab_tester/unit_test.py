import pytest
import pandas as pd
import numpy as np





def test_split_ab(split_ab_result, data_size):
    """
        Should return Dict with 2 keys (test and control). 
        Values should return DataFrames with DATA_SIZE len
    """
    assert len(split_ab_result["test"]) == data_size
    assert len(split_ab_result["control"]) == data_size


def test_calc_difference(ab_test, split_ab_result, target_field, previous_value_field):
    """
        Tests 3 method calculation differences.
        ate - (average treatment effect). 
            With key 'ate' of result returns a mean of test and control group values delta. 
            The size of test and control group should be equals!
            Since we generated test and control groups with centers in 12 and 10 and str=2
            we should get values between 1 and 3
        couped - (Controlled-experiment Using Pre-Experiment Data).
            With key 'cuped' of result returns difference of means of test and control groups,
            corrected by theta value.
            Since control group means are the same and test group means have difference near 2,
            expects value between 1 and 3.
        diff_in_diff - (difference in difference)
            With key 'diff_in_diff' of result returns changing in mean values of test and control groups.
            As in couped method expected value between 1 and 3.
        

    """
    result = ab_test.calc_difference(split_ab_result, target_field, previous_value_field)
    assert 1 < result["ate"] < 3 
    assert 1 < result["cuped"] < 3
    assert 1 < result["diff_in_diff"] < 3


def test_calc_difference_with_previous_value(ab_test, data, group_field, target_field, previous_value_field):
    """
        Previous values generated with normal distribution with center 10 value.
        Means of  2 subgroups should be close each other.
    """
    ab_test.calc_difference_method = "ate"
    splitted_data = ab_test.split_ab(data, group_field)
    result = ab_test.calc_difference(splitted_data, previous_value_field)
    assert -1 < result["ate"] < 1


def test_calc_p_value(ab_test, split_ab_result, target_field, previous_value_field, alpha):
    """
         Expected significant difference in target_field between test and control.
         No expect significant difference in previous_value_field between test and control.
    """
    result = ab_test.calc_p_value(split_ab_result, target_field)
    assert result["t-test"] < alpha
    assert result["mann_whitney"] < alpha

    result = ab_test.calc_p_value(split_ab_result, previous_value_field)
    assert result["t-test"] > alpha
    assert result["mann_whitney"] > alpha


def test_execute(ab_test, data, group_field, target_field, previous_value_field, alpha, data_size):
    """
        Just tests execute method, that using calc_difference and calc_p_value functions.
    """
    result = ab_test.execute(data, target_field, group_field, previous_value_field)
    assert result["size"]["test"] == data_size
    assert result["size"]["control"] == data_size
    assert 1 < result["difference"]["ate"] < 3
    assert 1 < result["difference"]["cuped"] < 3
    assert 1 < result["difference"]["diff_in_diff"] < 3
    assert result["p-value"]["t-test"] < alpha
    assert result["p-value"]["mann_whitney"] < alpha
