import pytest
import pandas as pd
from hypex import AATest



def test_aa_simple(simple_AATest_results, data, iterations, info_col):
    """
        old test
    """
    res, datas_dict = simple_AATest_results

    assert isinstance(res, pd.DataFrame), "Metrics are not dataframes"
    assert res.shape[0] == iterations, (
        "Metrics dataframe contains more or less rows with random states "
        "(#rows should be equal #of experiments"
    )
    assert isinstance(datas_dict, dict), "Result is not dict"
    assert len(datas_dict) == iterations, "# of dataframes is not equal # of iterations"
    assert all(data.columns) == all(
        datas_dict[0].drop(columns=["group"]).columns
    ), "Columns in the result are not the same as columns in initial data "

def test_aa_group(grouped_AATest_results, iterations, group_col, data):
    """
        old test
    """
    res, datas_dict = grouped_AATest_results
    assert isinstance(res, pd.DataFrame), "Metrics are not dataframes"
    assert res.shape[0] == iterations, (
        "Metrics dataframe contains more or less rows with random states "
        "(#rows should be equal #of experiments"
    )
    assert isinstance(datas_dict, dict), "Result is not dict"
    assert len(datas_dict) == iterations, "# of dataframes is not equal # of iterations"
    assert all(data.columns) == all(datas_dict[0].drop(columns=["group"]).columns), (
        "Columns in the result are not " "the same as columns in initial " "data "
    )

def test_aa_quantfields(data, iterations, quanted_grouped_AATest_results):

    res, datas_dict = quanted_grouped_AATest_results
    assert isinstance(res, pd.DataFrame), "Metrics are not dataframes"
    assert res.shape[0] == iterations, (
        "Metrics dataframe contains more or less rows with random states "
        "(#rows should be equal #of experiments"
    )
    assert isinstance(datas_dict, dict), "Result is not dict"
    assert len(datas_dict) == iterations, "# of dataframes is not equal # of iterations"
    assert all(data.columns) == all(datas_dict[0].drop(columns=["group"]).columns), (
        "Columns in the result are not " "the same as columns in initial " "data "
    )

def test_result_rows_eq_num_iterations(simple_AATest_results, grouped_AATest_results,quanted_grouped_AATest_results,iterations):
    """
        res variable should return the same number of rows as iterations count
    """
    res, datas_dict = simple_AATest_results
    assert res.shape[0] == iterations, (
        "simple_AATest_results: Metrics dataframe contains more or less rows with random states "
        "(#rows should be equal #of experiments"
    ) 
    res, datas_dict = grouped_AATest_results
    assert res.shape[0] == iterations, (
        "grouped_AATest_results: Metrics dataframe contains more or less rows with random states "
        "(#rows should be equal #of experiments)"
    ) 
    res, datas_dict = quanted_grouped_AATest_results
    assert res.shape[0] == iterations, (
        "quanted_grouped_AATest_results: Metrics dataframe contains more or less rows with random states "
        "(#rows should be equal #of experiments)"
    ) 

def test_datas_dict_rows_eq_num_iterations(simple_AATest_results, grouped_AATest_results,quanted_grouped_AATest_results,iterations):
    """
        datas_dict variable should return the same number of rows as iterations count
    """
    res, datas_dict = simple_AATest_results
    assert len(datas_dict) == iterations, "simple_AATest_results: # of dataframes is not equal # of iterations"

    res, datas_dict = grouped_AATest_results
    assert len(datas_dict) == iterations, "grouped_AATest_results: # of dataframes is not equal # of iterations"

    res, datas_dict = quanted_grouped_AATest_results
    assert len(datas_dict) == iterations, "quanted_grouped_AATest_results: # of dataframes is not equal # of iterations"

def test_datas_dict_columns_eq_initial_data_columns(simple_AATest_results,grouped_AATest_results,quanted_grouped_AATest_results, data):
    """
        data_dict variable should have the same columns as in initial data
    """
    res, datas_dict = simple_AATest_results
    assert all(data.columns) == all(
        datas_dict[0].drop(columns=["group"]).columns
    ), "simple_AATest_results: Columns in the result are not the same as columns in initial data "

    res, datas_dict = grouped_AATest_results
    assert all(data.columns) == all(
        datas_dict[0].drop(columns=["group"]).columns
    ), "grouped_AATest_results: Columns in the result are not the same as columns in initial data "

    res, datas_dict = quanted_grouped_AATest_results
    assert all(data.columns) == all(
        datas_dict[0].drop(columns=["group"]).columns
    ), "quanted_grouped_AATest_results: Columns in the result are not the same as columns in initial data "


def test_aa_process(data, grouped_AATest, iterations):
    """
        Just tests for some error
    """
    model = grouped_AATest
    model.process(data, iterations=iterations, show_plots=False, pbar=False)

def test_group_optimization(simple_AATest,data):
    """
        Just tests some error when optimize_groups is True
    """
    model = simple_AATest
    model.process(data, optimize_groups=True, iterations=5, show_plots=False, pbar=False)

def test_unbalanced_groups(simple_AATest, data, iterations, info_col):
    """
        Just tests when groups are unbalanced
    """
    test_size = 0.3
    model = simple_AATest
    results = model.process(
        data,
        iterations=iterations,
        show_plots=False,
        test_size=test_size,
        pbar=False
    )
    assert abs(results["split_stat"]["test %"] / 100 - test_size) < 0.05

def test_aa_empty_unbalanced_process(data, iterations):
    """
        Just tests  the process when test_size too small (e.g. 5% vs 95%)
    """
    model = AATest()
    model.process(data, iterations=iterations, show_plots=False, pbar=False, test_size=0.05)

def test_calc_sample_size(simple_AATest, data, iterations, info_col):
    """
    """
    model = simple_AATest
    splitted_data = model.sampling_metrics(data)['data_from_experiment'][None]
    model.calc_sample_size(data=splitted_data, target_field='post_spends')