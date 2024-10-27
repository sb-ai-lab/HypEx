import pytest
import pandas as pd
from hypex import AATest

def test_aa_simple(data, iterations, info_col):
    
    model = AATest(target_fields=["pre_spends", "post_spends"], info_cols=info_col)
    res, datas_dict = model.calc_uniform_tests(data, iterations=iterations, pbar=False)

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

def test_aa_group(data, iterations, info_col):
    group_cols = "industry"

    model = AATest(
        target_fields=["pre_spends", "post_spends"],
        info_cols=info_col,
        group_cols=group_cols,
    )
    res, datas_dict = model.calc_uniform_tests(data, iterations=iterations, pbar=False)

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

def test_aa_quantfields(data, iterations, info_col):
    group_cols = "industry"
    quant_field = "gender"

    model = AATest(
        target_fields=["pre_spends", "post_spends"],
        info_cols=info_col,
        group_cols=group_cols,
        quant_field=quant_field,
    )
    res, datas_dict = model.calc_uniform_tests(data, iterations=iterations, pbar=False)

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

def test_aa_process(data, iterations, info_col):
    group_cols = "industry"

    model = AATest(
        target_fields=["pre_spends", "post_spends"],
        info_cols=info_col,
        group_cols=group_cols,
    )
    model.process(data, iterations=iterations, show_plots=False, pbar=False)

def test_group_optimization(data, info_col):
    model = AATest(target_fields=["pre_spends", "post_spends"], info_cols=info_col)
    model.process(data, optimize_groups=True, iterations=5, show_plots=False, pbar=False)

def test_unbalanced_groups(data, iterations, info_col):
    test_size = 0.3
    model = AATest(target_fields=["pre_spends", "post_spends"], info_cols=info_col)
    results = model.process(
        data,
        optimize_groups=False,
        iterations=iterations,
        show_plots=False,
        test_size=test_size,
        pbar=False
    )
    assert abs(results["split_stat"]["test %"] / 100 - test_size) < 0.05

def test_aa_empty_unbalanced_process(data, iterations):
    model = AATest()
    model.process(data, iterations=iterations, show_plots=False, pbar=False, test_size=0.05)

def test_calc_sample_size(data, iterations, info_col):
    model = AATest(target_fields=["pre_spends", "post_spends"], info_cols=info_col)
    splitted_data = model.sampling_metrics(data)['data_from_experiment'][None]
    model.calc_sample_size(data=splitted_data, target_field='post_spends')