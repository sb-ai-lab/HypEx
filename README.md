# HypEx: Advanced Causal Inference and AB Testing Toolkit

[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/hypexchat)

## Introduction

HypEx (Hypotheses and Experiments) is a comprehensive library crafted to streamline the causal inference and AB testing
processes in data analytics. Developed for efficiency and effectiveness, HypEx employs Rubin's Causal Model (RCM) for
matching closely related pairs, ensuring equitable group comparisons when estimating treatment effects.

Boasting a fully automated pipeline, HypEx adeptly calculates the Average Treatment Effect (ATE), Average Treatment
Effect on the Treated (ATT), and Average Treatment Effect on the Control (ATC). It offers a standardized interface for
executing these estimations, providing insights into the impact of interventions across various population subgroups.

Beyond causal inference, HypEx is equipped with robust AB testing tools, including Difference-in-Differences (
Diff-in-Diff) and CUPED methods, to rigorously test hypotheses and validate experimental results.

## Features

- **Faiss KNN Matching**: Utilizes Faiss for efficient and precise nearest neighbor searches, aligning with RCM for
  optimal pair matching.
- **Data Filters**: Built-in outlier and Spearman filters ensure data quality for matching.
- **Result Validation**: Offers multiple validation methods, including random treatment, feature, and subset
  validations.
- **Data Tests**: Incorporates SMD, KS, PSI, and Repeats tests to affirm the robustness of effect estimations.
- **Automated Feature Selection**: Employs LGBM feature selection to pinpoint the most impactful features for causal
  analysis.
- **AB Testing Suite**: Features a suite of AB testing tools for comprehensive hypothesis evaluation.
- **Stratification support**: Stratify groups for nuanced analysis
- **Weights support**:  Empower your analysis by assigning custom weights to features, enhancing the matching precision
  to suit your specific research needs

## Quick Start

Explore usage examples and tutorials [here](https://github.com/sb-ai-lab/Hypex/blob/master/examples/tutorials/).

## Installation

```bash
pip install hypex
```

## Quick start

### Matching example

```python
from hypex import Matcher
from hypex.utils.tutorial_data_creation import create_test_data

# Define your data and parameters
df = create_test_data(rs=42, na_step=45, nan_cols=['age', 'gender'])

info_col = ['user_id']
outcome = 'post_spends'
treatment = 'treat'
model = Matcher(input_data=df, outcome=outcome, treatment=treatment, info_col=info_col)
results, quality_results, df_matched = model.estimate()
```

### AA-test example

```python
from hypex import AATest
from hypex.utils.tutorial_data_creation import create_test_data

data = create_test_data(rs=52, na_step=10, nan_cols=['age', 'gender'])

info_cols = ['user_id', 'signup_month']
target = ['post_spends', 'pre_spends']

experiment = AATest(info_cols=info_cols, target_fields=target)
results = experiment.process(data, iterations=1000)
results.keys()
```

### AB-test example

```python
from hypex import ABTest
from hypex.utils.tutorial_data_creation import create_test_data

data = create_test_data(rs=52, na_step=10, nan_cols=['age', 'gender'])

model = ABTest()
results = model.execute(
    data=data, 
    target_field='post_spends', 
    target_field_before='pre_spends', 
    group_field='group'
)

model.show_beautiful_result()
```

## Documentation

For more detailed information about the library and its features, visit
our [documentation on ReadTheDocs](https://hypex.readthedocs.io/en/latest/).

You'll find comprehensive guides and tutorials that will help you get started with HypEx, as well as detailed API
documentation for advanced use cases.

## Community and Contributions

Join our vibrant community! For guidelines on contributing, reporting issues, or seeking support, please refer to
our [Contributing Guidelines](https://github.com/sb-ai-lab/Hypex/blob/master/.github/CONTRIBUTING.md).

## Success Stories

Discover how HypEx is revolutionizing causal inference in various fields. Check out our featured article
on [Habr (ru)](https://habr.com/ru/companies/sberbank/articles/778774/).

## Join Our Community

Have questions or want to discuss HypEx? Join our [Telegram chat](https://t.me/HypExChat) and connect with the community
and the developers.

## Conclusion

HypEx stands as an indispensable resource for data analysts and researchers delving into causal inference and AB
testing. With its automated capabilities, sophisticated matching techniques, and thorough validation procedures, HypEx
is poised to unravel causal relationships in complex datasets with unprecedented speed and precision.
