# HypEx: Streamlined Causal Inference and AB Testing

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

## Quick Start

Explore usage examples and tutorials [here](https://github.com/sb-ai-lab/Hypex/blob/master/examples/tutorials/).

## Installation

```bash
pip install hypex
```

## Quick Example

```python
from hypex import Matcher

# Define your data and parameters
data = pd.DataFrame()
treatment_column = "treatment"
outcome_column = ['target1', 'target2']

# Create and run the experiment
matcher = Matcher(data, treatment_column, outcome_column)
results, quality_results, df_matcher = matcher.estimate()
```

## Documentation

For more detailed information about the library and its features, visit
our [documentation on ReadTheDocs](https://hypex.readthedocs.io/en/latest/).

You'll find comprehensive guides and tutorials that will help you get started with HypEx, as well as detailed API
documentation for advanced use cases.

## Conclusion

HypEx stands as an indispensable resource for data analysts and researchers delving into causal inference and AB
testing. With its automated capabilities, sophisticated matching techniques, and thorough validation procedures, HypEx
is poised to unravel causal relationships in complex datasets with unprecedented speed and precision.
