# HypEx: Advanced Causal Inference and AB Testing Toolkit

![Last release](https://img.shields.io/badge/pypi-v1.0.2-darkgreen)
[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/hypexchat)
![Pypi downloads](https://img.shields.io/badge/downloads-70K-1E782B)
![Python versions](https://img.shields.io/badge/python-3.8_|_3.9_|_3.10_|_3.11_|_3.12-blue)
![Pypi downloads\month](https://img.shields.io/badge/downloads\month->20K-1E782B)

# Moving to a New Architecture! ⚠️

## ❗ ВАЖНОЕ ОБНОВЛЕНИЕ: HypEx переходит на новую архитектуру! ❗

## 🔥 What's changing? / Что изменится?

New interface! Import paths, class methods, and overall API structure will be different.

Old version (0.1.10) will no longer be supported.

New tutorials are available to help you migrate: Check them
out [here](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials).

Try the new version now by installing the beta release:

```
pip install --upgrade hypex
```

Prefer the old version? You can still use it, but it won't receive updates:

```
pip install hypex==0.1.10
```

🔗 Learn more: [Telegram Chat](t.me/hypexchat)

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
- **Feature Selection**: Employs LGBM and Catboost feature selection to pinpoint the most impactful features for causal
  analysis.
- **AB Testing Suite**: Features a suite of AB testing tools for comprehensive hypothesis evaluation.
- **Stratification support**: Stratify groups for nuanced analysis
- **Weights support**:  Empower your analysis by assigning custom weights to features, enhancing the matching precision
  to suit your specific research needs

## Warnings

Some functions in HypEx can facilitate solving specific auxiliary tasks but cannot automate decisions on experiment
design. Below, we will discuss features that are implemented in HypEx but do not automate the design of experiments.

**Note:** For Matching, it's recommended not to use more than 7 features as it might result in the curse of
dimensionality, making the results unrepresentative.

## Installation

```bash
pip install -U hypex
```

## Quick start

Explore usage examples and tutorials [here](https://github.com/sb-ai-lab/Hypex/blob/master/examples/tutorials/).

### Matching example

```python
from hypex.dataset import Dataset, InfoRole, TreatmentRole, TargetRole, DefaultRole, FeatureRole
from hypex import Matching

data = Dataset(
    roles={
        "user_id": InfoRole(int),  # InfoRole for ID
        "treat": TreatmentRole(int),  # TreatmentRole is for identify user group (control or target)
        "post_spends": TargetRole(float)  # TargetRole for Target :)
    },
    data="data.csv",
    default_role=FeatureRole(),  # All remaining columns will be of type FeatureRole (searching for similar ones)
)

test = Matching()  # Classic Matching (maha distance + full metrics)
test = Matching(metric="att")  # Calc only ATT
test = Matching(distance="l2")  # Choose distance here

result = test.execute(data)
result.resume  # Resume of results 
result.full_data  # old df_matched. Wide df with pairs
result.indexes  # Only indexed pairs (good for join)

```  
More about Matching [here](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials/MatchingTutorial.ipynb)

### AA-test example

```python
from hypex.dataset import Dataset, InfoRole, TreatmentRole, TargetRole, StratificationRole
from hypex import AATest

data = Dataset(
    roles={
        "user_id": InfoRole(int),  # InfoRole for ID.
        "pre_spends": TargetRole(),  # TargetRole for check homogeneity
        "post_spends": TargetRole(),  # TargetRole for check homogeneity
        "gender": StratificationRole(str)  # StratificationRole for strata
    }, data="data.csv",
)

aa = AATest(n_iterations=10)
res = aa.execute(data)

res.resume  # Resume for all test
res.aa_score  # AA score 
res.best_split  # The best homogeneity split
res.best_split_statistic  # Statistics for best split 
```
More about AA test [here](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials/AATestTutorial.ipynb)

### AB-test example

```python
from hypex.dataset import Dataset, InfoRole, TreatmentRole, TargetRole
from hypex import ABTest

data = Dataset(
    roles={
        "user_id": InfoRole(int),  # InfoRole use for ID
        "treat": TreatmentRole(),  # TreatmentRole is for identify user group (control or target)
        "pre_spends": TargetRole(),  # Target for A/B(n) Tests
        "post_spends": TargetRole(),  # Target for A/B(n) Tests
    }, data="data.csv",
)

test = ABTest()  # Classic A/B test
test = ABTest(multitest_method="bonferroni")  # A/Bn test with Bonferroni corrections
test = ABTest(additional_tests=['t-test', 'u-test', 'chi2-test'])  # Use can choose tests

result = test.execute(data)
result.resume  # Resume of results
```
More about A/B test [here](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials/ABTestTutorial.ipynb)

## Documentation

For more detailed information about the library and its features, visit
our [documentation on ReadTheDocs](https://hypex.readthedocs.io/en/latest/).

You'll find comprehensive guides and tutorials that will help you get started with HypEx, as well as detailed API
documentation for advanced use cases.

## Contributions

Join our vibrant community! For guidelines on contributing, reporting issues, or seeking support, please refer to
our [Contributing Guidelines](https://github.com/sb-ai-lab/Hypex/blob/master/.github/CONTRIBUTING.md).

## More Information & Resources

[Habr (ru)](https://habr.com/ru/companies/sberbank/articles/778774/) - discover how HypEx is revolutionizing causal
inference in various fields.      
[A/B testing seminar](https://www.youtube.com/watch?v=B9BE_yk8CjA&t=53s&ab_channel=NoML) - Seminar in NoML about
matching and A/B testing       
[Matching with HypEx: Simple Guide](https://www.kaggle.com/code/kseniavasilieva/matching-with-hypex-simple-guide) -
Simple matching guide with explanation           
[Matching with HypEx: Grouping](https://www.kaggle.com/code/kseniavasilieva/matching-with-hypex-grouping) - Matching
with grouping guide    
[HypEx vs Causal Inference and DoWhy](https://www.kaggle.com/code/kseniavasilieva/hypex-vs-causal-inference-and-dowhy) -
discover why HypEx is the best solution for causal inference           
[HypEx vs Causal Inference and DoWhy: part 2](https://www.kaggle.com/code/kseniavasilieva/hypex-vs-causal-inference-part-2) -
discover why HypEx is the best solution for causal inference

### Testing different libraries for the speed of matching

Visit [this](https://www.kaggle.com/code/kseniavasilieva/hypex-vs-causal-inference-part-2) notebook ain Kaggle and
estimate results by yourself.

| Group size             | 32 768 | 65 536 | 131 072 | 262 144 | 524 288 | 1 048 576 | 2 097 152 | 4 194 304 |
|------------------------|--------|--------|---------|---------|---------|-----------|-----------|-----------|
| Causal Inference       | 46s    | 169s   | None    | None    | None    | None      | None      | None      |
| DoWhy                  | 9s     | 19s    | 40s     | 77s     | 159s    | 312s      | 615s      | 1 235s    |
| HypEx with grouping    | 2s     | 6s     | 16s     | 42s     | 167s    | 509s      | 1 932s    | 7 248s    |
| HypEx without grouping | 2s     | 7s     | 21s     | 101s    | 273s    | 982s      | 3 750s    | 14 720s   |

## Join Our Community

Have questions or want to discuss HypEx? Join our [Telegram chat](https://t.me/HypExChat) and connect with the community
and the developers.

## Conclusion

HypEx stands as an indispensable resource for data analysts and researchers delving into causal inference and AB
testing. With its automated capabilities, sophisticated matching techniques, and thorough validation procedures, HypEx
is poised to unravel causal relationships in complex datasets with unprecedented speed and precision.

##                                                                                
