Quick Start Guide
=================

This guide will help you get started with HypEx.

Basic Usage
-----------

A/B Testing
~~~~~~~~~~~

.. code-block:: python

    from hypex import ABTest
    from hypex.dataset import Dataset, TargetRole, TreatmentRole
    import pandas as pd

    # Load your data
    df = pd.read_csv('your_data.csv')

    # Create dataset with roles
    data = Dataset(
        roles={
            'conversion': TargetRole(),
            'group': TreatmentRole(),
            'feature1': FeatureRole(),
            'feature2': FeatureRole()
        },
        data=df
    )

    # Run A/B test
    ab_test = ABTest()
    results = ab_test.execute(data)

    # View results
    print(results.resume)

A/A Testing
~~~~~~~~~~~

.. code-block:: python

    from hypex import AATest

    # Run A/A test to check for sample ratio mismatch
    aa_test = AATest(
        n_iterations=100,
        stratification=True
    )
    results = aa_test.execute(data)

    # Check if splits are good
    print(results.resume)

Matching
~~~~~~~~

.. code-block:: python

    from hypex import Matching

    # Perform matching analysis
    matching = Matching(
        distance="mahalanobis",
        metric="att"
    )
    results = matching.execute(data)

    # View matched pairs and treatment effects
    print(results.resume)
