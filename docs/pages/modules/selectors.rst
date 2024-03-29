.. currentmodule:: hypex.selectors

Selectors
======================

Classes for feature selectors

Selector classes
-----------------------------

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    feature_selector.FeatureSelector
    outliers_filter.OutliersFilter
    spearman_filter.SpearmanFilter

.. warnings::
        FeatureSelector does not rule out the possibility of overlooked features,
        the complex impact of features on target description, or
        the significance of features from a business logic perspective.

.. currentmodule:: hypex.selectors.base_filtration
.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    const_filtration
    nan_filtration
