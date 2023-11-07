.. currentmodule:: hypex.algorithms

Algorithms
===================

Classes for search Nearest Neighbors in a quick way

Matching classes
------------------------

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    FaissMatcher
    MatcherNoReplacement

Utility Functions
------------------------
.. currentmodule:: hypex.algorithms.faiss_matcher
.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    map_func
    f2
    _get_index
    _transform_to_np
    calc_atx_var
    calc_atc_se
    conditional_covariance
    calc_att_se
    calc_ate_se
    pval_calc
    scaled_counts
    bias_coefs
    bias
    f3

.. currentmodule:: hypex.algorithms.no_replacement_matching
.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    optimally_match_distance_matrix
    _ensure_array_columnlike



