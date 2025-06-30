Installation
============

Requirements
------------

* Python 3.8 or higher
* NumPy
* Pandas
* SciPy
* Scikit-learn
* Statsmodels

Basic Installation
------------------

Install HypEx using pip:

.. code-block:: bash

    pip install hypex

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/sb-ai-lab/HypEx.git
    cd HypEx
    pip install -e .

Optional Dependencies
---------------------

For additional functionality, install with extras:

.. code-block:: bash

    # For CatBoost support
    pip install hypex[cat]

    # For LightGBM support
    pip install hypex[lgbm]

    # All extras
    pip install hypex[all]
