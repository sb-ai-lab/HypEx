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

    # For Spark backend support (also requires Java 8, 11, or 17 at runtime)
    pip install "hypex[spark]"

    # For CatBoost support
    pip install "hypex[cat]"

    # For LightGBM support
    pip install "hypex[lgbm]"

    # All extras
    pip install "hypex[all]"

The base installation supports pandas and does not install PySpark. The
``spark`` extra installs PySpark 3.5.1. If you request Spark functionality
without it, HypEx raises an error with the installation command.

For a full development environment, install all extras in editable mode with
``pip install -e ".[all]"``.
