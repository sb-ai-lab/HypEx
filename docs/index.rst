.. HypEx documentation master file

Welcome to HypEx's documentation!
==================================

HypEx is a fast and customizable framework for Causal Inference.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api_reference
   examples

Installation
------------

.. code-block:: bash

   pip install hypex

Quick Start
-----------

.. code-block:: python

   from hypex import ABTest, AATest, Matching
   from hypex.dataset import Dataset, TargetRole, TreatmentRole

   # Your code here

API Reference
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:
   :template: autosummary/module.rst

   hypex

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`