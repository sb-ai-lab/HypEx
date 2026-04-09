API Reference
=============

.. currentmodule:: hypex

Main Classes
------------

High-level experiment runners that compose the lower-level building blocks.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   AATest
   ABTest
   HomogeneityTest
   matching.Matching

Comparators
-----------

All comparators live in :mod:`hypex.comparators`.

Hypothesis Tests
~~~~~~~~~~~~~~~~

Backend-adaptive tests — automatically select the best implementation for the
active dataset backend (pandas vs. Spark).

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   comparators.TTest
   comparators.Chi2Test
   comparators.KSTest
   comparators.UTest

Group Metrics
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   comparators.GroupDifference
   comparators.GroupSizes
   comparators.PSI
   comparators.MahalanobisDistance

Power & Sample Size
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   comparators.PowerTesting
   comparators.MDEBySize

Base Classes
~~~~~~~~~~~~

Extend these to implement custom comparators.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   comparators.Comparator
   comparators.StatHypothesisTesting

Splitters
---------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   splitters.AASplitter
   splitters.AASplitterWithStratification

Transformers
------------

Pre-processing steps applied to :class:`~hypex.dataset.Dataset` objects.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   transformers.NaFiller
   transformers.NanFilter
   transformers.OutliersFilter
   transformers.ConstFilter
   transformers.CorrFilter
   transformers.CVFilter
   transformers.DummyEncoder
   transformers.TypeCaster
   transformers.CategoryAggregator
   transformers.CUPEDTransformer
   transformers.Shuffle

Experiments
-----------

Pipeline runners that chain executors over :class:`~hypex.dataset.ExperimentData`.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   experiments.Experiment
   experiments.GroupExperiment
   experiments.OnRoleExperiment
   experiments.CycledExperiment

Reporters
---------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   reporters.Reporter
   reporters.DatasetReporter
   reporters.DictReporter
   reporters.HomoDatasetReporter
   reporters.HomoDictReporter

Operators
---------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   operators.SMD

Calculators
-----------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   executor.MinSampleSize

Dataset Module
--------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: autosummary/class.rst

   dataset.Dataset
   dataset.ExperimentData

Roles
~~~~~

Roles tag columns with semantic meaning so executors can locate them
without hard-coding column names.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   dataset.TargetRole
   dataset.TreatmentRole
   dataset.GroupingRole
   dataset.FeatureRole
   dataset.InfoRole
   dataset.PreTargetRole
   dataset.StatisticRole
   dataset.StratificationRole
   dataset.FilterRole
   dataset.AdditionalTargetRole
   dataset.AdditionalFeatureRole
   dataset.AdditionalGroupingRole
   dataset.AdditionalMatchingRole
   dataset.AdditionalTreatmentRole
