from hypex.experiment.experiment import Experiment
from hypex.splitters.aa_splitter import AASplitter
from hypex.comparators.hypothesis_testing import TTest, KSTest

SingleAASplit = Experiment(
    [
        AASplitter(),
        TTest(),
        KSTest()
    ]
)