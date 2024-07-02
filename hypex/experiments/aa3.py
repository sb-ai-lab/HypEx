from hypex.analyzers.aa2 import AAScoreAnalyzer
from hypex.experiments import Experiment
from hypex.experiments.aa2 import AATest

AA_TEST = Experiment(executors=[AATest(), AAScoreAnalyzer()])

AA_TEST_WITH_STRATIFICATION = Experiment(
    executors=[AATest(stratification=True), AAScoreAnalyzer()],
)
