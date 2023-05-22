import abc
from typing import List

from ConfigSpace.hyperparameters import Hyperparameter

from src.calibration.set_constructing_functions.set_constructing_function import PredictionSetConstructingFunction, \
    PredictionIntervalConstructingFunctionWithCQR, PredictionIntervalMatrixConstructingFunctionWithMeanAndStd, \
    ThetaAggregation
from src.calibration.stretching_functions.stretching_functions_factory import StretchingFunctionFactory


class PredictionSetConstructingFunctionFactory(abc.ABC):
    def __init__(self, stretching_function_factory: StretchingFunctionFactory):
        self.stretching_function_factory = stretching_function_factory

    @abc.abstractmethod
    def generate(self, **params) -> PredictionSetConstructingFunction:
        pass

    @abc.abstractmethod
    def get_hyperparameters(self) -> List[Hyperparameter]:
        pass

    @property
    def name(self):
        return f"{self.name_aux}_{self.stretching_function_factory.name}"

    @property
    @abc.abstractmethod
    def name_aux(self):
        pass


class PredictionIntervalConstructingFunctionWithCQRFactory(PredictionSetConstructingFunctionFactory):

    def __init__(self, stretching_function_factory: StretchingFunctionFactory):
        super().__init__(stretching_function_factory)

    def generate(self, **params) -> PredictionSetConstructingFunction:
        stretching_function = self.stretching_function_factory.generate(**params)
        return PredictionIntervalConstructingFunctionWithCQR(stretching_function=stretching_function)

    def get_hyperparameters(self) -> List[Hyperparameter]:
        return self.stretching_function_factory.get_hyperparameters()

    @property
    def name_aux(self):
        return "cqr"


class PredictionIntervalMatrixConstructingFunctionWithMeanAndStdFactory(PredictionSetConstructingFunctionFactory):

    def __init__(self, stretching_function_factory: StretchingFunctionFactory, aggregation: ThetaAggregation):
        super().__init__(stretching_function_factory)
        self.aggregation = aggregation

    def generate(self, **params) -> PredictionSetConstructingFunction:
        stretching_function = self.stretching_function_factory.generate(**params)
        return PredictionIntervalMatrixConstructingFunctionWithMeanAndStd(stretching_function=stretching_function,
                                                                          aggregation=self.aggregation)

    def get_hyperparameters(self) -> List[Hyperparameter]:
        return self.stretching_function_factory.get_hyperparameters()

    @property
    def name_aux(self):
        return "cqr"