import abc
from typing import List

import numpy as np
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter

from src.calibration.stretching_functions.stretching_functions import AdaptiveStretchingFunction, \
    ExponentialStretching, ErrorAdaptiveStretching, IdentityStretching
from src.losses import Loss
from src.losses_factory import LossFactory


class StretchingFunctionFactory(abc.ABC):

    @abc.abstractmethod
    def generate(self, **params) -> AdaptiveStretchingFunction:
        pass

    @abc.abstractmethod
    def get_hyperparameters(self) -> List[Hyperparameter]:
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


class ExponentialStretchingFactory(StretchingFunctionFactory):

    def generate(self, exp_base=np.e) -> ExponentialStretching:
        return ExponentialStretching(exp_base)

    def get_hyperparameters(self) -> List[Hyperparameter]:
        return [CategoricalHyperparameter("exp_base", choices=[2, np.e, 5])]

    @property
    def name(self):
        return "exp"



class IdentityStretchingFactory(StretchingFunctionFactory):

    def generate(self) -> IdentityStretching:
        return IdentityStretching()

    def get_hyperparameters(self) -> List[Hyperparameter]:
        return []

    @property
    def name(self):
        return "id"


class ErrorAdaptiveStretchingFactory(StretchingFunctionFactory):

    def __init__(self, loss_factory: LossFactory, alpha: float):
        self.loss_factory = loss_factory
        self.alpha = alpha

    def generate(self, exp_base: float = np.e, beta_low: float = -1, beta_high: float = 1, beta_score: float = 1,
                 beta_loss: float = 1) -> ErrorAdaptiveStretching:
        return ErrorAdaptiveStretching(exp_base, beta_low, beta_high, beta_score, beta_loss, self.loss_factory.generate(), self.alpha)

    def get_hyperparameters(self) -> List[Hyperparameter]:
        return [
            UniformFloatHyperparameter("beta_low", lower=-1, upper=0),
            UniformFloatHyperparameter("beta_high", lower=0, upper=1),
            CategoricalHyperparameter("exp_base", choices=[0, 2, np.e, 5]),
            UniformFloatHyperparameter("beta_score", lower=0.01, upper=2),
            UniformFloatHyperparameter("beta_loss", lower=0.01, upper=2),
        ]

    @property
    def name(self):
        return "error_adaptive"
