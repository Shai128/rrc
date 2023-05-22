import abc
from enum import Enum, auto
from typing import List

from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter

from src.calibration.aci_calibration import ACICalibration
from src.calibration.dummy_calibration import DummyI2ICalibration
from src.calibration.online_calibration import OnlineCalibration
from src.calibration.rolling_risk_control import RollingRiskControl, RollingRiskControlWithMultipleRisks
from src.calibration.set_constructing_functions.set_constructing_function_factory import \
    PredictionSetConstructingFunctionFactory
from src.losses_factory import OnlineLossFactory


class CalibrationType(Enum):
    RollingRiskControl = auto()
    ACI = auto()
    MultipleRisksRRC = auto()


class OnlineCalibrationFactory:
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate(self, **params) -> OnlineCalibration:
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def get_hyperparameters(self) -> List[Hyperparameter]:
        pass


GAMMA_CHOICES = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 10]


class RollingRiskControlFactory(OnlineCalibrationFactory):

    def __init__(self, alpha: float, set_constructing_function_factory: PredictionSetConstructingFunctionFactory,
                 loss_factory: OnlineLossFactory, gamma=0.01, gamma_choices=None, **set_constructing_function_params):
        super().__init__()
        self.alpha = alpha
        self.set_constructing_function_factory = set_constructing_function_factory
        self.loss_factory = loss_factory
        self.gamma = gamma
        self.set_constructing_function_params = set_constructing_function_params
        self.gamma_choices = gamma_choices if gamma_choices is not None else GAMMA_CHOICES

    def generate(self, rrc_gamma=None, **set_constructing_function_params) -> RollingRiskControl:
        set_constructing_function = self.set_constructing_function_factory.generate(**set_constructing_function_params)
        gamma = self.gamma if rrc_gamma is None else rrc_gamma
        return RollingRiskControl(gamma=gamma, alpha=self.alpha, loss=self.loss_factory.generate(),
                                  set_constructing_function=set_constructing_function)

    def get_hyperparameters(self) -> List[Hyperparameter]:
        return [
            CategoricalHyperparameter("rrc_gamma", choices=self.gamma_choices),
            *self.set_constructing_function_factory.get_hyperparameters()
        ]

    @property
    def name(self):
        return f"rrc_f={self.set_constructing_function_factory.name}_loss={self.loss_factory.name}"


class RollingRiskControlWithMultipleRisksFactory(OnlineCalibrationFactory):

    def __init__(self, alphas: List[float], set_constructing_function_factory: PredictionSetConstructingFunctionFactory,
                 losses_factories: List[OnlineLossFactory], gamma=0.01, gamma_choices=None, **set_constructing_function_params):
        super().__init__()
        self.alphas = alphas
        self.set_constructing_function_factory = set_constructing_function_factory
        self.losses_factories = losses_factories
        self.gamma = gamma
        self.set_constructing_function_params = set_constructing_function_params
        self.gamma_choices = gamma_choices if gamma_choices is not None else GAMMA_CHOICES

    def generate(self, **params) -> RollingRiskControlWithMultipleRisks:
        if len(params) == 0:
            gammas = [self.gamma] * len(self.losses_factories)
            set_constructing_function = self.set_constructing_function_factory.generate(
                **self.set_constructing_function_params)
        else:
            gammas = list(map(lambda i: params[f'multi_rrc_gamma_{i}'], range(len(self.losses_factories))))
            set_constructing_function_params = {key: params[key] for key in params.keys() if
                                                'multi_rrc_gamma_' not in key}
            set_constructing_function = self.set_constructing_function_factory.generate(
                **set_constructing_function_params,
                **self.set_constructing_function_params)
        losses = [loss_factory.generate() for loss_factory in self.losses_factories]
        return RollingRiskControlWithMultipleRisks(gammas, self.alphas, set_constructing_function, losses)

    def get_hyperparameters(self) -> List[Hyperparameter]:
        gammas_params = [
            CategoricalHyperparameter(f"multi_rrc_gamma_{i}", choices=self.gamma_choices)
            for i in range(len(self.losses_factories))
        ]
        return [
            *self.set_constructing_function_factory.get_hyperparameters(),
            *gammas_params
        ]

    @property
    def name(self):
        return f"multi_rrc_f={self.set_constructing_function_factory.name}_losses={[loss.name for loss in self.losses_factories]}"


class DummyI2ICalibrationFactory(OnlineCalibrationFactory):

    def __init__(self, lambda_hat: int = 1):
        super().__init__()
        self.lambda_hat = lambda_hat
        pass

    def generate(self, lambda_hat: int = None, **params) -> DummyI2ICalibration:
        if lambda_hat is None:
            lambda_hat = self.lambda_hat
        return DummyI2ICalibration(lambda_hat)

    def get_hyperparameters(self) -> List[Hyperparameter]:
        return [
            UniformFloatHyperparameter("lambda_hat", lower=-20, upper=20),
        ]

    @property
    def name(self) -> str:
        return "uncalibrated"


class ACIFactory(OnlineCalibrationFactory):

    def __init__(self, alpha: float, calibration_set_size: int, gamma=0.01, gamma_choices=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_choices = gamma_choices if gamma_choices is not None else GAMMA_CHOICES
        self.calibration_set_size = calibration_set_size

    def generate(self, aci_gamma=None) -> ACICalibration:
        gamma = self.gamma if aci_gamma is None else aci_gamma
        return ACICalibration(calibration_set_size=self.calibration_set_size, gamma=gamma, alpha=self.alpha)

    def get_hyperparameters(self) -> List[Hyperparameter]:
        return [
            CategoricalHyperparameter("aci_gamma", choices=self.gamma_choices),
        ]

    @property
    def name(self):
        return f"aci"
