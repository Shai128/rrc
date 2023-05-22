import abc
from typing import Union, List

import numpy as np
import torch

from src.uncertainty_quantifiers.prediction_interval import PredictionInterval
from src.uncertainty_quantifiers.prediction_intervals_matrix import PredictionIntervalsMatrix
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet
from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample
from src.model_prediction.model_prediction import ModelPrediction
from src.model_prediction.qr_model_prediction import QRModelPrediction
from src.model_prediction.i2i_model_prediction import I2IUQModelPrediction
from src.calibration.stretching_functions.stretching_functions import AdaptiveStretchingFunction


class PredictionSetConstructingFunction(abc.ABC):
    def __init__(self, stretching_function: AdaptiveStretchingFunction):
        self.stretching_function = stretching_function
        pass

    @abc.abstractmethod
    def __call__(self, data_sample: DataSample, theta, model_prediction: ModelPrediction) -> UncertaintySet:
        pass

    @property
    def name(self):
        return f"{self.name_aux}_{self.stretching_function.name}"

    def update(self, labeled_sample: LabeledDataSample, model_prediction: ModelPrediction,
               calibrated_uncertainty_set: UncertaintySet, **kwargs) -> None:
        pass

    @property
    @abc.abstractmethod
    def name_aux(self):
        pass


class PredictionIntervalConstructingFunctionWithCQR(PredictionSetConstructingFunction):

    def __init__(self, stretching_function: AdaptiveStretchingFunction, **kwargs):
        super().__init__(stretching_function)

    def __call__(self, data_sample: DataSample, theta: Union[torch.Tensor, int, float],
                 model_prediction: QRModelPrediction) -> PredictionInterval:
        lower, upper = model_prediction.intervals
        if not torch.is_tensor(theta):
            theta = torch.Tensor([theta]).to(lower.device)
        Q = self.stretching_function(theta)
        return PredictionInterval(torch.cat([lower - Q, upper + Q], dim=-1))

    def update(self, labeled_sample: LabeledDataSample, model_prediction: QRModelPrediction,
               calibrated_uncertainty_set: PredictionInterval, **kwargs) -> None:
        self.stretching_function.update(labeled_sample, model_prediction, calibrated_uncertainty_set)

    @property
    def name_aux(self):
        return "cqr"


class ThetaAggregation(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def aggregate(self, thetas: List[float]) -> float:
        pass

    @property
    def name(self) -> str:
        pass


class MaxAggregation(ThetaAggregation):
    def __init__(self):
        super().__init__()

    def aggregate(self, thetas: List[float]) -> float:
        return max(thetas)

    @property
    def name(self) -> str:
        return "max_agg"


class MeanAggregation(ThetaAggregation):
    def __init__(self):
        super().__init__()

    def aggregate(self, thetas: List[float]) -> float:
        return np.mean(thetas).item()

    @property
    def name(self) -> str:
        return "mean_agg"


class PredictionIntervalMatrixConstructingFunctionWithMeanAndStd(PredictionSetConstructingFunction):
    def __init__(self, stretching_function: AdaptiveStretchingFunction, aggregation: ThetaAggregation):
        super().__init__(stretching_function)
        self.aggregation = aggregation

    def __call__(self, data_sample: DataSample, thetas: Union[torch.Tensor, List[float]],
                 model_prediction: I2IUQModelPrediction) -> PredictionIntervalsMatrix:
        lambdas = [self.stretching_function(torch.Tensor([theta])).item() for theta in thetas]
        lambda_hat = self.aggregation.aggregate(lambdas)
        estimated_mean, l, u = model_prediction.estimated_mean.mean_estimate, model_prediction.lower_std, model_prediction.upper_std
        if torch.is_tensor(l):
            l = l.cpu()
        if torch.is_tensor(u):
            u = u.cpu()
        estimated_mean = estimated_mean.cpu()
        shape = estimated_mean.squeeze().shape + (2,)
        calibrated_interval = torch.zeros(*shape)

        calibrated_interval[..., 0] = estimated_mean.clone() - lambda_hat * l
        calibrated_interval[..., 1] = estimated_mean.clone() + lambda_hat * u
        calibrated_interval = PredictionIntervalsMatrix(calibrated_interval, estimated_mean)
        return calibrated_interval

    @property
    def name_aux(self):
        return f"mean_std_agg={self.aggregation.name}"
