import abc
import numbers

import numpy as np
import torch

from src.data_utils.data_sample.data_sample import LabeledDataSample
from src.losses import Loss
from src.model_prediction.model_prediction import ModelPrediction
from src.model_prediction.qr_model_prediction import QRModelPrediction
from src.uncertainty_quantifiers.prediction_interval import PredictionInterval
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet


def exp(x: torch.Tensor, base) -> torch.Tensor:
    device = x.device
    x = x.cpu().numpy()
    return torch.tensor((np.nan_to_num(np.power(base, x) - 1)) * (x > 0) -
                        (np.nan_to_num(np.power(base, -x) - 1)) * (x < 0)).to(device)


def exp_float(x: float, base) -> float:
    return np.nan_to_num(np.power(base, x) - 1) * (x > 0) - np.nan_to_num(np.power(base, -x) - 1) * (x < 0)


class AdaptiveStretchingFunction(abc.ABC):

    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def __call__(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def update(self, labeled_sample: LabeledDataSample, model_prediction: ModelPrediction,
               calibrated_uncertainty_set: UncertaintySet, **kwargs) -> None:
        pass


class IdentityStretching(AdaptiveStretchingFunction):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return "id"

    def __call__(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        return theta


class ExponentialStretching(AdaptiveStretchingFunction):

    def __init__(self, base=np.e):
        super().__init__()
        self.base = base

    @property
    def name(self):
        base_name = 'e' if self.base == np.e else np.round(self.base, 3)
        return f"exp_{base_name}"

    def __call__(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        base = self.base
        theta_exp = exp(theta, base)
        linear_idx = (-0.1 <= exp(theta, base)) & (exp(theta, base) <= 0.1)
        return linear_idx * theta + (~linear_idx) * theta_exp


class ErrorAdaptiveStretching(AdaptiveStretchingFunction):

    def __init__(self, exp_base: float, beta_low: float, beta_high: float, beta_score: float, beta_loss: float,
                 loss: Loss, alpha: float):
        super().__init__()
        self.exp_base = exp_base
        self.beta_low = beta_low
        self.beta_high = beta_high
        self.beta_score = beta_score
        self.beta_loss = beta_loss
        self.lambda_t = 0
        self.loss = loss
        self.alpha = alpha

    @property
    def name(self):
        params_dict = self.__dict__
        new_params_dict = {}
        for param, value in params_dict.items():
            if isinstance(value, numbers.Number):
                new_params_dict[param] = np.round(value, 3)
        return f"error_adpt_{str(new_params_dict)}"

    def __call__(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        base = self.exp_base
        theta_exp = exp(theta, base)
        linear_idx = (-0.1 <= theta_exp) & (theta_exp <= 0.1)
        return linear_idx * theta + (~linear_idx) * theta_exp + self.lambda_t

    def update(self, labeled_sample: LabeledDataSample, model_prediction: QRModelPrediction,
               calibrated_uncertainty_set: PredictionInterval, **kwargs) -> None:
        interval = calibrated_uncertainty_set.intervals.squeeze()
        y = labeled_sample.y.squeeze()
        score = torch.min(abs(interval[..., 0] - y), abs(y - interval[..., 1])).item()
        delta_loss = self.loss(calibrated_uncertainty_set, labeled_sample) - self.alpha
        error_term = exp_float(self.beta_loss * abs(delta_loss), self.exp_base) * np.sign(delta_loss)
        self.lambda_t += self.beta_score * score * error_term
        self.lambda_t = np.clip(self.lambda_t, self.beta_low, self.beta_high)
