from typing import Union

import torch

from src.model_prediction.model_prediction import ModelPrediction
from src.model_prediction.mean_regressor_prediction import MeanRegressorPrediction


class I2IStdPrediction:
    def __init__(self, lower_std: Union[float, torch.Tensor], upper_std: Union[float, torch.Tensor]):
        self.lower_std = lower_std
        self.upper_std = upper_std


class I2IUQModelPrediction(ModelPrediction):

    def __init__(self, estimated_mean: MeanRegressorPrediction, std_prediction: I2IStdPrediction):
        super().__init__()
        self._estimated_mean = estimated_mean
        self._lower_std = std_prediction.lower_std
        self._upper_std = std_prediction.upper_std

    @property
    def estimated_mean(self):
        return self._estimated_mean


    @property
    def lower_std(self) -> Union[float, torch.Tensor]:
        return self._lower_std

    @property
    def upper_std(self) -> Union[float, torch.Tensor]:
        return self._upper_std