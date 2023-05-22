import abc
from abc import ABC

import torch

from src.data_utils.data_sample.depth_data_sample import DepthStdRegressionDataSample
from src.data_utils.data_sample.i2i_data_sample import I2IDataSample, LabeledI2IDataSample
from src.model_prediction.i2i_model_prediction import I2IStdPrediction
from src.models.abstract_models.online_learning_model import OnlineLearningModel


def dummy_loss(device):
    loss = torch.Tensor([0]).to(device)
    loss.requires_grad = True
    return loss


class I2IStdRegressor(OnlineLearningModel, ABC):
    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device

    @abc.abstractmethod
    def predict(self, data_sample: DepthStdRegressionDataSample) -> I2IStdPrediction:
        pass

    def fit(self, labeled_data_sample: LabeledI2IDataSample, **kwargs):
        pass


class BaselineStdRegressor(I2IStdRegressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def online_fit(self, labeled_data_sample: LabeledI2IDataSample, **kwargs):
        pass

    def predict(self, data_sample: I2IDataSample) -> I2IStdPrediction:
        return I2IStdPrediction(1, 1)

    @property
    def name(self):
        return "baseline"
