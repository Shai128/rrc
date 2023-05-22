import abc
from abc import ABC

from src.data_utils.data_sample.i2i_data_sample import I2IDataSample
from src.model_prediction.mean_regressor_prediction import MeanRegressorPrediction
from src.models.image_to_image_models.abstract_models.online_i2i_model import OnlineI2ILearningModel


class I2IMeanRegressor(OnlineI2ILearningModel, ABC):

    def __init__(self, **kwargs):
        OnlineI2ILearningModel.__init__(self, **kwargs)

    @abc.abstractmethod
    def predict(self, data_sample: I2IDataSample) -> MeanRegressorPrediction:
        pass

