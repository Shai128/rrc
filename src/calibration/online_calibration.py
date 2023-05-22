import abc

from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet
from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample
from src.model_prediction.model_prediction import ModelPrediction


class OnlineCalibration(abc.ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def calibrate(self, data_sample: DataSample, model_prediction: ModelPrediction, **kwargs) -> UncertaintySet:
        pass

    @abc.abstractmethod
    def update(self, labeled_sample: LabeledDataSample, model_prediction: ModelPrediction,
               calibrated_uncertainty_set: UncertaintySet, timestamp : int,
               **kwargs):
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass
