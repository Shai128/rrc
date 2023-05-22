import abc

from src.model_prediction.model_prediction import ModelPrediction


class MeanRegressorPrediction(ModelPrediction, abc.ABC):
    def __init__(self, mean_estimate):
        super().__init__()
        self.mean_estimate = mean_estimate