import torch

from src.model_prediction.model_prediction import ModelPrediction


class QRModelPrediction(ModelPrediction):

    def __init__(self, intervals: torch.Tensor):
        super().__init__()
        self.intervals = intervals

    @property
    def intervals(self) -> torch.Tensor:
        return self._intervals

    @intervals.setter
    def intervals(self, value):
        self._intervals = value
