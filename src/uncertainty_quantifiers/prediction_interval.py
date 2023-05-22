from typing import List

import torch

from src.data_utils.data_sample.regression_data_sample import LabeledRegressionDataSample
from src.data_utils.data_scaler import DataScaler
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet, \
    UncertaintySetsCollection


class PredictionInterval(UncertaintySet):
    @property
    def size(self) -> float:
        return (self.intervals[..., 1] - self.intervals[..., 0]).item()

    def __init__(self, intervals: torch.Tensor):
        super().__init__()
        self.intervals = intervals

    def __contains__(self, sample: LabeledRegressionDataSample) -> bool:
        intervals = self.intervals.squeeze()
        y = sample.y.squeeze()
        return ((y <= intervals[..., 1]) & (y >= intervals[..., 0])).bool().item()


class PredictionIntervalsCollection(UncertaintySetsCollection):

    def __init__(self,  sample_size: int, scaler: DataScaler, device : torch.device):
        super().__init__(sample_size, scaler)
        self.device = device
        self.intervals_list = []
        self.sample_size = sample_size

    @property
    def intervals(self) -> torch.Tensor:
        return torch.stack(self.intervals_list).to(self.device).reshape(len(self.intervals_list), 2)

    def add_uncertainty_sets(self, new_uncertainty_sets: PredictionInterval):
        self.intervals_list += [new_uncertainty_sets.intervals]

    def __compute_lengths_aux(self, intervals):
        y_lower = intervals[:, 0]
        y_upper = intervals[:, 1]
        return y_upper - y_lower

    def compute_lengths(self):
        return self.__compute_lengths_aux(self.intervals)

    def compute_unscaled_lengths(self):
        return self.__compute_lengths_aux(self.unscaled_intervals)

    def __getitem__(self, key):
        if isinstance(key, int):
            return PredictionInterval(self.intervals_list[key])
        elif isinstance(key, slice):
            sub_list = self.intervals_list[key]
            new_collection = PredictionIntervalsCollection(len(sub_list), self.scaler, self.device)
            for interval in sub_list:
                new_collection.add_uncertainty_sets(PredictionInterval(interval))
            return new_collection
        elif isinstance(key, List):
            new_collection = PredictionIntervalsCollection(len(key), self.scaler, self.device)
            for i in key:
                new_collection.add_uncertainty_sets(PredictionInterval(self.intervals_list[i]))
            return new_collection
        else:
            raise NotImplementedError()


    @property
    def unscaled_intervals(self):
        intervals = self.intervals
        y_lower = self.scaler.unscale_y(intervals[:, 0])
        y_upper = self.scaler.unscale_y(intervals[:, 1])
        return torch.cat([y_lower.unsqueeze(-1), y_upper.unsqueeze(-1)], dim=-1)
