import torch

from src.data_utils.data_scaler import DataScaler
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet, \
    UncertaintySetsCollection


class PredictionIntervalsMatrix(UncertaintySet):

    def __init__(self, intervals: torch.Tensor, estimated_means : torch.Tensor):
        super().__init__()
        self.intervals = intervals
        self.estimated_means = estimated_means

    @property
    def size(self) -> float:
        return (self.intervals[..., 1] - self.intervals[..., 0]).mean().item()


class PredictionIntervalsMatrixCollection(UncertaintySetsCollection):

    def __init__(self, sample_size, device, scaler: DataScaler, width, height):
        super().__init__(sample_size, scaler)
        self.device = device
        self.intervals_list = []
        self.estimated_means_list = []
        self.sample_size = sample_size
        self.width = width
        self.height = height

    def add_uncertainty_sets(self, new_uncertainty_sets: PredictionIntervalsMatrix):
        self.intervals_list += [new_uncertainty_sets.intervals.cpu()]
        self.estimated_means_list += [new_uncertainty_sets.estimated_means.cpu()]

    @property
    def unscaled_intervals(self):
        return torch.stack(self.intervals_list).reshape(self.sample_size, self.width, self.height, 2).to(self.device)

    @property
    def unscaled_mean(self):
        return torch.stack(self.estimated_means_list).reshape(self.sample_size, self.width, self.height).to(self.device)

    def __getitem__(self, key):
        if isinstance(key, int):
            return PredictionIntervalsMatrix(self.intervals_list[key], self.estimated_means_list[key])
        else:
            raise NotImplementedError()

