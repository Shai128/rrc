import torch

from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample


class RegressionDataSample(DataSample):
    def __init__(self, x: torch.Tensor, previous_ys: torch.Tensor, **kwargs):
        DataSample.__init__(self, x, **kwargs)
        self.previous_ys = previous_ys


class LabeledRegressionDataSample(LabeledDataSample, RegressionDataSample):
    def __init__(self, x : torch.Tensor, y: torch.Tensor, all_x: torch.Tensor, all_y: torch.Tensor,
                 previous_ys: torch.Tensor, all_previous_ys: torch.Tensor, **kwargs):
        super().__init__(x=x, y=y, previous_ys=previous_ys, **kwargs)
        self.all_previous_ys = all_previous_ys
        self.all_x = all_x
        self.all_y = all_y

