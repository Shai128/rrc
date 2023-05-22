from typing import Union

import torch

from src.data_utils.data_sample.regression_data_sample import RegressionDataSample, LabeledRegressionDataSample
from src.data_utils.data_scaler import DataScaler
from src.data_utils.data_utils import DataType
from src.data_utils.datasets.dataset import Dataset


class RegressionDataset(Dataset):

    def __init__(self, x: torch.Tensor, y: torch.Tensor, backward_size: int, max_data_size: int,
                 offline_train_ratio: float,
                 test_ratio: float, validation_ratio: float, data_type: DataType,
                 dataset_name: str, device):
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)
        self._scaler = None
        self.unscaled_x = x[:max_data_size].to(device)
        self.unscaled_y = y[:max_data_size].to(device)
        self.data_size = self.unscaled_x.shape[0]
        self.backward_size = backward_size
        super().__init__(x.shape[-1], y.shape[-1], max_data_size,  offline_train_ratio, test_ratio, validation_ratio, data_type, dataset_name, device)
        train_timestamps = self.get_online_train_timestamps()
        x_train = self.unscaled_x[train_timestamps]
        y_train = self.unscaled_y[train_timestamps]
        self._scaler = DataScaler()
        self._scaler.initialize_scalers(x_train, y_train)
        x = self._scaler.scale_x(self.unscaled_x).to(device)
        y = self._scaler.scale_y(self.unscaled_y).to(device)
        self.x_repeated = RegressionDataset.create_data_with_window_dimension(x, prev_x=None,
                                                                              backward_size=self.backward_size)
        self.y_repeated = RegressionDataset.create_data_with_window_dimension(y, prev_x=None,
                                                                              backward_size=self.backward_size)
        self.inference_samples = []
        self.labeled_inference_samples = []
        for t in range(self.data_size):
            self.inference_samples.append(RegressionDataSample(self.x_repeated[t].unsqueeze(0),
                                                               self.y_repeated[t, :-1].unsqueeze(0)))
            self.labeled_inference_samples.append(LabeledRegressionDataSample(self.x_repeated[t], self.y_repeated[t, -1],
                                           self.x_repeated[:t + 1], self.y_repeated[:t + 1],
                                           self.y_repeated[t, :-1], self.y_repeated[:t + 1, :-1]))

    @property
    def scaler(self) -> DataScaler:
        return self._scaler

    @property
    def data_size(self):
        return self._data_size

    @data_size.setter
    def data_size(self, value):
        self._data_size = value

    def get_inference_sample(self, index: int) -> RegressionDataSample:
        return self.inference_samples[index]

    def get_labeled_inference_sample(self, index: int) -> LabeledRegressionDataSample:
        return self.labeled_inference_samples[index]

    def get_train_sample(self, index: int) -> LabeledRegressionDataSample:
        return self.get_labeled_inference_sample(index)

    @staticmethod
    def create_data_with_window_dimension(x: torch.Tensor, prev_x: Union[torch.Tensor, None], backward_size: int):
        if prev_x is None:
            prev_x = torch.zeros_like(x)[:backward_size + 1]
        if backward_size > 0:
            x = torch.cat([prev_x[-backward_size:], x], dim=0)

        x_repeated = torch.zeros(x.shape[0] - backward_size, backward_size + 1, x.shape[1]).to(x.device)
        for i in range(backward_size + 1):
            x_repeated[:, i, :] = x[i:i + x.shape[0] - backward_size]

        return x_repeated