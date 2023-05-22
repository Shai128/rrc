import abc
from typing import Tuple, List

from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample
from src.data_utils.data_scaler import DataScaler
from src.data_utils.data_utils import DataType


class Dataset(abc.ABC):
    def __init__(self, x_dim: int, y_dim: int, max_data_size: int, offline_train_ratio: float, test_ratio: float,
                 validation_ratio: float,
                 data_type: DataType, dataset_name: str, device):
        self._x_dim = x_dim
        self._y_dim = y_dim
        self.max_data_size = max_data_size
        self.offline_train_ratio = offline_train_ratio
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.data_type = data_type
        self.dataset_name = dataset_name
        self.device = device
        offline_train_size = round(self.data_size * self.offline_train_ratio)
        test_size = round(self.data_size * self.test_ratio)
        validation_size = round(self.data_size * self.validation_ratio)
        train_size = self.data_size - test_size - validation_size
        self.offline_train_timestamps = list(range(offline_train_size))
        self.online_train_timestamps = list(range(offline_train_size, train_size))
        self.validation_timestamps = list(range(train_size, train_size + validation_size))
        self.test_timestamps = list(range(train_size + validation_size, self.data_size))

    def get_data_timestamps(self) -> Tuple[List[int], List[int], List[int]]:
        return self.online_train_timestamps, self.validation_timestamps, self.test_timestamps

    def get_offline_train_timestamps(self) -> List[int]:
        return self.offline_train_timestamps

    def get_online_train_timestamps(self)-> List[int]:
        return self.online_train_timestamps

    def get_validation_timestamps(self) -> List[int]:
        return self.validation_timestamps

    def get_test_timestamps(self) -> List[int]:
        return self.test_timestamps


    @property
    def x_dim(self):
        return self._x_dim

    @property
    @abc.abstractmethod
    def scaler(self) -> DataScaler:
        pass

    @property
    def y_dim(self):
        return self._y_dim

    @property
    @abc.abstractmethod
    def data_size(self):
        pass

    @abc.abstractmethod
    def get_inference_sample(self, index: int) -> DataSample:
        pass

    @abc.abstractmethod
    def get_labeled_inference_sample(self, index: int) -> LabeledDataSample:
        pass

    @abc.abstractmethod
    def get_train_sample(self, index: int) -> LabeledDataSample:
        pass