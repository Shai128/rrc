import abc
from abc import abstractmethod

from src.data_utils.data_scaler import DataScaler


class UncertaintySet(abc.ABC):
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def size(self) -> float:
        pass


class UncertaintySetsCollection(abc.ABC):
    def __init__(self, sample_size: int, scaler: DataScaler):
        self.sample_size = sample_size
        self.scaler = scaler

    @abstractmethod
    def add_uncertainty_sets(self, new_uncertainty_sets: UncertaintySet):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass
