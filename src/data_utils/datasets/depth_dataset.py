from typing import List

import numpy as np
import torch

from src.data_utils.data_sample.depth_data_sample import DepthDataSample, LabeledDepthDataSample, AugmentationInfo
from src.data_utils.data_scaler import DataScaler
from src.data_utils.data_utils import DataType
from src.data_utils.datasets.dataset import Dataset
from src.models.image_to_image_models.depth_models.Leres.data.multi_dataset import MultiDataset


class DepthDataset(Dataset):

    def __init__(self, base_data_path: str, annotations_path: str, dataset_name: str, max_data_size: int,
                 offline_train_ratio: float, test_ratio: float, validation_ratio: float, data_type: DataType, device : torch.device):
        self.multi_data = MultiDataset(base_data_path, annotations_path, dataset_name)
        self._data_size = min(self.multi_data.data_size, max_data_size)
        super().__init__(self.multi_data.x_dim, self.multi_data.y_dim, max_data_size, offline_train_ratio, test_ratio, validation_ratio,
                         data_type, dataset_name, device)
        data = self.multi_data.get_data(0, apply_augmentation=False)
        self.image_width = data['rgb'].shape[1]
        self.image_height = data['rgb'].shape[2]

    def get_inference_sample(self, index: int) -> DepthDataSample:
        data = self.multi_data.get_data(index, apply_augmentation=False)
        data['augmentation_info'] = AugmentationInfo(**data['augmentation_info'])
        sparse_y = data['depth'].clone()
        sparse_y[~data['feature_points_mask']] = np.nan
        return DepthDataSample(x=data['rgb'],
                               sparse_y=sparse_y,
                               **data)

    def get_labeled_inference_sample(self, index: int) -> LabeledDepthDataSample:
        data = self.multi_data.get_data(index, apply_augmentation=False)
        data['augmentation_info'] = AugmentationInfo(**data['augmentation_info'])
        sparse_y = data['depth'].clone()
        sparse_y[~data['feature_points_mask']] = np.nan
        sample = LabeledDepthDataSample(x=data['rgb'], y=data['depth'], sparse_y=sparse_y, **data)
        return sample

    def get_train_sample(self, index: int) -> LabeledDepthDataSample:
        data = self.multi_data.get_data(index, apply_augmentation=True)
        data['augmentation_info'] = AugmentationInfo(**data['augmentation_info'])
        sparse_y = data['depth'].clone()
        sparse_y[~data['feature_points_mask']] = np.nan
        return LabeledDepthDataSample(x=data['rgb'], y=data['depth'],
                                      sparse_y=sparse_y,
                                      **data)

    @property
    def scaler(self) -> DataScaler:
        return None

    @property
    def data_size(self):
        return self._data_size