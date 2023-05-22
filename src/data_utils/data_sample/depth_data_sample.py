from typing import List
from src.data_utils.data_sample.i2i_data_sample import I2IDataSample, LabeledI2IDataSample
from src.model_prediction.mean_regressor_prediction import MeanRegressorPrediction


class AugmentationInfo:
    def __init__(self, pad: List[int], flip_flg, resize_size, crop_size, **kwargs):
        self.pad = pad
        self.flip_flg = flip_flg
        self.resize_size = resize_size
        self.crop_size = crop_size


class DepthDataSample(I2IDataSample):
    def __init__(self, x, valid_image_mask, center_pixel, augmentation_info: AugmentationInfo, sparse_y,
                 feature_points_mask, **kwargs):
        super().__init__(x, valid_image_mask, center_pixel, **kwargs)
        self.feature_points_mask = feature_points_mask
        self.sparse_y = sparse_y
        self.augmentation_info = augmentation_info
        self.rgb = x


class LabeledDepthDataSample(LabeledI2IDataSample, DepthDataSample):
    def __init__(self, x, y, valid_image_mask, center_pixel, augmentation_info: AugmentationInfo, sparse_y,
                 quality_flg, disp, planes, focal_length, scaling_factor, meters_factor,
                 feature_points_mask, **kwargs):
        LabeledI2IDataSample.__init__(self, x, y, valid_image_mask, center_pixel, **kwargs)
        DepthDataSample.__init__(self, x, valid_image_mask, center_pixel, augmentation_info, sparse_y,
                                 feature_points_mask, **kwargs)
        self.depth = y
        self.quality_flg = quality_flg
        self.disp = disp
        self.planes = planes
        self.focal_length = focal_length
        self.scaling_factor = scaling_factor
        self.meters_factor = meters_factor


class DepthStdRegressionDataSample(DepthDataSample):
    def __init__(self, x, valid_image_mask, center_pixel, mean_estimate: MeanRegressorPrediction,
                 augmentation_info: AugmentationInfo, sparse_y,
                 feature_points_mask, **kwargs):
        super().__init__(x, valid_image_mask, center_pixel, augmentation_info, sparse_y, feature_points_mask, **kwargs)
        self.mean_estimate = mean_estimate


class LabeledDepthStdRegressionDataSample(LabeledI2IDataSample, DepthStdRegressionDataSample):
    def __init__(self, x, y, valid_image_mask, center_pixel, mean_estimate: MeanRegressorPrediction,
                 augmentation_info: AugmentationInfo, sparse_y,
                 feature_points_mask, **kwargs):
        LabeledI2IDataSample.__init__(self, x, y, valid_image_mask, center_pixel, **kwargs)
        DepthStdRegressionDataSample.__init__(self, x, valid_image_mask, center_pixel, mean_estimate, augmentation_info,
                                              sparse_y, feature_points_mask)
