import abc
from typing import List

import numpy as np
import torch

from src.data_utils.data_sample.data_sample import LabeledDataSample
from src.data_utils.data_sample.i2i_data_sample import LabeledI2IDataSample
from src.results_helper.metrics import get_avg_miscoverage_streak_len
from src.uncertainty_quantifiers.prediction_interval import PredictionInterval, PredictionIntervalsCollection
from src.uncertainty_quantifiers.prediction_intervals_matrix import PredictionIntervalsMatrix
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet


class Loss(abc.ABC):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, uncertainty_set: UncertaintySet, labeled_sample: LabeledDataSample) -> float:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass


class OnlineLoss(Loss, abc.ABC):

    def update(self, uncertainty_set: UncertaintySet, labeled_sample: LabeledDataSample) -> None:
        pass


class ObjectiveLoss(Loss, abc.ABC):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    @abc.abstractmethod
    def batch_loss(self, calibrated_uncertainty_sets: List[UncertaintySet],
                   labeled_samples: List[LabeledDataSample]) -> float:
        pass


class MiscoverageLoss(OnlineLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, prediction_interval: PredictionInterval, labeled_sample: LabeledDataSample) -> float:
        interval = prediction_interval.intervals.squeeze()
        y = labeled_sample.y.squeeze()
        return ((y > interval[..., 1]) | (y < interval[..., 0])).float().item()

    def compute_risk(self, prediction_intervals: PredictionIntervalsCollection,
                     labeled_samples: List[LabeledDataSample]):
        intervals = prediction_intervals.intervals.squeeze()
        ys = torch.stack([labeled_sample.y for labeled_sample in labeled_samples]).squeeze()
        return ((ys > intervals[..., 1]) | (ys < intervals[..., 0])).float().mean().item()

    @property
    def name(self) -> str:
        return "miscoverage"


class PinballLoss(ObjectiveLoss):

    def __init__(self, miscoverage_level, **kwargs):
        super().__init__(miscoverage_level, **kwargs)
        self.alpha_lo = miscoverage_level / 2
        self.alpha_hi = 1 - miscoverage_level / 2

    @staticmethod
    def batch_pinball_loss(quantile_level, quantile, y):
        diff = quantile - y.squeeze()
        mask = (diff.ge(0).float() - quantile_level)
        return (mask * diff).mean().item()

    def __call__(self, prediction_interval: PredictionInterval, labeled_sample: LabeledDataSample) -> float:
        interval = prediction_interval.intervals.squeeze()
        y = labeled_sample.y.squeeze()

        return PinballLoss.batch_pinball_loss(self.alpha_hi, interval[..., 1], y) + \
               PinballLoss.batch_pinball_loss(self.alpha_lo, interval[..., 0], y)

    def batch_loss(self, calibrated_uncertainty_sets: List[PredictionInterval],
                   labeled_samples: List[LabeledDataSample]) -> float:
        intervals = torch.stack([interval.intervals.squeeze() for interval in calibrated_uncertainty_sets])
        ys = torch.stack([sample.y.squeeze() for sample in labeled_samples])
        return PinballLoss.batch_pinball_loss(self.alpha_hi, intervals[..., 1], ys) + \
               PinballLoss.batch_pinball_loss(self.alpha_lo, intervals[..., 0], ys)

    @property
    def name(self) -> str:
        return "pinball_loss"


def compute_average_miscoverage_streak_length(intervals: PredictionIntervalsCollection, labeled_samples : List[LabeledDataSample]):
    y = torch.stack([labeled_sample.y.squeeze() for labeled_sample in labeled_samples])
    y_lower = torch.stack([interval.intervals[..., 0].squeeze() for interval in intervals])
    y_upper = torch.stack([interval.intervals[..., 1].squeeze() for interval in intervals])
    return get_avg_miscoverage_streak_len(y, y_upper, y_lower)


class MiscoverageCounterLoss(OnlineLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.curr_miscoverage_counter = 0
        self.miscoverage_loss = MiscoverageLoss()

    def __call__(self, prediction_interval: PredictionInterval, labeled_sample: LabeledDataSample) -> float:
        is_miscoverage = self.miscoverage_loss(prediction_interval, labeled_sample)
        return (self.curr_miscoverage_counter + 1) * is_miscoverage

    def compute_risk(self, prediction_intervals: PredictionIntervalsCollection,
                     labeled_samples: List[LabeledDataSample]):
        curr_miscoverage_counter = 0
        all_miscoverage_counters = []
        for i in range(len(labeled_samples)):
            is_miscoverage = self.miscoverage_loss(prediction_intervals[i], labeled_samples[i])
            curr_miscoverage_counter = (curr_miscoverage_counter + 1) * is_miscoverage
            all_miscoverage_counters += [curr_miscoverage_counter]
        return np.mean(all_miscoverage_counters)

    @property
    def name(self) -> str:
        return "miscoverage_counter"

    def update(self, uncertainty_set: PredictionInterval, labeled_sample: LabeledDataSample) -> None:
        is_miscoverage = self.miscoverage_loss(uncertainty_set, labeled_sample)
        self.curr_miscoverage_counter = (self.curr_miscoverage_counter + 1) * is_miscoverage


class ImageMiscoverageLoss(OnlineLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, prediction_interval_per_pixel: PredictionIntervalsMatrix,
                 labeled_sample: LabeledI2IDataSample) -> float:
        y = labeled_sample.y.squeeze()
        valid_idx_mask = labeled_sample.valid_image_mask.to(y.device).squeeze()
        y = y[valid_idx_mask]
        intervals = prediction_interval_per_pixel.intervals.squeeze().to(y.device)[valid_idx_mask]
        coverages = ((y >= intervals[..., 0]) & (y <= intervals[..., 1])).float()
        loss = 1 - coverages.mean().item()
        return loss

    @property
    def name(self) -> str:
        return "image_miscoverage"


class PoorCenterCoverageLoss(OnlineLoss):
    def __init__(self, poor_coverage_threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.poor_coverage_threshold = poor_coverage_threshold

    def __call__(self, prediction_interval_per_pixel: PredictionIntervalsMatrix,
                 labeled_sample: LabeledI2IDataSample) -> float:
        y_t = labeled_sample.y.squeeze()
        valid_idx_mask = labeled_sample.valid_image_mask.to(y_t.device).squeeze()
        relevant_idx = torch.zeros_like(valid_idx_mask)
        center_pixel = labeled_sample.center_pixel
        relevant_idx[center_pixel[0] - 25: center_pixel[0] + 25, center_pixel[1] - 25: center_pixel[1] + 25] = True
        relevant_idx = relevant_idx & valid_idx_mask
        prediction_interval_per_pixel = prediction_interval_per_pixel.intervals.squeeze().to(y_t.device)[relevant_idx]
        y_t = y_t[relevant_idx]
        with torch.no_grad():
            cov = ((y_t >= prediction_interval_per_pixel[..., 0]) & (
                    y_t <= prediction_interval_per_pixel[..., 1])).float().mean().item()
        if cov <= self.poor_coverage_threshold:
            err_t = 1
        else:
            err_t = 0
        # args = {'image_err_t': err_t}
        return err_t

    @property
    def name(self) -> str:
        return 'poor_center_coverage'
#
