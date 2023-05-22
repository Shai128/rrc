from typing import List

import numpy as np
import torch

from src.data_utils.data_sample.depth_data_sample import LabeledDepthDataSample
from src.data_utils.data_sample.regression_data_sample import LabeledRegressionDataSample
from src.losses import ObjectiveLoss
from src.results_helper.metrics import get_avg_miscoverage_counter, get_avg_miscoverage_streak_len
from src.uncertainty_quantifiers.prediction_interval import PredictionInterval
from src.uncertainty_quantifiers.prediction_intervals_matrix import PredictionIntervalsMatrix
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet


class RegressionObjectiveLoss(ObjectiveLoss):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(alpha, **kwargs)

    def __call__(self, uncertainty_set: PredictionInterval, labeled_sample: LabeledRegressionDataSample) -> float:
        raise NotImplementedError()

    def batch_loss(self, calibrated_uncertainty_sets: List[PredictionInterval],
                   labeled_samples: List[LabeledRegressionDataSample]) -> float:
        ys = torch.stack([sample.y.squeeze() for sample in labeled_samples]).squeeze()
        intervals = torch.stack([interval.intervals.squeeze() for interval in calibrated_uncertainty_sets]).squeeze()
        miscoverage = 1 - ((ys <= intervals[:, 1]) & (ys >= intervals[:, 0])).float().mean().item()
        delta_coverage = abs(miscoverage - self.alpha)
        if delta_coverage > 0.005:
            return delta_coverage * 1000000
        else:
            msl = get_avg_miscoverage_streak_len(ys, intervals[:, 1], intervals[:, 0])
            delta_msl = abs(msl - 1 / (1-self.alpha))
            length = (intervals[:, 1] - intervals[:, 0]).mean().item()
            loss = (1 + delta_msl) * length
            return loss

    @property
    def name(self) -> str:
        return "cov&msl&len"


class MCRegressionObjectiveLoss(ObjectiveLoss):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(alpha, **kwargs)

    def __call__(self, uncertainty_set: PredictionInterval, labeled_sample: LabeledRegressionDataSample) -> float:
        raise NotImplementedError()

    def batch_loss(self, calibrated_uncertainty_sets: List[PredictionInterval],
                   labeled_samples: List[LabeledRegressionDataSample]) -> float:
        ys = torch.stack([sample.y.squeeze() for sample in labeled_samples]).squeeze()
        intervals = torch.stack([interval.intervals.squeeze() for interval in calibrated_uncertainty_sets]).squeeze()
        mc = get_avg_miscoverage_counter(ys, intervals[:, 1], intervals[:, 0])
        delta_mc = abs(mc - self.alpha)
        if delta_mc > 0.005:
            return delta_mc * 1000000
        else:
            length = (intervals[:, 1] - intervals[:, 0]).mean().item()
            return length

    @property
    def name(self) -> str:
        return "mc&len"


class ImageMiscoverageObjectiveLoss(ObjectiveLoss):
    def __init__(self, alpha: float, device, **kwargs):
        super().__init__(alpha, **kwargs)
        self.device = device

    def __call__(self, uncertainty_set: UncertaintySet, labeled_sample: LabeledDepthDataSample) -> float:
        raise NotImplementedError()

    def batch_loss(self, calibrated_uncertainty_sets: List[PredictionIntervalsMatrix],
                   labeled_samples: List[LabeledDepthDataSample]) -> float:
        valid_image_mask = torch.stack([sample.valid_image_mask for sample in labeled_samples]).cpu()
        ys = torch.stack([sample.y.squeeze() for sample in labeled_samples]).squeeze().cpu()
        intervals = torch.stack([interval.intervals.squeeze() for interval in calibrated_uncertainty_sets]).squeeze().cpu()
        miscoverage = 1 - ((ys <= intervals[..., 1]) & (ys >= intervals[..., 0]))[valid_image_mask].float().mean().item()
        delta_coverage = abs(miscoverage - self.alpha)
        if delta_coverage > 0.002:
            return delta_coverage * 1000000
        else:
            length = (intervals[..., 1] - intervals[..., 0]).mean().item()
            return length

    @property
    def name(self) -> str:
        return "cov&len"


class MultipleRisksObjectiveLoss(ObjectiveLoss):
    def __init__(self, image_miscoverage_alpha: float, poor_center_coverage_alpha: float, device, **kwargs):
        super().__init__(image_miscoverage_alpha, **kwargs)
        self.image_miscoverage_alpha = image_miscoverage_alpha
        self.poor_center_coverage_alpha = poor_center_coverage_alpha
        self.device = device

    def __call__(self, uncertainty_set: UncertaintySet, labeled_sample: LabeledDepthDataSample) -> float:
        raise NotImplementedError()

    def batch_loss(self, calibrated_uncertainty_sets: List[PredictionIntervalsMatrix],
                   labeled_samples: List[LabeledDepthDataSample]) -> float:
        valid_image_mask = torch.stack([sample.valid_image_mask for sample in labeled_samples]).cpu()
        image_center = torch.stack([torch.Tensor(sample.center_pixel) for sample in labeled_samples]).cpu().int()

        ys = torch.stack([sample.y.squeeze() for sample in labeled_samples]).squeeze().cpu()
        intervals = torch.stack([interval.intervals.squeeze() for interval in calibrated_uncertainty_sets]).squeeze().cpu()
        miscoverages = 1 - ((ys <= intervals[..., 1]) & (ys >= intervals[..., 0])).float()

        poor_center_coverages = []
        for i in range(len(image_center)):
            center_idxs = torch.zeros_like(valid_image_mask[i])
            center_idxs[image_center[i, 0].item() - 25: image_center[i, 0].item() + 25,
            image_center[i, 1].item() - 25: image_center[i, 1].item() + 25] = True
            poor_center_coverages += [
                1 if miscoverages[i, center_idxs & valid_image_mask[i]].mean().item() < 0.6 else 0]
        poor_center_coverage = np.mean(poor_center_coverages)

        image_miscoverage = miscoverages[valid_image_mask].mean().item()
        loss = (intervals[..., 1] - intervals[..., 0]).mean().item()
        if image_miscoverage > self.image_miscoverage_alpha + 0.02:
            loss += abs(image_miscoverage - self.image_miscoverage_alpha) * 1000000

        if poor_center_coverage > self.poor_center_coverage_alpha + 0.01:
            loss += abs(poor_center_coverage - self.poor_center_coverage_alpha) * 1000000

        return loss

    @property
    def name(self) -> str:
        return "cov&center_cov&len"

