import abc
from typing import List

import numpy as np
import torch

from src.data_utils.data_sample.data_sample import LabeledDataSample
from src.data_utils.data_sample.i2i_data_sample import LabeledI2IDataSample
from src.losses import Loss, MiscoverageLoss, MiscoverageCounterLoss, ImageMiscoverageLoss, PoorCenterCoverageLoss, \
    OnlineLoss
from src.results_helper.metrics import get_avg_miscoverage_streak_len
from src.uncertainty_quantifiers.prediction_interval import PredictionInterval, PredictionIntervalsCollection
from src.uncertainty_quantifiers.prediction_intervals_matrix import PredictionIntervalsMatrix
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet


class LossFactory(abc.ABC):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def generate(self) -> Loss:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass


class OnlineLossFactory(LossFactory, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def generate(self) -> OnlineLoss:
        pass


class MiscoverageLossFactory(OnlineLossFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self) -> MiscoverageLoss:
        return MiscoverageLoss()

    @property
    def name(self) -> str:
        return "miscoverage"


class MiscoverageCounterLossFactory(OnlineLossFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self) -> MiscoverageCounterLoss:
        return MiscoverageCounterLoss()

    @property
    def name(self) -> str:
        return "miscoverage_counter"


class ImageMiscoverageLossFactory(OnlineLossFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self) -> ImageMiscoverageLoss:
        return ImageMiscoverageLoss()

    @property
    def name(self) -> str:
        return "image_miscoverage"


class PoorCenterCoverageLossFactory(OnlineLossFactory):
    def __init__(self, poor_coverage_threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.poor_coverage_threshold = poor_coverage_threshold

    def generate(self) -> PoorCenterCoverageLoss:
        return PoorCenterCoverageLoss(self.poor_coverage_threshold)

    @property
    def name(self) -> str:
        return 'poor_center_coverage'
#
