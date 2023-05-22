import traceback
from typing import List

import numpy as np
import torch

from src.calibration.online_calibration import OnlineCalibration
from src.calibration.parameters_free_online_calibration import ParameterFreeOnlineCalibration
from src.data_utils.datasets.regression_dataset import RegressionDataset
from src.models.abstract_models.online_learning_model import OnlineLearningModel
from src.results_helper.metrics import get_avg_miscoverage_streak_len, get_avg_miscoverage_counter, \
    calculate_pearson_corr, \
    calculate_coverage, calculate_average_length, calculate_median_length
from src.results_helper.results_helper import ResultsHelper
from src.uncertainty_quantifiers.prediction_interval import PredictionIntervalsCollection


class RegressionResultsHelper(ResultsHelper):

    def __init__(self, base_results_save_dir, seed):
        super().__init__(base_results_save_dir, seed)

    def compute_performance_metrics(self, uncertainty_set_collection: PredictionIntervalsCollection,
                                    dataset: RegressionDataset, timestamps: List[int], model: OnlineLearningModel,
                                    calibration_scheme: OnlineCalibration) -> dict:
        base_results = super().compute_performance_metrics(uncertainty_set_collection, dataset, timestamps,
                                                           model, calibration_scheme)
        intervals = uncertainty_set_collection.unscaled_intervals
        lower_q_pred, upper_q_pred = intervals[:, 0], intervals[:, 1]
        y = dataset.unscaled_y[timestamps].squeeze()
        x = dataset.unscaled_x[timestamps]
        results = self.compute_performance_metrics_aux(x, y, upper_q_pred, lower_q_pred)
        return {**base_results, **results}

    def compute_performance_metrics_aux(self, x, y, upper_q_pred, lower_q_pred, **kwargs) -> dict:
        results = {}
        try:
            results['average miscoverage streak length'] = get_avg_miscoverage_streak_len(y, upper_q_pred, lower_q_pred)
            results['average miscoverage counter'] = get_avg_miscoverage_counter(y, upper_q_pred, lower_q_pred)
            results['coverage'] = calculate_coverage(y, upper_q_pred, lower_q_pred)
            results['average length'] = calculate_average_length(upper_q_pred, lower_q_pred)
            results['median length'] = calculate_median_length(upper_q_pred, lower_q_pred)
            results['corr'] = calculate_pearson_corr(y, upper_q_pred, lower_q_pred)
        except:
            traceback.print_exc()
        return results


# TODO: complete
class SyntheticDataRegressionResultsHelper(RegressionResultsHelper):
    def __init__(self, base_results_save_dir, seed):
        super().__init__(base_results_save_dir, seed)


class RealDataRegressionResultsHelper(RegressionResultsHelper):

    def compute_performance_metrics_aux(self, x, y, upper_q_pred, lower_q_pred, **kwargs) -> dict:
        results = super().compute_performance_metrics_aux(x, y, upper_q_pred, lower_q_pred, **kwargs)

        has_all_days = torch.unique(x[:, -1]).squeeze().shape[0] == 7
        if has_all_days:
            days_coverages = []
            for day in range(0, 7):
                idx = x[:, -1] == day
                coverage = calculate_coverage(y[idx], upper_q_pred[idx], lower_q_pred[idx])
                days_coverages += [coverage]
                results[f'coverage in day {day}'] = coverage
            days_coverages = np.array(days_coverages)
            results['days avg. Δ-coverage'] = np.abs((days_coverages - results[f'coverage'])).mean()
            results['days worst. Δ-coverage'] = np.abs((days_coverages - results[f'coverage'])).max()
            results['days Δ-coverage std.'] = np.abs((days_coverages - results[f'coverage'])).std()
        return results

