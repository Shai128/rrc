import traceback
from typing import List

import numpy as np
import torch

from src.calibration.online_calibration import OnlineCalibration
from src.calibration.parameters_free_online_calibration import ParameterFreeOnlineCalibration
from src.data_utils.datasets.depth_dataset import DepthDataset
from src.models.image_to_image_models.depth_models.depth_uq_model import DepthUQModel
from src.models.online_qr import batch_pinball_loss
from src.results_helper.results_helper import ResultsHelper
from src.uncertainty_quantifiers.prediction_intervals_matrix import PredictionIntervalsMatrixCollection

ALPHA = 0.1


class DepthResultsHelper(ResultsHelper):
    def __init__(self, base_results_save_dir, seed):
        super().__init__(base_results_save_dir, seed)

    def compute_performance_metrics(self, uncertainty_set_collection: PredictionIntervalsMatrixCollection,
                                    dataset: DepthDataset,
                                    timestamps: List[int], model: DepthUQModel,
                                    calibration_scheme: OnlineCalibration) -> dict:
        base_results = super().compute_performance_metrics(uncertainty_set_collection, dataset, timestamps,
                                                           model, calibration_scheme)
        intervals = uncertainty_set_collection.unscaled_intervals.cpu()
        # estimated_means = uncertainty_set_collection.unscaled_mean
        samples = [dataset.get_labeled_inference_sample(t) for t in timestamps]
        valid_idx_mask = torch.stack([sample.valid_image_mask for sample in samples]).cpu()
        image_center = torch.stack([torch.Tensor(sample.center_pixel) for sample in samples]).cpu()
        scaling_factor = torch.Tensor([sample.scaling_factor for sample in samples]).cpu()
        meters_factor = torch.Tensor([sample.meters_factor for sample in samples]).cpu()
        y = torch.stack([sample.y for sample in samples]).cpu()
        results = self.compute_performance_metrics_aux(intervals, y, valid_idx_mask, image_center, scaling_factor,
                                                       meters_factor, ALPHA)
        return {**base_results, **results}


    def compute_performance_metrics_aux(self, intervals, y, valid_idx_mask, image_center, scaling_factor, meters_factor,
                                        alpha) -> dict:
        results = {}
        image_center = image_center.int()
        for image_part in ['image', 'center', 'non center']:
            if image_part == 'image':
                relevant_idxs = torch.ones_like(valid_idx_mask)
            elif image_part == 'center':
                relevant_idxs = torch.zeros_like(valid_idx_mask)
                for i in range(relevant_idxs.shape[0]):
                    relevant_idxs[i, image_center[i, 0].item() - 25: image_center[i, 0].item() + 25,
                    image_center[i, 1].item() - 25: image_center[i, 1].item() + 25] = True
            elif image_part == 'non center':
                relevant_idxs = torch.ones_like(valid_idx_mask, dtype=torch.bool)
                for i in range(relevant_idxs.shape[0]):
                    relevant_idxs[i, image_center[i, 0].item() - 25: image_center[i, 0].item() + 25,
                    image_center[i, 1].item() - 25: image_center[i, 1].item() + 25] = False
            else:
                raise Exception("error, invalid image part")
            part_results = self.compute_result_per_part(intervals, y, valid_idx_mask, scaling_factor, meters_factor,
                                                        relevant_idxs, alpha)
            for key, val in part_results.items():
                results[f'{image_part} {key}'] = val
        return results

    def compute_result_per_part(self, intervals, y, valid_idx_mask, scaling_factor, meters_factor, relevant_idxs,
                                alpha):
        original_y = y
        original_upper_q = intervals[..., 1]
        original_lower_q = intervals[..., 0]
        original_valid_idx_mask = valid_idx_mask
        coverages = []
        lengths = []
        pb_losses = []
        pixelwise_coverage_matrix = torch.zeros_like(original_y[0].squeeze()).cpu()
        pixelwise_n_valid_matrix = torch.zeros_like(original_y[0].squeeze()).cpu()
        avg_unscaled_y = []
        max_unscaled_y = []
        for i in range(0, original_y.shape[0]):
            valid_idx_mask = original_valid_idx_mask[i].to(y.device).squeeze()
            relevant_idx = relevant_idxs[i]
            relevant_idx = relevant_idx & valid_idx_mask
            y = original_y[i].squeeze()
            upper_q = original_upper_q[i].squeeze()
            lower_q = original_lower_q[i].squeeze()

            v = ((y <= upper_q) & (y >= lower_q)).float() * relevant_idx.float()
            pixelwise_coverage_matrix += v.cpu().squeeze()
            pixelwise_n_valid_matrix += relevant_idx

            y = y[relevant_idx].squeeze()
            upper_q = upper_q[relevant_idx].squeeze()
            lower_q = lower_q[relevant_idx].squeeze()

            coverages += [((y <= upper_q) & (y >= lower_q)).float().mean().item()]
            lengths += [(meters_factor[i] * (upper_q[i] - lower_q[i]).mean().item() / scaling_factor[i])]

            lower_pb = batch_pinball_loss(alpha / 2, lower_q, y)
            upper_pb = batch_pinball_loss(1 - alpha / 2, upper_q, y)
            pb_loss = (lower_pb + upper_pb) / 2
            pb_losses += [pb_loss.item()]

            try:
                avg_unscaled_y += [(meters_factor[i] * y.mean() / scaling_factor[i]).item()]
                max_unscaled_y += [(meters_factor[i] * y.max() / scaling_factor[i]).item()]
            except:
                traceback.print_exc()
                print("failed calculating max unscaled y")

        coverages = np.array(coverages) * 100
        delta_coverage = np.mean(np.abs(coverages - (1 - alpha) * 100))
        actual_delta_coverage = np.mean(np.abs(coverages - np.mean(coverages)))
        max_delta_coverage = np.max(np.abs(coverages - (1 - alpha) * 100))
        max_actual_delta_coverage = np.max(np.abs(coverages - np.mean(coverages)))

        avg_pixelwise_coverage = (pixelwise_coverage_matrix / pixelwise_n_valid_matrix)[pixelwise_n_valid_matrix > 50]
        if len(avg_pixelwise_coverage) == 0:
            avg_pixelwise_coverage = (pixelwise_coverage_matrix / pixelwise_n_valid_matrix)

        pixelwise_delta_coverage = (avg_pixelwise_coverage - (1 - alpha)).abs().mean().item() * 100
        pixelwise_actual_delta_coverage = (avg_pixelwise_coverage - avg_pixelwise_coverage.mean()).abs().mean().item() * 100
        pixelwise_max_delta_coverage = (avg_pixelwise_coverage - (1 - alpha)).abs().max().item() * 100
        pixelwise_max_actual_delta_coverage = (avg_pixelwise_coverage - avg_pixelwise_coverage.mean()).abs().max().item() * 100

        results = {}
        results['pb loss'] = np.mean(pb_losses)
        results[f'coverage'] = np.mean(coverages)
        results[f'average length'] = np.mean(lengths)
        results[f'median length'] = np.median(lengths)
        results[f'delta coverage'] = delta_coverage
        results[f'actual delta coverage'] = actual_delta_coverage
        results[f'max delta coverage'] = max_delta_coverage
        results[f'max actual delta coverage'] = max_actual_delta_coverage
        results[f'pixel-wise Δ-coverage'] = pixelwise_delta_coverage
        results[f'max pixel-wise Δ-coverage'] = pixelwise_max_delta_coverage
        results[f'actual pixel-wise Δ-coverage'] = pixelwise_actual_delta_coverage
        results['actual max pixel-wise Δ-coverage'] = pixelwise_max_actual_delta_coverage
        results[f'max depth'] = np.max(max_unscaled_y)
        results[f'average depth'] = np.mean(avg_unscaled_y)

        for rate in [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.78, 0.79, 0.8, 0.85, 0.9, 0.95]:
            results[f'poor coverage occurrences ({rate * 100} %)'] = (np.array(coverages) < 100 * rate).astype(
                np.float32).mean().item() * 100

        return results
