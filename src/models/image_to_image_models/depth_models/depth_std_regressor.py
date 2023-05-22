import argparse

import numpy as np
import torch
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
from torch import nn

from src.data_utils.data_sample.depth_data_sample import DepthStdRegressionDataSample, \
    LabeledDepthStdRegressionDataSample
from src.model_prediction.i2i_model_prediction import I2IStdPrediction
from src.models.image_to_image_models.abstract_models.i2i_std_regressor import I2IStdRegressor, dummy_loss
from src.models.image_to_image_models.abstract_models.online_i2i_model import OnlineI2ILearningModel
from src.models.image_to_image_models.depth_models.Leres.data.multi_dataset import get_feature_points_mask
from src.models.image_to_image_models.depth_models.Leres.lib.models.multi_depth_model_auxiv2 import ModelOptimizer, \
    DepthModel, RelDepthModel
from src.models.image_to_image_models.depth_models.Leres.lib.utils.evaluate_depth_error import recover_metric_depth
from src.models.image_to_image_models.depth_models.Leres.lib.utils.lr_scheduler_custom import make_lr_scheduler
from src.models.image_to_image_models.depth_models.Leres.lib.utils.net_tools import load_model_ckpt


def register_using_optical_flow(image0, image1):
    device = 'cpu'
    if torch.is_tensor(image0):
        device = image0.device
        image0 = image0.cpu().detach().numpy()
    if torch.is_tensor(image1):
        image1 = image1.cpu().detach().numpy()
    nr, nc = image0.shape
    v, u = optical_flow_tvl1(image1, image0, num_iter=2, num_warp=1)
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')
    image0_warp = warp(image0, np.array([row_coords + v, col_coords + u]), mode='edge')

    return torch.from_numpy(image0_warp).to(device)


class StdRegressionByAlignedPreviousResiduals(I2IStdRegressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_depths = []
        self.prev_depth_estimates = []

    def online_fit(self, labeled_data_sample: LabeledDepthStdRegressionDataSample,
                   labeled_inference_data_sample: LabeledDepthStdRegressionDataSample = None, **kwargs):
        assert labeled_inference_data_sample is not None
        pad = labeled_inference_data_sample.augmentation_info.pad
        estimated_mean = labeled_inference_data_sample.mean_estimate.mean_estimate.to(self.device)
        m, n = estimated_mean.shape
        self.prev_depths += [
            labeled_inference_data_sample.y.to(self.device).squeeze()[pad[0]: m - pad[1], pad[2]: n - pad[3]]]
        self.prev_depth_estimates += [estimated_mean.squeeze()[pad[0]: m - pad[1], pad[2]: n - pad[3]]]
        self.prev_depths = self.prev_depths[-5:]
        self.prev_depth_estimates = self.prev_depth_estimates[-5:]

    def predict(self, data_sample: DepthStdRegressionDataSample) -> I2IStdPrediction:
        pad = data_sample.augmentation_info.pad
        estimated_mean = data_sample.mean_estimate.mean_estimate.squeeze().to(self.device)
        m, n = estimated_mean.shape
        residual = (estimated_mean - data_sample.sparse_y.to(self.device))[pad[0]: m - pad[1], pad[2]: n - pad[3]]
        curr_l = residual * (residual > 0)
        curr_u = - residual * (residual < 0)
        if len(self.prev_depths) == 0:
            l = u = 1
        else:
            # TODO: align previous depths to current timestamp's depth by registering prev grayscale image to curr grayscale image
            for i in range(len(self.prev_depths) - 1):
                self.prev_depths[i] = register_using_optical_flow(self.prev_depths[i], self.prev_depths[-1])
                self.prev_depth_estimates[i] = register_using_optical_flow(self.prev_depth_estimates[i],
                                                                           self.prev_depth_estimates[-1])
            prev_residuals = torch.stack(self.prev_depth_estimates) - torch.stack(self.prev_depths)

            l = (prev_residuals * (prev_residuals > 0)).mean(dim=0)
            u = (- prev_residuals * (prev_residuals < 0)).mean(dim=0)

            mask = data_sample.feature_points_mask.squeeze()[pad[0]: m - pad[1], pad[2]: n - pad[3]]
            if curr_l[mask].sum() > 20 and curr_u[mask].sum() > 20:
                l = torch.from_numpy(recover_metric_depth(l, curr_l, mask0=mask)).to(l.device)
                u = torch.from_numpy(recover_metric_depth(u, curr_u, mask0=mask)).to(u.device)
            l_padded = torch.zeros_like(estimated_mean)
            u_padded = torch.zeros_like(estimated_mean)
            l_padded[pad[0]: m - pad[1], pad[2]: n - pad[3]] = l
            u_padded[pad[0]: m - pad[1], pad[2]: n - pad[3]] = u
            l, u = l_padded, u_padded
            l = l.cpu()
            u = u.cpu()

        heuristics = I2IStdPrediction(l, u)

        return heuristics

    @property
    def name(self):
        return "aligned_prev_resid"


class ResidualMagnitudeRegressor(I2IStdRegressor, OnlineI2ILearningModel):

    def __init__(self, backbone, device, loss_batch_size, **kwargs):
        super().__init__(device, **kwargs)

        network = RelDepthModel(device, backbone).to(device).depth_model
        modules = list(network.decoder_modules.outconv.adapt_conv.children())[:-2]
        network.decoder_modules.outconv.adapt_conv = torch.nn.Sequential(*modules)
        pre_trained_network_without_last_layer = network
        model_out_channels = network.decoder_modules.midchannels[0] // 2

        self._network = torch.nn.Sequential(pre_trained_network_without_last_layer.to(device),
                                            nn.Conv2d(model_out_channels, 1, kernel_size=3, padding=1, stride=1,
                                                      bias=True).to(device),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).to(device))
        # load_model_ckpt(argparse.Namespace(load_ckpt=trained_std_estimator_path, resume=True), self._network, None, None)
        self._optimizer = ModelOptimizer(self)
        self._scheduler = make_lr_scheduler(optimizer=self.optimizer.optimizer)
        self.loss_batch_size = loss_batch_size

    @property
    def network(self) -> torch.nn.Module:
        return self._network

    def predict(self, data_sample: DepthStdRegressionDataSample) -> I2IStdPrediction:
        estimated_residual = self.forward(data_sample).squeeze().to(self.device)
        sparse_gt_residual = (data_sample.sparse_y.squeeze().to(self.device) -
                              data_sample.mean_estimate.mean_estimate.squeeze().to(self.device)).abs()
        sparsity_mask = data_sample.feature_points_mask.squeeze().to(self.device)
        residual = recover_metric_depth(estimated_residual, sparse_gt_residual, mask0=sparsity_mask)
        residual = torch.Tensor(residual).cpu()
        return I2IStdPrediction(residual.squeeze(), residual.squeeze())

    def forward(self, data: DepthStdRegressionDataSample):
        network_out = self.network(data.x.unsqueeze(0).to(self.device))
        return network_out

    def loss(self, data: LabeledDepthStdRegressionDataSample, **kwargs):
        batch_size = self.loss_batch_size
        y = data.y.squeeze()
        feature_points_mask = get_feature_points_mask(data.valid_image_mask.squeeze(),
                                                      y.cpu().detach().numpy(), batch_size)
        if feature_points_mask.float().sum() <= 10:
            return dummy_loss(y.device)
        pred = self.forward(data).squeeze()
        estimated_mean = data.mean_estimate.mean_estimate.detach().squeeze()[feature_points_mask]
        ground_truth = y[feature_points_mask].to(self.device)
        gt_residual = (estimated_mean - ground_truth).abs()
        return ((pred[feature_points_mask] - gt_residual).abs()).mean()

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def name(self) -> str:
        return "residual_magnitude"

    @property
    def scheduler(self):
        return self._scheduler
