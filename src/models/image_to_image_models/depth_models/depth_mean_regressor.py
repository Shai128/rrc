from argparse import Namespace

import torch
from torch import nn

from src.data_utils.data_sample.depth_data_sample import DepthDataSample
from src.data_utils.data_sample.i2i_data_sample import LabeledI2IDataSample
from src.model_prediction.mean_regressor_prediction import MeanRegressorPrediction
from src.models.image_to_image_models.abstract_models.i2i_mean_regressor import I2IMeanRegressor
from src.models.image_to_image_models.depth_models.Leres.lib.configs.config import cfg
from src.models.image_to_image_models.depth_models.Leres.lib.models.multi_depth_model_auxiv2 import RelDepthModel, \
    ModelOptimizer
from src.models.image_to_image_models.depth_models.Leres.lib.utils.evaluate_depth_error import recover_metric_depth
from src.models.image_to_image_models.depth_models.Leres.lib.utils.lr_scheduler_custom import \
    make_lr_scheduler_from_cfg
from src.models.image_to_image_models.depth_models.Leres.lib.utils.net_tools import load_model_ckpt


class DepthMeanRegressor(I2IMeanRegressor):

    def __init__(self, trained_model_path: str, backbone: str, device, **kwargs):
        super().__init__(**kwargs)
        self._network: RelDepthModel = RelDepthModel(device, backbone=backbone)
        self.backbone = backbone
        self._optimizer = ModelOptimizer(self.network)
        self._scheduler = make_lr_scheduler_from_cfg(cfg=cfg, optimizer=self.optimizer.optimizer)
        train_args = Namespace(load_ckpt=trained_model_path, resume=True)
        load_model_ckpt(train_args, self.network, None, None)
        self.device = device

    def loss(self, data: LabeledI2IDataSample, estimated_mean: MeanRegressorPrediction=None, **kwargs):
        logit = estimated_mean.mean_estimate if estimated_mean is not None else None
        out = self.network.forward(data.__dict__, is_train=True, logit=logit)
        return out['losses']['total_loss']

    @property
    def name(self) -> str:
        return f'depth_qr_backbone={self.backbone}'

    def predict(self, data_sample: DepthDataSample) -> MeanRegressorPrediction:
        out = self.network.inference(data_sample.__dict__)
        pred_depth = out['pred_depth'].squeeze().detach()
        pred_depth = recover_metric_depth(pred_depth.squeeze(), data_sample.sparse_y.squeeze(), data_sample.feature_points_mask)
        return MeanRegressorPrediction(torch.Tensor(pred_depth).cpu())

    @property
    def network(self) -> nn.Module:
        return self._network

    @property
    def optimizer(self) -> ModelOptimizer:
        return self._optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self._scheduler
