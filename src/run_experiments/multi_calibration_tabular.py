import warnings
from sys import platform
import sys


sys.path.append("../")

import matplotlib
from src.losses_factory import MiscoverageLossFactory, MiscoverageCounterLossFactory
from src.run_experiments.objective_losses import RegressionObjectiveLoss, MCRegressionObjectiveLoss
from src.main_tabular import parse_args, get_model, get_results_handler
from multi_calibration_helper import run_experiments
from src.calibration.calibration_schemes import CalibrationSchemes
from src.calibration.calibrations_factory import RollingRiskControlFactory, ACIFactory
from src.calibration.dummy_calibration import DummyCalibration
from src.calibration.parameters_free_online_calibration import ParameterFreeOnlineCalibration
from src.calibration.set_constructing_functions.set_constructing_function_factory import \
    PredictionIntervalConstructingFunctionWithCQRFactory
from src.calibration.stretching_functions.stretching_functions_factory import ExponentialStretchingFactory, \
    IdentityStretchingFactory, ErrorAdaptiveStretchingFactory
from src.data_utils.get_dataset_utils import get_dataset
from src.losses import MiscoverageLoss, MiscoverageCounterLoss

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")


def get_calibration_schemes(alpha, calibration_set_size) -> CalibrationSchemes:
    calibration_schemes = []
    reg_objective_loss = RegressionObjectiveLoss(alpha)
    mc_alpha = alpha / (1 - alpha)
    mc_objective_loss = MCRegressionObjectiveLoss(mc_alpha)
    for curr_alpha, objective_loss, curr_loss_factory in [
        (alpha, reg_objective_loss, MiscoverageLossFactory()),
        (mc_alpha, mc_objective_loss, MiscoverageCounterLossFactory())
    ]:
        for stretching_factory in [
            IdentityStretchingFactory(),
            ExponentialStretchingFactory(),
            ErrorAdaptiveStretchingFactory(curr_loss_factory, curr_alpha)
        ]:
            set_constructing_function_factory = PredictionIntervalConstructingFunctionWithCQRFactory(stretching_factory)
            rrc_factory = RollingRiskControlFactory(curr_alpha, set_constructing_function_factory, curr_loss_factory)
            calibration = ParameterFreeOnlineCalibration(objective_loss, rrc_factory)
            calibration_schemes.append(calibration)

    aci_factory = ACIFactory(alpha, calibration_set_size)
    param_free_aci = ParameterFreeOnlineCalibration(reg_objective_loss, aci_factory)
    calibration_schemes.append(param_free_aci)
    calibration_schemes.append(DummyCalibration())
    return CalibrationSchemes(calibration_schemes)


def main():
    args = parse_args()
    dataset = get_dataset(args)
    model = get_model(args, dataset)
    calibration_schemes = get_calibration_schemes(args.alpha, args.calibration_set_size)
    results_handler = get_results_handler(dataset.data_type, args.base_results_save_dir, args.seed)
    run_experiments(args.task, args.device, model, dataset, calibration_schemes, results_handler, args.training_epochs)


if __name__ == '__main__':
    main()
