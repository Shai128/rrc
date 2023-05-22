import warnings
from sys import platform
from typing import List
import sys
import matplotlib

sys.path.append("../")
sys.path.append("../../")

from src.losses_factory import ImageMiscoverageLossFactory, PoorCenterCoverageLossFactory
from src.run_experiments.objective_losses import ImageMiscoverageObjectiveLoss, MultipleRisksObjectiveLoss
from src.calibration.online_calibration import OnlineCalibration
from src.calibration.set_constructing_functions.set_constructing_function import \
    MaxAggregation
from src.results_helper.depth_results_helper import DepthResultsHelper
from src.main_depth import parse_args, get_model
from multi_calibration_helper import run_experiments
from src.calibration.calibration_schemes import CalibrationSchemes
from src.calibration.calibrations_factory import RollingRiskControlWithMultipleRisksFactory, DummyI2ICalibrationFactory
from src.calibration.parameters_free_online_calibration import ParameterFreeOnlineCalibration
from src.calibration.set_constructing_functions.set_constructing_function_factory import \
    PredictionIntervalMatrixConstructingFunctionWithMeanAndStdFactory
from src.calibration.stretching_functions.stretching_functions_factory import ExponentialStretchingFactory
from src.data_utils.get_dataset_utils import get_depth_dataset

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")


def get_single_risk_calibration_schemes(device) -> List[OnlineCalibration]:
    stretching_factory = ExponentialStretchingFactory()
    set_constructing_factory = PredictionIntervalMatrixConstructingFunctionWithMeanAndStdFactory(stretching_factory,
                                                                                                 MaxAggregation())
    alpha = 0.2
    alphas = [alpha]
    losses_factories = [ImageMiscoverageLossFactory()]
    rrc_factory = RollingRiskControlWithMultipleRisksFactory(alphas, set_constructing_factory, losses_factories)
    rrc_calibration = ParameterFreeOnlineCalibration(ImageMiscoverageObjectiveLoss(alpha, device), rrc_factory)
    dummy_calibration = ParameterFreeOnlineCalibration(ImageMiscoverageObjectiveLoss(alpha, device),
                                                       DummyI2ICalibrationFactory())
    calibration_schemes = [rrc_calibration, dummy_calibration]
    return calibration_schemes


def get_multi_single_risk_calibration_schemes(device) -> List[OnlineCalibration]:
    stretching_factory = ExponentialStretchingFactory()
    set_constructing_factory = PredictionIntervalMatrixConstructingFunctionWithMeanAndStdFactory(stretching_factory,
                                                                                                 MaxAggregation())
    image_miscoverage_alpha, poor_center_coverage_alpha = 0.2, 0.1
    alphas = [image_miscoverage_alpha, poor_center_coverage_alpha]
    losses_factories = [ImageMiscoverageLossFactory(), PoorCenterCoverageLossFactory(0.6)]
    rrc_factory = RollingRiskControlWithMultipleRisksFactory(alphas, set_constructing_factory, losses_factories)
    rrc_calibration = ParameterFreeOnlineCalibration(
        MultipleRisksObjectiveLoss(image_miscoverage_alpha, poor_center_coverage_alpha, device=device), rrc_factory)
    return [rrc_calibration]


def get_calibration_schemes(device) -> CalibrationSchemes:
    return CalibrationSchemes(get_single_risk_calibration_schemes(device) + get_multi_single_risk_calibration_schemes(device))


def main():
    args = parse_args()
    dataset = get_depth_dataset(args)
    model = get_model(args)
    calibration_schemes = get_calibration_schemes(args.device)
    results_handler = DepthResultsHelper(args.base_results_save_dir, args.seed)
    run_experiments(args.task, args.device, model, dataset, calibration_schemes, results_handler, args.training_epochs)


if __name__ == '__main__':
    main()
