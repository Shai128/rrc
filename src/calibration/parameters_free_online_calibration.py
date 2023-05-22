from typing import List
from ConfigSpace import ConfigurationSpace, Configuration
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from src.calibration.calibrations_factory import OnlineCalibrationFactory
from src.calibration.online_calibration import OnlineCalibration
from src.uncertainty_quantifiers.uncertainty_set import UncertaintySet
from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample
from src.losses import ObjectiveLoss
from src.model_prediction.model_prediction import ModelPrediction


class ParameterFreeOnlineCalibration(OnlineCalibration):

    def __init__(self, objective_loss: ObjectiveLoss, online_calibration_factory: OnlineCalibrationFactory, **kwargs):
        self.calibration_scheme_factory = online_calibration_factory
        self.calibration_scheme = online_calibration_factory.generate(**kwargs)
        self._name = f"parameter_free_{self.calibration_scheme_factory.name}"
        self.objective_loss = objective_loss
        self.labeled_samples = {}
        self.model_predictions = {}

    def calibrate(self, data_sample: DataSample, model_prediction: ModelPrediction, **kwargs) -> UncertaintySet:
        return self.calibration_scheme.calibrate(data_sample, model_prediction, **kwargs)

    def update(self, labeled_sample: LabeledDataSample, model_prediction: ModelPrediction,
               calibrated_uncertainty_set: UncertaintySet, timestamp : int, **kwargs):
        self.labeled_samples[timestamp] = labeled_sample
        self.model_predictions[timestamp] = model_prediction
        return self.calibration_scheme.update(labeled_sample, model_prediction, calibrated_uncertainty_set,
                                              timestamp=timestamp, **kwargs)

    @property
    def name(self) -> str:
        return self._name  # self.calibration_scheme.name

    def compute_objective(self, calibration_scheme: OnlineCalibration, validation_timestamps: List[int]):
        calibrated_sets = []
        for t in validation_timestamps:
            sample = self.labeled_samples[t]
            model_prediction = self.model_predictions[t]
            calibrated_set = calibration_scheme.calibrate(sample, model_prediction)
            calibration_scheme.update(sample, model_prediction, calibrated_set, t)
            calibrated_sets += [calibrated_set]
        labeled_samples = [self.labeled_samples[t] for t in validation_timestamps]
        loss = self.objective_loss.batch_loss(calibrated_sets, labeled_samples)
        return {'loss': loss}

    def set_hyperparameters_on_validation_set(self, validation_timestamps: List[int]):
        config_space = ConfigurationSpace()
        hyperparameters = self.calibration_scheme_factory.get_hyperparameters()
        for hyperparameter in hyperparameters:
            config_space.add_hyperparameter(hyperparameter)

        scenario = Scenario(
            configspace=config_space,
            deterministic=True,  # Only one seed
            n_trials=50,
            n_workers=1,
            # walltime_limit=60*10,
            objectives=["loss"]
        )

        def run_on_config(config: Configuration, seed : int = 0):
            calibration_scheme = self.calibration_scheme_factory.generate(**config)
            return self.compute_objective(calibration_scheme, validation_timestamps)

        smac = HPOFacade(
            scenario=scenario,
            target_function=run_on_config,
            overwrite=True,
        )
        incumbent = smac.optimize()
        self.calibration_scheme = self.calibration_scheme_factory.generate(**incumbent)

        for t in validation_timestamps:
            sample = self.labeled_samples[t]
            model_prediction = self.model_predictions[t]
            calibrated_set = self.calibration_scheme.calibrate(sample, model_prediction)
            self.calibration_scheme.update(sample, model_prediction, calibrated_set, timestamp=t)
