import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ConformaRegressor:

    def __init__(self, model: nn.Module, loss_function: nn.Module):

        self.model = model
        self.loss_function = loss_function


    def calibrate(self, calibration_set: DataLoader) -> tuple[list[float], int]:

        """ Computes the residuals from the provided calibration set

        :param calibration_set: the set for which the residuals will be computed
        :return: the list of residuals and the length of the calibration dataset
        """

        residuals = []
        for image, target, _, _ in calibration_set:

            logits = self.model(image)
            pred_age, variance = self.loss_function.predict_age(logits)

            normalized_score = torch.abs(pred_age - target) / variance
            residuals.append(normalized_score.item())

        return residuals, len(calibration_set.dataset)

    def predict_age_range(
            self,
            instance: torch.Tensor,
            residuals: list[float],
            num_samples: int,
            coverage: float=0.9
    ) -> tuple[float, float]:

        """ Makes the final prediction, taking into account the computed residuals

        :param instance: image-instance used for prediction
        :param residuals: the list of normalized errors from the calibration set
        :param num_samples: the length of the calibration set
        :param coverage: a float in [0, 1] that specifies the percentage of errors that the algorithm takes into account
        :return: a lower bound and upper bound for the prediction
        """

        sorted_residuals = sorted(residuals, reverse=False)
        k_idx = round((num_samples + 1) / coverage)
        quantile = sorted_residuals[k_idx]

        logits = self.model(instance)
        pred_age, variance = self.loss_function.predict_age(logits)

        lower_bound = pred_age - quantile * variance
        upper_bound = pred_age + quantile * variance

        return lower_bound, upper_bound