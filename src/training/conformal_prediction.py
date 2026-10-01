import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ConformaRegressor:

    def __init__(self, model: nn.Module):

        self.model = model

    def calibrate(self, calibration_set: DataLoader) -> list[float]:
        """ Computes the residuals from the provided calibration set

        :param calibration_set: the set for which the residuals will be computed
        :return: the list of residuals
        """
        pass

    def predict_range(self, instance: torch.Tensor, confidence: float) -> tuple[float, float]:
        """ Makes the final age-prediction, taking into account the computed residuals

        :param instance: the image used for prediction
        :param confidence: a float in [0, 1] that specifies the range of residuals that will be added to the prediction
        :return: a lower bound and upper bound for the prediction
        """