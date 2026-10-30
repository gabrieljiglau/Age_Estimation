from pathlib import Path
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ConformalRegressor:

    """
    Uncertainty quantification method that produces statistically valid
    prediction regions / intervals for the face age estimator
    """

    def __init__(self, model: nn.Module, loss_function: nn.Module):

        self.model = model
        self.loss_function = loss_function


    def calibrate(
            self,
            calibration_set: DataLoader,
            residuals_path: Path | str
    ) -> dict[str, int]:

        """ Computes the residuals from the provided calibration set
            it computes and saves (using pickle) the residuals if they don't already exist,
            otherwise directly reads (loads) them

        :param residuals_path: path to save the computed residuals
        :param calibration_set: the set for which the residuals will be computed
        :return: a dictionary, containing the residuals and the length of the calibration dataset
        """

        residuals_path = Path(residuals_path)
        if residuals_path.exists():
            with open(residuals_path, 'rb') as file:
                dict_info = pickle.load(file=file)

            return dict_info

        residuals = []
        for image, target, _, _ in calibration_set:

            logits = self.model(image)
            pred_age, variance = self.loss_function.predict_age(logits)

            normalized_score = torch.abs(pred_age - target) / variance
            residuals.append(normalized_score.item())

        dict_info = {
            'residuals': residuals,
            'calibration_set_size': len(calibration_set.dataset)
        }

        with open(residuals_path, 'wb') as file:
            pickle.dump(dict_info, file=file)

        return dict_info

    def predict_age_range(
            self,
            instance: torch.Tensor,
            residuals: tuple[float],
            num_samples: int,
            coverage: float=0.9
    ) -> tuple[float, float, float]:

        """ Makes the final prediction, taking into account the computed residuals

        :param instance: image-instance used for prediction
        :param residuals: the list of normalized errors from the calibration set
        :param num_samples: the length of the calibration set
        :param coverage: a float in [0, 1] that specifies the percentage of errors that the algorithm takes into account
        :return: the prediction, as well as a lower bound and upper bound for it
        """

        sorted_residuals = sorted(residuals, reverse=False)
        k_idx = round((num_samples + 1) / coverage)
        quantile = sorted_residuals[k_idx]

        logits = self.model(instance)
        pred_age, variance = self.loss_function.predict_age(logits)

        lower_bound = pred_age - quantile * variance
        upper_bound = pred_age + quantile * variance

        return pred_age, lower_bound, upper_bound