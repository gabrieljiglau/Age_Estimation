from typing import Any

import torch
import torch.nn as nn


class MeanVarianceLoss(nn.Module):

    """
        mean_loss = λ_mean * max[(E[age] - ground_truth)]
        var_loss = λ_loss * Var(age)
        total_loss = mean_loss + var_loss
    """

    def __init__(
            self,
            lambda_mean: float=0.2,
            lambda_variance: float=0.05,
            start_age: int=1,
            end_age: int=80
    ) -> None:

        super(MeanVarianceLoss, self).__init__()

        self.lambda_mean = lambda_mean
        self.lambda_variance = lambda_variance

        self.ages = torch.arange(start_age, end_age + 1, dtype=torch.float32)
        self.register_buffer("ages", self.ages)

    def compute_loss(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
            the network predicts a distribution over ages
            we will track the mean and variance of that distribution and aim to minimize it
        """
        # view(a, b)
        # a = -1 if you don't know the number of dimensions beforehand
        # b = the size of that specific dimensions;
        # in this case, the tensor is 2D, therefore, view(-1, 1) means 'flatten it into a 1D column vector'

        target = target.view(-1, 1)
        probs = torch.softmax(logits, dim=1)

        expected_age = torch.sum(self.ages * probs, dim=1, keepdim=True)
        mean_loss = torch.abs(expected_age - target).mean()

        expected_age_squared = torch.sum((self.ages ** 2) * probs, dim=1, keepdim=True)
        var_loss = torch.abs(expected_age_squared - expected_age ** 2).mean()

        total_loss = self.lambda_mean * mean_loss + self.lambda_variance * var_loss
        return total_loss, mean_loss, var_loss

    @torch.no_grad
    def predict_age(self, logits: torch.Tensor) -> torch.Tensor:

        """
            :returns: the expected age based on their distribution (found after applying softmax)
        """

        probs = torch.softmax(logits, dim=1)
        return torch.sum(self.ages * probs, dim=1)