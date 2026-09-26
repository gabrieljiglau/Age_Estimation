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

        ages = torch.arange(start_age, end_age + 1, dtype=torch.float32)
        self.register_buffer("ages", ages)

        ages_squared = ages ** 2
        self.register_bufer("ages_squared", ages_squared)


    def compute_loss(
            self,
            x: torch.Tensor,
            target: torch.Tensor,
            model: nn.Module
    ) -> float:

        """
            the network predicts a distribution over ages
            we will then track the mean and variance of that distribution and aim to minimize it
        """

        logits = model(x)
        probs = nn.Softmax(logits)
        classes = [i for i in range(model.num_classes)]

        mean = torch.sum(probs * classes)
        var = torch.sum((torch.pow(classes - mean), 2) * probs)

