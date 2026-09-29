import os
import random
import numpy as np
import torch

def set_seed(seed: int=13) -> None:

    """
        set the seed for reproducibility purposes
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad
def compute_mae(predictions: torch.Tensor, target: torch.Tensor) -> float:

    """
        computes Mean Absolute Error between the predicted ages and the target
    """

    return torch.abs(predictions.view(-1) - target.view(-1)).mean().item()

@torch.no_grad()
def compute_cumulative_score(predictions: torch.Tensor, target: torch.Tensor, threshold: int=5) -> float:

    """
    Cumulative Score (CS) - proportion of samples with |pred - true| <= threshold

    CS@5 is a common secondary metric for age estimation benchmarks

    :return: cumulative score in [0, 1]
    """

    errors = torch.abs(predictions.view(-1) - target.view(-1))
    return (errors <= threshold).float().mean().item()