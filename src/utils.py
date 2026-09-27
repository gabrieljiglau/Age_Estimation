import os
import random
import numpy as np
import torch

def set_seed(seed: int=13) -> None:

    """
        set the seed for reproductibility purposes
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
        computes Mean Absolute Error
    """

    return torch.abs(predictions.view(-1) - target.view(-1)).mean().item()