import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:

    def __init__(
            self,
            model: nn.Module,
            loss_function: nn.Module
    ) -> None:

        self.model = model
        self.loss_function = loss_function