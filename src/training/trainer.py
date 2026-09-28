from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .loss_metric import MeanVarianceLoss
from omegaconf import DictConfig

class Trainer:

    def __init__(
            self,
            model: nn.Module,
            loss_function: nn.Module,
            device: torch.Device,
            cfg: DictConfig
    ) -> None:

        self.model = model
        self.loss_function = loss_function.to(device)
        self.device = device
        self.cfg = cfg

        backbone_params, head_params = self._split_params()
        self.optimizer = torch.optim.AdamW(
            [
                {'backbone': backbone_params, 'lr': cfg.optimizer.backbone_lr},
                {'head': head_params, 'lr': cfg.optimizer.head_lr}
            ],
            weight_decay = cfg.optimizer.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR
                            
        self.scaler = GradScaler(str(device), enabled=True)

        # logging
        self.writer = SummaryWriter(log_dir=cfg.logging.log_dir)

        self.save_dir = Path(cfg.logging.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_mae: float = np.inf
        self.best_checkpoint: Path | None = None

    def _split_params(self):

        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'head' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        return backbone_params, head_params


    def train(
            self,
            num_epochs: int,
            batch_size:int,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            save_path: str
    ) -> None:

        min_loss = np.inf
        self.model.train()
        for epoch in range(1, num_epochs + 1):

            epoch_loss = 0
            for image, target, _, _ in train_loader:

                self.optimizer.zero_grad()  # reset gradient for every batch
                logits = self.model(image)

                current_loss, mean_loss, var_loss = self.loss_function.compute_loss(image, logits)
                current_loss.backward()

                self.optimizer.step()

                epoch_loss += current_loss

            self.scheduler.step()

            print(f"Epoch loss = {epoch_loss / len(train_loader.dataset)}")
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                torch.save(self.model.state_dict(), save_path)










