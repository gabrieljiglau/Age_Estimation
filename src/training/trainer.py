import time
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from .loss_metric import MeanVarianceLoss
from torch.utils.data import DataLoader
from ..utils import compute_mae, compute_cumulative_score

class Trainer:

    def __init__(
            self,
            model: nn.Module,
            loss_function: MeanVarianceLoss,
            device: torch.Device,
            cfg: DictConfig
    ) -> None:

        self.model = model
        self.loss_function = loss_function.to(device)
        self.device = device
        self.cfg = cfg

        self.use_amp = torch.cuda.is_available()

        backbone_params, head_params = self._split_params()
        self.optimizer = torch.optim.AdamW(
            [
                {'backbone': backbone_params, 'lr': cfg.optimizer.backbone_lr},
                {'head': head_params, 'lr': cfg.optimizer.head_lr}
            ],
            weight_decay = cfg.optimizer.weight_decay
        )

        self.scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None
                            
        self.scaler = GradScaler(device=str(device), enabled=self.use_amp)

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

    def _build_scheduler(self, steps_per_epoch: int) -> None:

        total_steps = self.cfg.training.epocgs + steps_per_epoch
        max_lrs = [self.cfg.optimizer.backbone_lr, self.cfg.optimizer.head_lr]
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=self.cfg.scheduler.pct_start,
            anneal_strategy='cos'
        )

    def _train_one_epoch(self, test_loader: DataLoader) -> tuple[float, float]:
        pass

    @torch.no_grad()
    def _validate_one_epoch(self, valid_loader: DataLoader) -> tuple[float, float, float]:
        pass

    def train(
            self,
            num_epochs: int,
            batch_size:int,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            save_path: str
    ) -> None:

        # TODO: break down this function using train / validate helpers

        self._build_scheduler(len(train_loader))

        min_loss = np.inf
        self.model.train()
        for epoch in range(1, num_epochs + 1):

            start_time = time.time()

            # training
            tqdm_test_loader = tqdm(
                train_loader,
                desc=f"Epoch {epoch} / {self.cfg.training_epochs} [train]",
                leave=False
            )

            epoch_train_loss = 0
            mae_train_loss = 0
            for image, target, _, _ in tqdm_test_loader:

                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()  # reset gradient for every batch

                # mixed precision training
                with autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                    logits = self.model(image)
                    mean_variance_loss, mean_loss, var_loss = self.loss_function.compute_loss(logits, target)

                self.scaler.scale(mean_variance_loss).backward()
                self.scaler.step(self.optimizer)

                epoch_train_loss += mean_variance_loss

                with torch.no_grad():
                    pred_ages = self.loss_function.predict_age(logits)

                mae = compute_mae(pred_ages, target)
                mae_train_loss += mae

            mae_train_loss /= len(train_loader)
            epoch_train_loss /= len(train_loader)

            self.scheduler.step()

            # validating
            epoch_valid_loss = 0
            mae_valid_loss = 0
            tqdm_valid_loader = tqdm(
                valid_loader,
                desc=f"Epoch {epoch} / {self.cfg.training_epochs} [validation]",
                leave=False
            )

            all_preds: list[torch.Tensor] = []
            all_targets: list[torch.Tensor] = []
            self.model.eval()
            with torch.no_grad():
                for image, target, _, _ in tqdm_valid_loader:

                    image = image.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    with autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                        logits = self.model(image)
                        epoch_valid_loss, _, _ = self.loss_function.compute_loss(logits, target)

                    pred_ages = self.loss_function.predict_age(logits)
                    mae = compute_mae(pred_ages, target)
                    mae_valid_loss += mae

                    all_preds.append(pred_ages.cpu())
                    all_targets.append(target.cpu().view(-1))

                epoch_valid_loss /= len(valid_loader)
                mae_valid_loss /= len(valid_loader)

            # timer statistics
            end_time = time.time()
            epoch_time = end_time - start_time

            # another metric
            cs5 = compute_cumulative_score(torch.cat(all_preds), torch.cat(all_targets), threshold=5)



            if epoch_train_loss < min_loss:
                min_loss = epoch_train_loss
                torch.save(self.model.state_dict(), save_path)










