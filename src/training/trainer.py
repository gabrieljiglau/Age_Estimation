import os
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

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def _train_one_epoch(self, train_loader: DataLoader, epoch: int) -> tuple[float, float]:

        """
        :return: train_loss, mae_train_loss, global_step
        """

        dataset_size = len(train_loader.dataset)

        tqdm_train_loader = tqdm(
            train_loader,
            desc=f"Epoch {epoch} / {self.cfg.training_epochs} [train]",
            leave=False
        )

        train_loss = 0
        mae = 0
        self.model.train()
        for image, target, _, _ in tqdm_train_loader:
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()  # reset gradient for every batch

            # mixed precision training
            with autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(image)
                loss, mean_loss, var_loss = self.loss_function.compute_loss(logits, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            train_loss += loss.item()

            with torch.no_grad():
                pred_ages = self.loss_function.predict_age(logits)

            mae += compute_mae(pred_ages, target)

            tqdm_train_loader.set_postfix(train_loss=f"{train_loss / dataset_size:.3f}",
                                          mae=f"{mae / dataset_size:.3f}")

        return train_loss, mae

    @torch.no_grad()
    def _validate_one_epoch(self, valid_loader: DataLoader, epoch:int) -> tuple[float, float, float]:

        """
        :return: valid_loss, mae_valid_loss, cs5
        """

        mae_valid_loss = 0
        tqdm_valid_loader = tqdm(
            valid_loader,
            desc=f"Epoch {epoch} / {self.cfg.training_epochs} [validation]",
            leave=False
        )

        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        self.model.eval()

        for image, target, _, _ in tqdm_valid_loader:
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(image)
                valid_loss, _, _ = self.loss_function.compute_loss(logits, target)

            pred_ages, _ = self.loss_function.predict_age(logits)
            mae_valid_loss = compute_mae(pred_ages, target)

            all_preds.append(pred_ages.cpu())
            all_targets.append(target.cpu().view(-1))

        # another metric
        cs5 = compute_cumulative_score(torch.cat(all_preds), torch.cat(all_targets), threshold=5)

        return valid_loss, mae_valid_loss, cs5


    def train(
            self,
            train_loader: DataLoader,
            valid_loader: DataLoader,
    ) -> None:

        """
        training function with model saving

        :param train_loader: training DataLoader
        :param valid_loader: valid DataLoader
        :return:
        """

        self._build_scheduler(len(train_loader))

        min_loss = np.inf

        print(f"\n{'='*43}")
        print(f"Model  : {self.cfg.model_name} ({self.model.num_parameters} million params)")
        print(f"Loss   : {self.cfg.loss_type}")
        print(f"Device : {self.device} | AMP : {self.use_amp}")
        print(f"Epochs : {self.cfg.training.epochs} | Batch : {self.cfg.training.batch_size}")
        print(f"{'='*43}")

        for epoch in range(1, self.cfg.training_epochs + 1):

            start_time = time.time()

            # training
            epoch_train_loss, mae_train_loss = self._train_one_epoch(train_loader, epoch)
            epoch_train_loss /= len(train_loader)
            mae_train_loss /= len(train_loader)

            self.scheduler.step()

            # validating
            epoch_valid_loss, mae_valid_loss, cs5 = self._validate_one_epoch(valid_loader, epoch)
            epoch_valid_loss /= len(valid_loader)
            mae_valid_loss /= len(valid_loader)

            # timer statistics
            end_time = time.time()
            epoch_time = end_time - start_time

            # logging
            self._log_scalar("train/epoch_loss", epoch_train_loss, epoch)
            self._log_scalar("train/epoch_mae", mae_train_loss, epoch)
            self._log_scalar("valid/epoch_loss", epoch_valid_loss, epoch)
            self._log_scalar("valid/epoch_mae", mae_valid_loss, epoch)
            self._log_scalar("valid/cs5", cs5, epoch)

            print(f"Epoch {epoch} / {self.cfg.training_epochs}"
                  f" train_loss = {epoch_train_loss:.4f} train_mae = {mae_train_loss:.4f}"
                  f" valid_loss = {epoch_valid_loss:.4f} valid_mae = {mae_valid_loss:.4f}"
                  f" cs5 = {cs5:.4f}"
                  f" epoch_time = {epoch_time}")

            if mae_train_loss < min_loss:
                min_loss = mae_train_loss
                self._save_model(epoch, mae_train_loss)

            if self.writer:
                self.writer.close()

    def _save_model(self, epoch: int, mae: float):
        """save the best weights; remove the previous weights"""

        if self.best_checkpoint and self.best_checkpoint.exists():
            os.remove(self.best_checkpoint)

        file_name = self.save_dir / f"best_{self.cfg.model.name}_epoch{epoch}_mae{mae:.2f}.pt"
        torch.save(
            {"epoch": epoch,
             "mae": mae,
             "model_state_dict": self.model.state_dict()
            },
            file_name
        )

        self.best_checkpoint = file_name
        print(f"Saved {file_name}, mae_loss = {mae:.2f}")

