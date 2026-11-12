import torch.cuda
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import Trainer, MeanVarianceLoss
from src.estimator import build_model
from src.data import (
    build_dataloaders,
    get_eval_transforms,
    get_train_transforms
)

"""
script that trains the AgeEstimator

Usage: 
--------------------------
    python scripts/train.py
            --cfg
            --root_dir
            --train_dataset
            --valid_dataset
            --test_train_split
            --force_run
"""

def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg',
        type=str,
        default='/home/gabriel/Documents/PycharmProjects/Age_Estimation/configs/cfg.yaml'
    )

    parser.add_argument(
        '--force_run',
        type=bool,
        default=False
    )

    return parser.parse_args()

def train(args: argparse.Namespace) -> None:

    """ a wrapper function that initializes and trains the estimator """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = OmegaConf.load(args.cfg)
    model = build_model(cfg=cfg)

    loss_function = MeanVarianceLoss(
        lambda_mean=cfg.loss.lambda_mean,
        lambda_variance=cfg.loss.lambda_variance,
        start_age=cfg.loss.start_age,
        end_age=cfg.loss.end_age
    )

    trainer = Trainer(
        model=model,
        loss_function=loss_function,
        device=device,
        cfg=cfg
    )

    train_transform = get_train_transforms(
        img_size=cfg.data.img_size,
        mean=cfg.data.mean,
        std=cfg.data.std
    )

    eval_transform = get_eval_transforms(
        img_size=cfg.data.img_size,
        mean=cfg.data.mean,
        std=cfg.data.std
    )

    train_loader, valid_loader, _, _ = build_dataloaders(
        root_dir=cfg.data.root_dir,
        csv_dir=cfg.data.csv_dir,
        raw_dir=cfg.data.raw_dir,
        train_transform=train_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
        train_batch_size=cfg.training.batch_size,
        eval_batch_size=cfg.training.eval_batch_size
    )

    trainer.train(train_loader=train_loader, valid_loader=valid_loader)


if __name__ == '__main__':

    cli_args = _parse_args()
    train(cli_args)
