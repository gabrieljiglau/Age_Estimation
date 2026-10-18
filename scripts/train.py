import torch.cuda
import argparse
from src.training import Trainer, MeanVarianceLoss
from src.estimator import build_model
from src.data import build_dataloaders, get_eval_transforms, get_train_transforms

"""
script that trains the AgeEstimator

Usage: 
--------------------------
    python scripts/train.py
            --cfg
            --train_loader
            --valid_loader
            --train_batch_size
            --eval_batch_size
            --test_train_split
            --img_size
            --mean
            --std 
--------------------------

    amendments: mean and std are lists of size 3, a mean / std for each channel (R, G, B)
                all parameters already have a default value
"""

def train(args: argparse.Namespace) -> None:

    """ a wrapper function that initializes and trains the estimator """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(cfg=args.cfg)

    loss_function = MeanVarianceLoss(
        lambda_mean=args.cfg.lambda_mean,
        lambda_variance=args.cfg.lambda_variance,
        start_age=args.cfg.start_age,
        end_age=args.cfg.end_age
    )

    trainer = Trainer(
        model=model,
        loss_function=loss_function,
        device=device,
        cfg=args.cfg
    )

    train_transform = get_train_transforms(img_size=args.img_size, mean=args.mean, std=args.std)
    eval_transform = get_eval_transforms(img_size=args.img_size, mean=args.mean, std=args.std)

    train_loader, valid_loader, _, _ = build_dataloaders(
        root_dir=args.root_dir,
        csv_dir=args.csv_dir,
        train_transform=train_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size
    )

    trainer.train(train_loader=train_loader, valid_laoder=valid_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='../configs/cfg.yaml')
    parser.add_argument('--train_loader', type=str, default='../processed/train_dataset.csv')
    parser.add_argument('--valid_loader', type=str, default='../processed/valid_dataset.csv')
    parser.add_argument('--train_batch_size', type=int, default=96)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--force_run', type=bool, default=False)
    parser.add_argument('--img_size', type=int, default=220)
    parser.add_argument('--std', type=float, nargs='+')
    parser.add_argument('--mean', type=float, nargs='+')

    cli_args = parser.parse_args()
    train(cli_args)
