import argparse
from pathlib import Path
import torch.cuda
import time
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.estimator import build_model
from src.training import ConformalRegressor, MeanVarianceLoss
from src.data import build_dataloaders, get_train_transforms, get_eval_transforms
from src.utils import compute_mae, compute_cumulative_score

"""
script that performs inference using the trained AgeEstimator

Usage: 
--------------------------
    python scripts/inference.py
            --cfg
            --checkpoint
            --root_dir
            --test_loader
            --calibrate_loader
            --residuals_path
            --eval_batch_size
            --full_eval
            --new_image
            --coverage
            --img_size
            --mean
            --std 
--------------------------

    amendments: mean and std are lists of size 3, a mean / std for each channel (R, G, B)
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg',
        type=str,
        default='../configs/cfg.yaml'
    )

    parser.add_argument(
        '--checkpoint',
        type=str
    )

    parser.add_argument(
        '--root_dir',
        type=str,
        default='/home/gabriel/Documents/PycharmProjects/Age_Estimation'
    )

    parser.add_argument(
        '--csv_dir',
        type=str,
        default='dataset/processed'
    )

    parser.add_argument(
        '--test_loader',
        type=str,
        default='train_dataset.csv'
    )

    parser.add_argument(
        '--calibration_loader',
        type=str,
        default='calibration_dataset.csv'
    )

    parser.add_argument(
        '--residuals_path',
        type=str,
        default='/home/gabriel/Documents/PycharmProjects/Age_Estimation/checkpoints/residuals.pkl'
    )

    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=128
    )

    parser.add_argument(
        '--img_size',
        type=int,
        default=220
    )

    parser.add_argument(
        '--std',
        type=float,
        nargs='+'
    )

    parser.add_argument(
        '--mean',
        type=float,
        nargs='+'
    )

    parser.add_argument(
        '--full_eval',
        type=bool,
        default=False
    )

    parser.add_argument(
        '--new_image',
        type=bool,
        default=False
    )

    parser.add_argument(
        '--coverage',
        type=float,
        default=0.85
    )

    return parser.parse_args()


def evaluate_dataset(
        args: argparse.Namespace,
        eval_loader: DataLoader) -> None:

    """ evaluates the estimator on the whole test dataset;
        prints the following statistics -> evaluation time, total loss, MAE loss, CS5 percentage
    """

    model = build_model(cfg=args.cfg)

    loss_function = MeanVarianceLoss(
        lambda_mean=args.cfg.lambda_mean,
        lambda_variance=args.cfg.lambda_variance,
        start_age=args.cfg.start_age,
        end_age=args.cfg.end_age
    )

    checkpoint = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    dataset_size = len(eval_loader.dataset)
    tqdm_eval_loader = tqdm(eval_loader)

    eval_loss = 0
    mae_loss = 0

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    model.eval()

    start_time = time.time()
    end_time = 0
    with torch.no_grad():

        for image, targets, _, _ in tqdm_eval_loader:
            image = image.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(image)
            pred_ages, _ = loss_function.predict_age(logits)

            loss, _, _ = loss_function.compute_loss(logits, targets)
            eval_loss += loss.item()

            mae_loss += compute_mae(pred_ages, targets)

            all_preds.append(pred_ages.cpu())
            all_targets.append(targets.cpu())

        end_time = time.time()
        # proportion of samples with |pred - true| <= threshold
        cs5 = compute_cumulative_score(torch.cat(all_preds), torch.cat(all_targets))

    print(f"Evaluation time = {(end_time - start_time):.4f}")
    print(f"Total eval loss = {eval_loss/dataset_size:.4f}")
    print(f"MAE loss = {mae_loss/dataset_size:.4f}")
    print(f"CS5 = {cs5:.4f}")


def predict_new_instance(
        args: argparse.Namespace,
        eval_loader: DataLoader | None = None,
        instance_idx: int = 123
) -> tuple[float, float, float]:

    """age prediction for a new image either from the evaluation set, or one completely new

    :return: the predicted age, as well as the adjusted age,
             based on the conformal residuals
    """

    if args.new_image:
        file_extensions = ["jpg", "jpeg", "png"]

        img_source = Path(args.img_source)
        if not img_source.exists():
            raise FileNotFoundError(f"{img_source} does not exist !")

        extension = img_source.suffix.lower()
        if extension not in file_extensions:
            raise ValueError(f"Unsupported extension {extension}. "
                             f"The only supported extensions are {file_extensions}")

        # TODO: after done training and results are satisfactory,
        #       modify the input size to be equal to the network so you can test on completely new images

    else:

        instance_idx %= len(eval_loader.dataset)
        img_source = eval_loader[instance_idx]['img_source']

    tensor = torch.tensor(Image.open(img_source).convert('RGB'))
    tensor.unsqueeze(dim=0)
    tensor = tensor.to(device, non_blocking=True)

    model = build_model(cfg=args.cfg)

    loss_function = MeanVarianceLoss(
        lambda_mean=args.cfg.lambda_mean,
        lambda_variance=args.cfg.lambda_variance,
        start_age=args.cfg.start_age,
        end_age=args.cfg.end_age
    )

    checkpoint = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    conformal_regressor = ConformalRegressor(model=model, loss_function=loss_function)
    dict_info = conformal_regressor.calibrate(args.calibrate_loader, args.residuals_path)

    pred_age, lower_bound, upper_bound = (
        conformal_regressor.predict_age_range(
            instance=tensor,
            residuals=dict_info['residuals'],
            num_samples=dict_info['calibration_set_size'],
            coverage=args.coverage
        )
    )

    print(f"Lower bound = {lower_bound}, predicted age = {pred_age}; upper bound = {upper_bound}")
    return pred_age, lower_bound, upper_bound


if __name__ == '__main__':

    cli_args = _parse_args()

    train_transform = get_train_transforms(img_size=cli_args.img_size, mean=cli_args.mean, std=cli_args.std)
    eval_transform = get_eval_transforms(img_size=cli_args.img_size, mean=cli_args.mean, std=cli_args.std)

    _, _, test_loader, _ = build_dataloaders(
        root_dir=cli_args.root_dir,
        csv_dir=cli_args.csv_dir,
        train_transform=train_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
        eval_batch_size=cli_args.eval_batch_size
    )

    if cli_args.full_eval():
        evaluate_dataset(eval_loader=test_loader)
    else:
        predict_new_instance(args=cli_args, eval_loader=test_loader, instance_idx=cli_args.instance_idx)