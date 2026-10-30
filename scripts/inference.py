import argparse
from pathlib import Path
import torch.cuda
import time
from PIL import Image
from tqdm import tqdm

from src.estimator import AgeEstimator
from src.training import ConformalRegressor, MeanVarianceLoss
from src.data import build_dataloaders, get_train_transforms, get_eval_transforms
from src.utils import compute_mae, compute_cumulative_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_dataset(args: argparse.Namespace) -> None:

    train_transform = get_train_transforms(img_size=args.img_size, mean=args.mean, std=args.std)
    eval_transform = get_eval_transforms(img_size=args.img_size, mean=args.mean, std=args.std)

    _, _, eval_loader, _ = build_dataloaders(
        root_dir=args.root_dir,
        csv_dir=args.csv_dir,
        train_transform=train_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
        eval_batch_size=args.eval_batch_size
    )

    dataset_size = len(eval_loader.dataset)
    tqdm_eval_loader = tqdm(eval_loader)

    eval_loss = 0
    mae_loss = 0

    model = AgeEstimator()
    loss_function = MeanVarianceLoss()

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


def predict_instance(args: argparse.Namespace) -> tuple[float, float, float]:

    """age prediction for an image (.jpg/.jpeg/.png file extension) outside the train set

    :return: the predicted age, as well as the adjusted age,
             based on the conformal residuals
    """

    file_extensions = ["jpg", "jpeg", "png"]

    img_source = Path(args.img_source)
    if not img_source.exists():
        raise FileNotFoundError(f"{img_source} does not exist !")

    extension = img_source.suffix.lower()
    if extension not in file_extensions:
        raise ValueError(f"Unsupported extension {extension}. "
                         f"The only supported extensions are {file_extensions}")

    model = AgeEstimator()
    loss_function = MeanVarianceLoss()

    checkpoint = torch.load(args.saved_dict, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    conformal_regressor = ConformalRegressor(model=model, loss_function=loss_function)
    dict_info = conformal_regressor.calibrate(args.calibration_set, args.residuals_path)

    # TODO: modify s.t. the input is a instance from the dataset
    tensor = torch.tensor(Image.open(img_source).convert('RGB'))
    tensor.unsqueeze(dim=0)
    tensor = tensor.to(device, non_blocking=True)

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

    # TODO: add argparse arguments
    #       test the modules
    #       train the network
    #       add compatibility to new images
    #       add support for real-time estimation
    #       wrap it inside a REST API
    #       containerize it with Docker
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='/home/gabriel/Documents/PycharmProjects/Age_Estimation')
    parser.add_argument('--csv_dir', type=str, default='dataset/processed')