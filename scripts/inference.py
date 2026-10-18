from mailbox import FormatError
from pathlib import Path
import torch.cuda
from argparse import ArgumentParser

from PIL import Image
from torch.utils.data import DataLoader
from src.estimator import AgeEstimator
from src.training import ConformalRegressor, MeanVarianceLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# it should be run only once, and if the residuals already exist, return them without computing anything
def apply_conformal_prediction(calibration_set: DataLoader) -> list[float] | Path:
    pass

def predict_dataset(eval_set: DataLoader) -> None:

    # TODO: compute the loss on the test dataset
    pass

# here the arguments should be args (parsed from the ArgumentParser)
def predict_instance(img_source: str, saved_dict: Path) -> tuple[float, float, float]:

    file_extensions = ["jpg", "jpeg", "png"]

    img_source = Path(img_source)
    if not img_source.exists():
        raise FileNotFoundError(f"{img_source} does not exist !")

    extension = img_source.suffix.lower()
    if extension not in file_extensions:
        raise ValueError(f"Unsupported extension {extension}. "
                         f"The only supported extensions are {file_extensions}")

    model = AgeEstimator()
    loss_function = MeanVarianceLoss()

    checkpoint = torch.load(saved_dict, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    conformal_regressor = ConformalRegressor(model=model, loss_function=loss_function)
    residuals = conformal_regressor.calibrate(args.calibration_set)

    tensor = torch.tensor(Image.open(img_source).convert('RGB'))
    tensor.unsqueeze(dim=0)

    pred_age, lower_bound, upper_bound = (
        conformal_regressor.predict_age_range(
            instance=tensor,
            residuals=residuals,
            num_samples=len(args.calibration_set),
            coverage=args.coverage
        )
    )

    print(f"Lower bound = {lower_bound}, predicted age = {pred_age}; upper bound = {upper_bound}")
    return pred_age, lower_bound, upper_bound





if __name__ == '__main__':
    parser = ArgumentParser()