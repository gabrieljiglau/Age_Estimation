import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import _T_co
from torchvision.transforms import Compose
from pathlib import Path
from PIL import Image


class UtkFaceDataset(Dataset):

    def __init__(self,
                 root_path: str | Path,
                 csv_dir: str | Path,
                 transform: Compose | None = None) -> None:

        """
        :param root_path: path/to/UTKFace dataset
        :param csv_dir: path to the csv file that contains the image, the age, sex and ethnicity
        :param transform: torchvision transform that will be added
        """

        self.root_dir = Path(root_path)
        self.dataset = pd.read_csv(csv_dir)
        self.transform = transform

        columns = {'img_source', 'age', 'gender', 'race'}
        missing = columns - set(self.dataset.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, int, int]:

        row = self.dataset.iloc[index]
        image_path = self.root_dir / row["img_source"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        age = torch.tensor([row["age"]], dtype=torch.float32)
        return image, age, row["gender"], row["ethnicity"]


def build_dataloader(
        root_dir: str | Path,
        csv_dir: str | Path,
        train_transform: Compose,
        val_transform: Compose,
        test_transform: Compose,
        train_batch_size: int=96,
        eval_batch_size: int=128,
        num_workers: int=4,
        pin_memory: bool=True
) -> tuple [DataLoader, DataLoader, DataLoader]:

    csv_dir = Path(csv_dir)

    train_dataset = UtkFaceDataset(root_dir, csv_dir / "train_dataset.csv", train_transform)
    valid_dataset = UtkFaceDataset(root_dir, csv_dir / "valid_dataset.csv", val_transform)
    test_dataset = UtkFaceDataset(root_dir, csv_dir / "test_dataset.csv", test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # uniform sizes for batches, because OneCycleLR will be used
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, valid_loader, test_loader

