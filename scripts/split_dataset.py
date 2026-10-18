import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

"""
this file is intended to be ran as a script;
it splits the original csv dataset into a train dataset (train/test) 
and a validation dataset

Usage:
--------------------------
    python scripts/split_dataset.py
            --input_csv
            --dataset_train
            --dataset_validate
            --dataset_test
            --dataset_calibrate
            --test_train_split
            --valid_train_split
            --test_calibrate_split
            --age_threshold
            --force_run
--------------------------
"""

def split_dataset(args: argparse.Namespace) -> None:

    """
    :param args: all CLI arguments
    :return: a DataFrame containing the image, alongside the age, sex and ethnicity
    """

    output_paths = tuple(Path(path) for path in (
        args.dataset_train,
        args.dataset_validate,
        args.dataset_calibrate,
        args.dataset_test
        )
    )

    if all(path.exists() for path in output_paths) and not args.force_run:
        return

    #   splits: i)   train / test (90% - 10%)
    #           ii)  train (train / validate / calibrate) (70% - 15% - 15%) on the original train split

    original_df = pd.read_csv(args.input_csv)

    # drop all rows where the age > 80, since such images are outliers
    indices = original_df[original_df['age'] > args.age_threshold].index
    original_df.drop(indices, inplace=True)

    full_size = len(original_df)
    print(f"full_size = {full_size}")  # full_size = 23618

    # randomize the dataset, since currently the images are sorted by age
    randomized_csv = original_df.sample(frac=1, random_state=13).reset_index(drop=True)

    full_train_set, full_test_set = train_test_split(
        randomized_csv,
        test_size=args.test_train_split,
        stratify=randomized_csv['age'],
        random_state=7
    )

    train_set, valid_calibrate_set = train_test_split(
        full_train_set,
        test_size=args.valid_calibrate_split,
        stratify=full_train_set['age'],
        random_state=7
    )

    valid_set, calibration_set = train_test_split(
        valid_calibrate_set,
        test_size=args.calibrate_split,
        stratify=valid_calibrate_set['age'],
        random_state=7
    )

    # 17720 images; median age = 29.00 mean age = 32.03
    print(f"train    -> {len(train_set)} images; median age = {train_set['age'].median():.2f}"
          f" mean age = {train_set['age'].mean():.2f}")

    # 1564 images; median age = 29.00 mean age = 32.06
    print(f"validate -> {len(valid_set)} images; median age = {valid_set['age'].median():.2f}"
          f" mean age = {valid_set['age'].mean():.2f}")

    # 1564 images; median age = 29.00 mean age = 32.0
    print(f"calibrate -> {len(calibration_set)} images; median age = {calibration_set['age'].median():.2f}"
          f" mean age = {calibration_set['age'].mean():.2f}")

    # 2317 images; median age = 29.00 mean age = 32.01
    print(f"test     -> {len(full_test_set)} images; median age = {full_test_set['age'].median():.2f}"
          f" mean age = {full_test_set['age'].mean():.2f}")

    # save the train/test datasets as separate csv files
    datasets = [train_set, valid_set, calibration_set, full_test_set]
    [datasets[i].to_csv(output_paths[i], index=False) for i in range(len(datasets))]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='dataset/processed/UTKFace.csv')
    parser.add_argument('--dataset_train', type=str, default='dataset/processed/train_dataset.csv')
    parser.add_argument('--dataset_validate', type=str, default='dataset/processed/valid_dataset.csv')
    parser.add_argument('--dataset_test', type=str, default='dataset/processed/test_dataset.csv')
    parser.add_argument('--dataset_calibrate', type=str, default='dataset/processed/calibrate_dataset.csv')
    parser.add_argument('--test_train_split', type=float, default=0.10)
    parser.add_argument('--valid_calibrate_split', type=float, default=0.15)
    parser.add_argument('--calibrate_split', type=float, default=0.5)
    parser.add_argument('--age_threshold', type=int, default=80)
    parser.add_argument('--force_run', type=bool, default=False)

    cl_arguments = parser.parse_args()
    split_dataset(cl_arguments)
