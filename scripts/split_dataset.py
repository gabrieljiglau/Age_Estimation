import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

"""
this file is intended to be ran as a script;
it splits the original csv dataset into a train dataset (train/test) 
and a validation dataset

Usage:
--------------------------
    python scripts/split_dataset.py --input_csv --dataset_train --dataset_validate --dataset_test
--------------------------
"""

def split_dataset(args: argparse.Namespace) -> None:

    """
    :param args: should contain -full/path/to/csv_dataset,
            desired/path/to/csv_train, desired/path/to_csv/validate, desired/path/to_csv/test
    :return: a DataFrame containing the image, alongside the age, sex and ethnicity
    """

    output_paths = tuple(Path(path) for path in (
        args.dataset_train,
        args.dataset_validate,
        args.dataset_test
        )
    )

    if all(path.exists() for path in output_paths) and not args.force_run:
        return

    # 2 splits: i) train / test (90% - 10%)
    #           ii) train (train / validate) (80% - 20%) of the remaining 90%

    full_size = len(args.input_csv)

    full_train_size = int(np.floor(full_size * 0.9))
    test_size = full_size - full_train_size

    train_size = int(np.floor(test_size * 0.8))
    valid_size = full_train_size - train_size

    # randomize the dataset, since currently the images are sorted by age
    randomized_csv = args.input_csv.sample(frac=1, random_state=13).reset_index(drop=True)

    full_train_set, test_set = train_test_split(
        randomized_csv,
        [full_train_size, test_size],
        stratify=randomized_csv['age'],
        random_state=7
    )

    train_set, valid_set = train_test_split(
        full_train_set,
        [train_size, valid_size],
        stratify=randomized_csv['age'],
        random_state=7
    )

    print(f"train    -> {len(train_set)} images; mean age = {train_set['age'].median():.2f}")
    print(f"validate -> {len(valid_set)} images; mean age = {valid_set['age'].median():.2f}")
    print(f"test     -> {len(test_set)} images; mean age = {test_set['age'].median():.2f}")

    # save the train/test datasets as separate csv files
    datasets = [train_set, valid_set, test_set]
    [datasets[i].to_csv(output_paths[i]) for i in range(len(datasets))]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--dataset_train', type=str)
    parser.add_argument('--dataset_validate', type=str)
    parser.add_argument('--dataset_test', type=str)
    parser.add_argument('--force_run', type=bool)

    cl_arguments = parser.parse_args()
    split_dataset(cl_arguments)



