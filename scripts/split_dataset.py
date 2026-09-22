import argparse
import numpy as np
import pandas as pd

"""
this file is intended to be ran as a script;
it splits the original csv dataset into a train dataset (train/test) 
and a validation dataset

Usage:
--------------------------
    python scripts/split_dataset.py --csv_dataset_dir --dataset_train --dataset_validate --dataset_test
--------------------------
"""

def split_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    :param args: should contain -full/path/to/csv_dataset,
            desired/path/to/csv_train, desired/path/to_csv/validate, desired/path/to_csv/test
    :return: a DataFrame containing the image, alongside the age, sex and ethnicity
    """

    num_train = len(args.input_csv)
    split_size = int(np.floor(num_train * 0.2))
    train_size = num_train - split_size

    ## implement the split by yourself !!
    csv_train, csv_test = custom_split(args.input_csv.to_numpy(), [train_size, split_size])



