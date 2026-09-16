import argparse
import re
import pandas as pd
from pathlib import Path

"""
this file is intended to be ran as a script
"""

def preprocess(args: argparse.Namespace) -> pd.DataFrame:

    """
    :param args: should contain - full/path/to/data, full/path/to/csv_dataset
    :return: a DataFrame containing the image, alongside the age, sex and ethnicity
    """

    """
    The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg

        [age] is an integer from 0 to 116, indicating the age
        [gender] is either 0 (male) or 1 (female)
        [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
    """

    if Path(args.out_path.exists()) and not args.force_run:
        return pd.read_csv(args.out_path)

    gender_map = {0: "Male", 1: "Female"}
    race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

    path = Path(args.input_path)

    file_names = []
    ages = []
    genders = []
    races = []
    for file in path.iterdir():
        if file.suffix == 'jpg':
            match = re.fullmatch("r([1-9])+([0-9])?([0-9])?_([0-1])_([0-4])_(.)*", file.stem)

            if match:
                age, gender, race = map(int, match.groups())
                gender = gender_map[gender] if gender_map[gender] else None
                race = race_map[race] if race_map[race] else None

                ages.append(age)
                genders.append(gender)
                races.append(race)
                file_names.append(file.name)
        else:
            raise FileNotFoundError("The input directory doesn't contain images in .jpg format")


    tuples = list(zip(file_names, ages, genders, races))
    df = pd.DataFrame(tuples, columns = ['img_source', 'age', 'gender', 'race'])
    df.to_csv(args.out_path, index=False)

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--force_run', type=bool)
    cl_arguments = parser.parse_args()