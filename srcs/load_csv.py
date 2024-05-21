import pandas as pd
import os


def load(path: str):
    """
Load .csv file with pandas.
Parse error and return dataset.
    """
    try:
        if not os.path.exists(path):
            raise AssertionError("File doesn't exist.")
        if not path.lower().endswith('.csv'):
            raise AssertionError("File isn't .csv")
        data = pd.read_csv(path)
        return data
    except AssertionError as error:
        print(f"AssertionError: {error}")
        exit()