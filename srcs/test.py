from load_csv import load
import numpy as np
import pandas as pd
import math

def mean(data):
    """
    Return mean of the data
    """
    return sum(data) / len(data)

def variance(args):
    """
    Return variance of the data
    """
    m = mean(args)
    return sum((x - m) ** 2 for x in args) / len(args)


def std(args):
    """
    Return standar deviation of the data
    """
    return variance(args) ** 0.5

def displayStats(stats):
    df = pd.DataFrame(stats, columns=[' ', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    df_transposed = df.set_index(' ').T
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.options.display.colheader_justify = 'center'

    def truncate_col_name(col_name, max_length=15):
        if len(col_name) > max_length:
            return col_name[:6] + '...' + col_name[-6:]
        return col_name

    df_transposed.columns = [truncate_col_name(col) for col in df_transposed.columns]
    print(df_transposed)

def main():
    stats = []
    dataset = load("../datasets/dataset_train.csv")
    for col in dataset.columns[6:]: # we only want usefull data
        dataset[col].values[np.isnan(dataset[col].values)] = 0 # replace every "nan" to zero
        lenght = len(dataset[col].values)
        line = []
        line.append(col)
        line.append(lenght)
        line.append(mean(dataset[col].values))
        line.append(std(dataset[col].values))
        line.append(dataset[col].values[0])
        line.append(dataset[col].values[math.floor(lenght * 0.25)])
        line.append(dataset[col].values[math.floor(lenght * 0.5)])
        line.append(dataset[col].values[math.floor(lenght * 0.75)])
        line.append(dataset[col].values[lenght - 1])
        # q1 = args[math.floor(len(args) / 4)]
        stats.append(line)

    displayStats(stats)


if __name__ == "__main__":
    main()