import argparse
import pandas as pd
import numpy as np
from srcs.logisticRegression import LogisticRegression
from srcs.standardScaler import StandardScaler

def main():
    # argparse init
    parser = argparse.ArgumentParser(description="Predict Hogwarts House using Logistic Regression")
    parser.add_argument("input_dataset", type=str, help="Path to the input dataset CSV file")
    parser.add_argument("input_weights", type=str, help="Path to the weights CSV file")
    args = parser.parse_args()

    try:
        # reading dataset
        dataset_df = pd.read_csv(args.input_dataset)
        dataset_df = dataset_df.ffill()
        features = np.array(dataset_df.iloc[:, [7, 8, 9, 10, 11, 12, 13, 17]], dtype=float)

        # save weights
        weights_df = pd.read_csv(args.input_weights)
        classes = list(weights_df.columns)[:4]
        mean_values = weights_df.iloc[1:, 4].values
        std_values = weights_df.iloc[1:, 5].values
        weights = weights_df.iloc[:, :4].values.T

        # standard scaler
        scaler = StandardScaler(mean=mean_values, std=std_values)
        standardized_features = scaler.transform(features)
    
    except ValueError as e:
        print(f"ValueError: {e}")
        exit()
    except IndexError as e:
        print(f"IndexError: {e}")
        exit()

    # initiate logistic regression with our weights
    log_reg = LogisticRegression(initial_weights=weights, classes=classes)
    predictions = log_reg.predict(standardized_features)

    # writing result into file
    output_file = "datasets/houses.csv"
    with open(output_file, 'w') as f:
        f.write('Index,Hogwarts House\n')
        for index, house in enumerate(predictions):
            f.write(f'{index},{house}\n')

if __name__ == "__main__":
    main()
