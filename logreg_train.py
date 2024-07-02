import numpy as np
import argparse
from srcs.utils import load
from srcs.standardScaler import StandardScaler
from srcs.logisticRegression import LogisticRegression
from sklearn.metrics import accuracy_score


def train_test_split(X, y, test_size=0.3, random_state=None):
    # Set the random seed for reproducibility if random_state is given
    if random_state is not None:
        np.random.seed(random_state)
    
    # Determine the number of test samples
    num_samples = len(X)
    num_test_samples = int(num_samples * test_size)
    
    # Generate a random permutation of indices
    indices = np.random.permutation(num_samples)
    
    # Split the indices into test and train indices
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    
    # Use the indices to create the train and test sets
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description="Ceci est un exemple de programme utilisant argparse.")

  parser.add_argument("dataset", type=str, help="Dataset")
  parser.add_argument("-m", "--method", type=str, help="Gradient descent Method")
  parser.add_argument("-v", "--visualize", action="store_true", help="Show évolution graph")
  args = parser.parse_args()
    
  df = load(args.dataset)
  df = df.dropna(subset=['Defense Against the Dark Arts'])
  df = df.dropna(subset=['Charms'])
  df = df.dropna(subset=['Herbology'])
  df = df.dropna(subset=['Divination'])
  df = df.dropna(subset=['Muggle Studies'])
  df = df.dropna(subset=['Astronomy'])
  df = df.dropna(subset=['Ancient Runes'])
  df = df.dropna(subset=['History of Magic'])
  if (args.method == "batch" or args.method == None):
    model = LogisticRegression(learning_rate=0.01, max_iterations=25, regularization_param=0, method="batch")
  elif (args.method == "stochastic"):
    model = LogisticRegression(learning_rate=0.01, max_iterations=25, regularization_param=0.5, method="stochastic")
  elif (args.method == "mini_batch"):
    model = LogisticRegression(learning_rate=0.01, max_iterations=500, regularization_param=0.5, method="mini_batch", batch_size=32)
  else:
      print(f"Error: Unknow method \"{args.method}\"")
      exit()

  X = np.array(df.values[:, [7, 8, 9, 10, 11, 12, 13, 17]], dtype=float)
  y = df.values[:, 1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

  sc = StandardScaler()
  sc.fit(X_train)

  X_train_std = sc.transform(X_train)
  X_test_std = sc.transform(X_test)

  model.fit(X_train_std, y_train)
  y_pred = model.predict(X_test_std)

  print(f'Misclasified samples: {sum(y_test != y_pred)}/{len(y_test)}')
  print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

  if (args.visualize):
      model.graph()

  model.save_model(sc)
