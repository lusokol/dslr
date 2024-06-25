from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, max_iterations=50, regularization_param=0, initial_weights=None, classes=None):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization_param = regularization_param
        self.weights = initial_weights
        self.classes = classes
        self.errors = []
        self.costs = []

    def fit(self, X, y, initial_weights=None):
        self.classes = np.unique(y).tolist()
        X_bias = np.insert(X, 0, 1, axis=1)
        num_samples = X_bias.shape[0]

        # initiate every weights at zero
        if initial_weights is None:
            self.weights = np.zeros(X_bias.shape[1] * len(self.classes))
        self.weights = self.weights.reshape(len(self.classes), X_bias.shape[1])

        # one hot incoding for houses
        y_one_hot = np.zeros((len(y), len(self.classes)))
        for i in range(len(y)):
            y_one_hot[i, self.classes.index(y[i])] = 1

        # gradient descent
        self.batch_gradient_descent(X_bias, y_one_hot, num_samples, X, y)
        # self.stochastic_gradient_descent(X_bias, y_one_hot, num_samples, X)

        cost = self.costs
        iterations = range(len(cost))

        # Tracé de la fonction de coût
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(iterations, cost, label='Fonction de coût')
        plt.xlabel('Nombre d\'itérations')
        plt.ylabel('Coût')
        plt.title('Évolution de la fonction de coût')
        plt.legend()

        # Tracé des erreurs de classification
        plt.subplot(1, 2, 2)
        plt.plot(iterations, self.errors, label='Erreurs de classification', color='red')
        plt.xlabel('Nombre d\'itérations')
        plt.ylabel('Nombre d\'erreurs')
        plt.title('Évolution des erreurs de classification')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return self

    def batch_gradient_descent(self, X_bias, y_one_hot, num_samples, X, y):
      for _ in range(self.max_iterations):
            probabilities = self._net_input(X_bias).T
            log_likelihood = y_one_hot.T.dot(np.log(probabilities)) + (1 - y_one_hot).T.dot(np.log(1 - probabilities))
            regularization_term = (self.regularization_param / (2 * num_samples)) * np.sum(self.weights[:, 1:] ** 2)
            cost = (-1 / num_samples) * np.sum(log_likelihood) + regularization_term
            self.costs.append(cost)
            self.errors.append(np.sum(y != self.predict(X)))
            print(self.errors)
            gradient = (1 / num_samples) * (probabilities - y_one_hot).T.dot(X_bias)
            regularization_gradient = (self.regularization_param / num_samples) * np.insert(self.weights[:, 1:], 0, 0, axis=1)
            self.weights -= self.learning_rate * (gradient + regularization_gradient)

    def stochastic_gradient_descent(self, X_bias, y_one_hot, num_samples, X):
        pbar = tqdm(total=self.max_iterations, desc='Iterations')
        for _ in range(self.max_iterations):
            for i in range(num_samples):
                xi = X_bias[i].reshape(1, -1)
                yi = y_one_hot[i].reshape(1, -1)
                prediction = self._net_input(xi).T

                lhs = yi.T.dot(np.log(prediction))
                rhs = (1 - yi).T.dot(np.log(1 - prediction))

                regularization_term = (self.regularization_param / (2 * num_samples)) * np.sum(np.sum(self.weights[:, 1:] ** 2))
                cost = (-1 / num_samples) * np.sum(lhs + rhs) + regularization_term
                self.costs.append(cost)
                self.errors.append(np.sum(np.argmax(y_one_hot, axis=1) != self.predict(X)))

                gradient_regularization_term = (self.regularization_param / num_samples) * self.weights[:, 1:]
                self.weights = self.weights - (self.learning_rate * (1 / num_samples) * (prediction - yi).T.dot(xi) + np.insert(gradient_regularization_term, 0, 0, axis=1))
            pbar.update(1)
        pbar.close()


    def _net_input(self, X):
        return self._sigmoid(np.dot(self.weights, X.T))

    def predict(self, X):
        X_bias = np.insert(X, 0, 1, axis=1)
        probabilities = self._net_input(X_bias).T
        return [self.classes[i] for i in probabilities.argmax(axis=1)]

    def save_model(self, scaler, filepath='../datasets/weights.csv'):
        with open(filepath, 'w+') as file:
            file.write(','.join(map(str, self.classes)) + ',Mean,Std\n')

            for j in range(self.weights.shape[1]):
                file.write(','.join(map(str, self.weights[:, j])) + ',')
                if j > 0:
                    file.write(f'{scaler._mean[j - 1]},{scaler._std[j - 1]}\n')
                else:
                    file.write(',\n')
        return self

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))


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

def normalization(X, mean=None, std=None):
    if mean is None or std is None:
        computed_mean = np.mean(X, axis=0)
        computed_std = np.std(X, axis=0, ddof=0)
    else:
        computed_mean = mean
        computed_std = std
    
    # Standardize the data
    X_scaled = (X - computed_mean) / computed_std

    return X_scaled, computed_mean, computed_std

###################################

def mean_(X):
  total = 0
  for x in X:
    if np.isnan(x):
      continue
    total = total + x
  return total / len(X)

def std_(X):
  mean = mean_(X)
  total = 0
  for x in X:
    if np.isnan(x):
      continue
    total = total + (x - mean) ** 2
  return (total / len(X)) ** 0.5

class StandardScaler(object):
  """Standardize features by removing the mean and scaling to unit variance

  Attributes
  ----------
  _mean: 1d-array, shape [n_features]
    Mean of the training samples or zero
  _std: 1d-array, shape [n_features]
    Standard deviation of the training samples or one
  """
  def __init__(self, mean=np.array([]), std=np.array([])):
    self._mean = mean
    self._std = std

  def fit(self, X):
    """Compute the mean and std to be used for later scaling.

    Parameters
    ----------
    X : array-like, shape [n_samples, n_features]
    """
    for i in range(0, X.shape[1]):
      self._mean = np.append(self._mean, mean_(X[:, i]))
      self._std = np.append(self._std, std_(X[:, i]))

  def transform(self, X):
    """Perform standardization by centering and scaling

    Parameters
    ----------
    X : array-like, shape [n_samples, n_features]
    """
    return ((X - self._mean) / self._std)
###################################
if __name__ == "__main__":
  df = pd.read_csv('students.csv')
  df = df.dropna(subset=['Defense Against the Dark Arts'])
  df = df.dropna(subset=['Charms'])
  df = df.dropna(subset=['Herbology'])
  df = df.dropna(subset=['Divination'])
  df = df.dropna(subset=['Muggle Studies'])
  df = df.dropna(subset=['Astronomy'])
  df = df.dropna(subset=['Ancient Runes'])
  df = df.dropna(subset=['History of Magic'])
  model = LogisticRegression(learning_rate=0.01, max_iterations=25, regularization_param=10)

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
  model.save_model(sc)
