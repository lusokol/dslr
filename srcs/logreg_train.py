import csv
import numpy as np
import pandas as pd

class LogisticRegression(object):
  """Logistic Regression classifier

  Parameters
  ----------
  eta: float, default: 0.1
    Learning rate (between 0.0 and 1.0)
  max_iter: int, default: 50
    Max passes over the training dataset
  Lambda: float, default: 0
    Regularization term

  Attributes
  ----------
  _w: {array-like}, shape = [n_class, n_feature]
    Weights after fitting
  _errors: list
    Number of misclassifications in every epoch
  _cost: list
    Number of cost values
  """
  def __init__(self, eta=0.1, max_iter=50, Lambda=0, initial_weight=None, multi_class=None):
    self.eta = eta
    self.max_iter = max_iter
    self.Lambda = Lambda
    self._w = initial_weight
    self._K = multi_class
    self._errors = []
    self._cost = []

  def fit(self, X, y, sample_weight=None):
    """Fit training data

    Parameters
    ----------
    X: {array-like}, shape = [n_samples, n_features]
      Training vectors
    y: array-like, shape = [n_samples]
      Target values
    sample_weight: 1d-array, default: None
      Initial weights

    Return
    ------
    self: object
    """
    self._K = np.unique(y).tolist()
    newX = np.insert(X, 0, 1, axis=1)
    m = newX.shape[0]

    self._w = sample_weight
    if not self._w:
      self._w = np.zeros(newX.shape[1] * len(self._K))
    self._w = self._w.reshape(len(self._K), newX.shape[1])

    yVec = np.zeros((len(y), len(self._K)))
    for i in range(0, len(y)):
      yVec[i, self._K.index(y[i])] = 1

    for _ in range(0, self.max_iter):
      predictions = self.net_input(newX).T

      lhs = yVec.T.dot(np.log(predictions))
      rhs = (1 - yVec).T.dot(np.log(1 - predictions))

      r1 = (self.Lambda / (2 * m)) * sum(sum(self._w[:, 1:] ** 2))
      cost = (-1 / m) * sum(lhs + rhs) + r1
      self._cost.append(cost)
      self._errors.append(sum(y != self.predict(X)))

      r2 = (self.Lambda / m) * self._w[:, 1:]
      self._w = self._w - (self.eta * (1 / m) * (predictions - yVec).T.dot(newX) + np.insert(r2, 0, 0, axis=1))
    return self

  def net_input(self, X):
    return self.sigmoid(self._w.dot(X.T))

  def predict(self, X):
    X = np.insert(X, 0, 1, axis=1)
    predictions = self.net_input(X).T
    return [self._K[x] for x in predictions.argmax(1)]

  def save_model(self, sc, filename='../datasets/weights.csv'):
    f = open(filename, 'w+')

    for i in range(0, len(self._K)):
      f.write(f'{self._K[i]},')
    f.write('Mean,Std\n')

    for j in range(0, self._w.shape[1]):
      for i in range(0, self._w.shape[0]):
        f.write(f'{self._w[i][j]},')
      f.write(f'{sc._mean[j - 1] if j > 0 else ""},{sc._std[j - 1] if j > 0 else ""}\n')

    f.close()
    return self

  def sigmoid(self, z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g


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
df = pd.read_csv('students.csv')
df = df.dropna(subset=['Defense Against the Dark Arts'])
df = df.dropna(subset=['Charms'])
df = df.dropna(subset=['Herbology'])
df = df.dropna(subset=['Divination'])
df = df.dropna(subset=['Muggle Studies'])
df = df.dropna(subset=['Astronomy'])
df = df.dropna(subset=['Ancient Runes'])
df = df.dropna(subset=['History of Magic'])
model = LogisticRegression(eta=0.01, max_iter=1000, Lambda=10)

X = np.array(df.values[:, [7, 8, 9, 10, 11, 12, 13, 17]], dtype=float)
y = df.values[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)

print(f'Misclasified samples: {sum(y_test != y_pred)}')
model.save_model(sc)
