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


# Étape 1: Charger les données du fichier CSV
def load_data(filename):
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return data

# Étape 2: Préparer les données
def preprocess_data(data):
    house_to_num = {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}
    X = []
    Y = []
    for row in data:
        if row['Hogwarts House'] in house_to_num:
            features = []
            for key in row:
                if key in ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']:
                    if row[key] != '':
                        features.append(float(row[key]))
                    else:
                        features.append(np.nan)  # Utiliser NaN pour les valeurs manquantes
            X.append(features)
            Y.append(house_to_num[row['Hogwarts House']])
    X = np.array(X)
    Y = np.array(Y).reshape(1, -1)
    
    # Remplacer les NaN par la moyenne de chaque colonne
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    
    return X, Y

# Normaliser les données
def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm

# Fonction sigmoïde numériquement stable
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Clipping des valeurs de z pour éviter overflow
    return 1 / (1 + np.exp(-z))

# Initialiser les coefficients
def initialize_weights(dim):
    w = np.random.randn(dim, 1) * 0.01  # Initialisation aléatoire
    b = 0
    return w, b

# Calculer les prévisions
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
    return grads, cost

# Optimiser les coefficients
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

# Prédire les classes
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction

# Mettre le tout ensemble
# def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.001, print_cost=False):
#     w, b = initialize_weights(X_train.shape[0])
#     parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
#     w = parameters["w"]
#     b = parameters["b"]
#     Y_prediction_test = predict(w, b, X_test)
#     Y_prediction_train = predict(w, b, X_train)
#     print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
#     print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
#     d = {"costs": costs, "Y_prediction_test": Y_prediction_test, "Y_prediction_train": Y_prediction_train, "w": w, "b": b, "learning_rate": learning_rate, "num_iterations": num_iterations}
#     return d

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
# Charger les données
df = load_data('students.csv')
df = pd.read_csv('students.csv')
model = LogisticRegression(eta=0.01, max_iter=50, Lambda=10)

X = np.array(df.values[:, [7, 8, 9, 10, 11, 12, 13, 14, 17]], dtype=float)
y = df.values[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

# X_scaled_train, computed_mean_train, computed_std_train = normalization(X_train)
# X_scaled_test, computed_mean_test, computed_std_test = normalization(X_test)
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


model.fit(X_train_std, y_train)
# model.fit(X_scaled_train, y_train)
# print(X)
# model.fit()
y_pred = model.predict(X_test_std)

print(f'Misclasified samples: {sum(y_test != y_pred)}')
# print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
model.save_model(sc)
# Préparer les données
# X, Y = preprocess_data(data)

# Normaliser les données
# X = normalize_data(X).T  # Transposer après normalisation pour obtenir la bonne forme

# Diviser les données en ensembles d'entraînement et de test
# train_size = int(0.8 * X.shape[1])
# X_train, X_test = X[:, :train_size], X[:, train_size:]
# Y_train, Y_test = Y[:, :train_size], Y[:, train_size:]

# # Apprentissage et évaluation du modèle
# d = model(X_train, Y_train, X_test, Y_test, num_iterations=10000, learning_rate=0.001, print_cost=True)
# print(d)

# exemple of weights
# Gryffindor,Hufflepuff,Ravenclaw,Slytherin,Mean,Std
# -0.13270739231324466,-0.07723638137537235,-0.11480145000342634,-0.14596581499686018,,
# -0.06375097344209484,-0.1036449265942309,0.09269168493906925,0.0747634090524936,-0.5575452616653115,5.178953400640137
# -0.07830098063878205,-0.007631018318870466,0.12768452443338985,-0.04165433406856253,-243.75053420642624,8.718582150439774
# -0.08714945606599672,0.0942687360273281,0.06430504787805108,-0.07134086664028969,0.9740944682957365,5.261204499641635
# 0.03680317086805964,0.05324787531197434,0.03884856206092268,-0.1288880596331818,3.113002921129502,4.163231576279715
# -0.03800279852010917,-0.06672416874704978,0.1356459961626889,-0.03084095293688348,-244.49405416561387,482.84578923071285

