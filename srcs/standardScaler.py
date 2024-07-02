import numpy as np
from .utils import mean_, std_

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