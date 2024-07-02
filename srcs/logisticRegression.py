import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.1, max_iterations=50, regularization_param=0, initial_weights=None, classes=None, method="batch", batch_size=10):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization_param = regularization_param
        self.weights = initial_weights
        self.classes = classes
        self.errors = []
        self.costs = []
        self.method = method
        self.batch_size = batch_size

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
        if self.method == "batch":
          self.batch_gradient_descent(X_bias, y_one_hot, num_samples, X, y)
        elif self.method == "stochastic":
          self.stochastic_gradient_descent(X_bias, y_one_hot, num_samples, X, y)
        elif self.method == "mini_batch":
          self.mini_batch_gradient_descent(X_bias, y_one_hot, num_samples, X, y)
        return self

    def batch_gradient_descent(self, X_bias, y_one_hot, num_samples, X, y):
      for _ in range(self.max_iterations):
            probabilities = self._net_input(X_bias).T
            log_likelihood = y_one_hot.T.dot(np.log(probabilities)) + (1 - y_one_hot).T.dot(np.log(1 - probabilities))
            regularization_term = (self.regularization_param / (2 * num_samples)) * np.sum(self.weights[:, 1:] ** 2)
            cost = (-1 / num_samples) * np.sum(log_likelihood) + regularization_term
            self.costs.append(cost)
            self.errors.append(np.sum(y != self.predict(X)))
            gradient = (1 / num_samples) * (probabilities - y_one_hot).T.dot(X_bias)
            regularization_gradient = (self.regularization_param / num_samples) * np.insert(self.weights[:, 1:], 0, 0, axis=1)
            self.weights -= self.learning_rate * (gradient + regularization_gradient)

    def stochastic_gradient_descent(self, X_bias, y_one_hot, num_samples, X, y):
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
                self.errors.append(np.sum(y != self.predict(X)))

                gradient_regularization_term = (self.regularization_param / num_samples) * self.weights[:, 1:]
                self.weights = self.weights - (self.learning_rate * (1 / num_samples) * (prediction - yi).T.dot(xi) + np.insert(gradient_regularization_term, 0, 0, axis=1))
            pbar.update(1)
        pbar.close()

    def mini_batch_gradient_descent(self, X_bias, y_one_hot, num_samples, X, y):
        for _ in range(self.max_iterations):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                X_batch = X_bias[batch_indices]
                y_batch = y_one_hot[batch_indices]

                predictions = self._net_input(X_batch).T

                lhs = y_batch.T.dot(np.log(predictions))
                rhs = (1 - y_batch).T.dot(np.log(1 - predictions))

                regularization_term = (self.regularization_param / (2 * num_samples)) * np.sum(np.sum(self.weights[:, 1:] ** 2))
                cost = (-1 / num_samples) * np.sum(lhs + rhs) + regularization_term

                gradient_regularization_term = (self.regularization_param / num_samples) * self.weights[:, 1:]
                self.weights = self.weights - (self.learning_rate * (1 / num_samples) * (predictions - y_batch).T.dot(X_batch) + np.insert(gradient_regularization_term, 0, 0, axis=1))
            self.costs.append(cost)
            self.errors.append(np.sum(y != self.predict(X)))

    def _net_input(self, X):
        return self._sigmoid(np.dot(self.weights, X.T))

    def predict(self, X):
        X_bias = np.insert(X, 0, 1, axis=1)
        probabilities = self._net_input(X_bias).T
        return [self.classes[i] for i in probabilities.argmax(axis=1)]

    def save_model(self, scaler, filepath='./datasets/weights.csv'):
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
    
    def graph(self):
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