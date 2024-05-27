import numpy as np
import pandas as pd
from utils import load, splitHouses

# # Exemple de dataframe
# data = {
#     'Hogwarts House': ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin'],
#     'Arithmancy': [4.0, 2.5, 3.0, 4.5],
#     'Astronomy': [3.0, 3.5, 2.0, 5.0],
#     'Herbology': [5.0, 3.0, 4.0, 3.5],
#     'Defense Against the Dark Arts': [4.0, 4.5, 3.0, 2.0],
#     'Divination': [2.5, 3.5, 2.0, 4.0],
#     'Muggle Studies': [3.0, 2.0, 3.5, 4.5],
#     'Ancient Runes': [4.5, 4.0, 2.5, 3.0],
#     'History of Magic': [3.0, 3.5, 4.0, 4.5],
#     'Transfiguration': [4.0, 2.5, 3.5, 3.0],
#     'Potions': [3.5, 4.0, 4.5, 2.0],
#     'Care of Magical Creatures': [4.0, 3.0, 3.5, 4.5],
#     'Charms': [3.0, 4.5, 4.0, 3.5],
#     'Flying': [4.5, 3.0, 2.5, 4.0]
# }

# df = pd.DataFrame(data)

# # Préparer les caractéristiques et les labels
# X = df.iloc[:, 1:].values
# y = df['Hogwarts House'].values

# # Encoder les labels (par exemple, Gryffindor=0, Ravenclaw=1, Hufflepuff=2, Slytherin=3)
# labels = {label: idx for idx, label in enumerate(np.unique(y))}
# y = np.array([labels[label] for label in y])

# # Ajouter une colonne de 1s pour le terme d'interception
# X = np.insert(X, 0, 1, axis=1)

# # Initialisation des paramètres
# theta = np.zeros(X.shape[1])
# learning_rate = 0.01
# iterations = 1000

# Fonction sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Fonction de coût (log-loss)
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5  # Pour éviter log(0)
    cost = (1 / m) * (-y.T @ np.log(h + epsilon) - (1 - y).T @ np.log(1 - h + epsilon))
    return cost

# Descente de gradient
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * (X.T @ (h - y))
        theta = theta - learning_rate * gradient
        cost_history[i] = cost_function(X, y, theta)

    return theta, cost_history


dataset = load("../datasets/dataset_train.csv")
data = splitHouses(dataset.values)

learning_rate = 0.01
iterations = 1000




# # Exécuter la descente de gradient
# theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

# # Prédiction
# def predict(X, theta):
#     return sigmoid(X @ theta)

# # Prédire sur l'ensemble de données
# predictions = predict(X, theta)

# # Mapper les prédictions numériques aux labels de maison
# predicted_labels = [list(labels.keys())[int(pred)] for pred in predictions]
# print(X)
# print(theta)
# print("Paramètres optimisés:", theta)
# print("Prédictions:", predictions)
