import csv
import numpy as np

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
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.001, print_cost=False):
    w, b = initialize_weights(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs, "Y_prediction_test": Y_prediction_test, "Y_prediction_train": Y_prediction_train, "w": w, "b": b, "learning_rate": learning_rate, "num_iterations": num_iterations}
    return d

# Charger les données
data = load_data('students.csv')

# Préparer les données
X, Y = preprocess_data(data)

# Normaliser les données
X = normalize_data(X).T  # Transposer après normalisation pour obtenir la bonne forme

# Diviser les données en ensembles d'entraînement et de test
train_size = int(0.8 * X.shape[1])
X_train, X_test = X[:, :train_size], X[:, train_size:]
Y_train, Y_test = Y[:, :train_size], Y[:, train_size:]

# Apprentissage et évaluation du modèle
d = model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.001, print_cost=True)
print(d)

# exemple of weights
# Gryffindor,Hufflepuff,Ravenclaw,Slytherin,Mean,Std
# -0.13270739231324466,-0.07723638137537235,-0.11480145000342634,-0.14596581499686018,,
# -0.06375097344209484,-0.1036449265942309,0.09269168493906925,0.0747634090524936,-0.5575452616653115,5.178953400640137
# -0.07830098063878205,-0.007631018318870466,0.12768452443338985,-0.04165433406856253,-243.75053420642624,8.718582150439774
# -0.08714945606599672,0.0942687360273281,0.06430504787805108,-0.07134086664028969,0.9740944682957365,5.261204499641635
# 0.03680317086805964,0.05324787531197434,0.03884856206092268,-0.1288880596331818,3.113002921129502,4.163231576279715
# -0.03800279852010917,-0.06672416874704978,0.1356459961626889,-0.03084095293688348,-244.49405416561387,482.84578923071285

