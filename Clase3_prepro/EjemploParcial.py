import numpy as np
import pandas as pd

semilla=int(input())

np.random.seed(semilla)

iteraciones=int(input())

# Cargar el dataset de California Housing

df=pd.read_csv('california.csv')

X=df.values[:,1:8]
y=df.values[:,9]

# Normalizar los datos
X_mean =   #Completar
X_std =    #Completar    
X_normalized = (X - X_mean) / X_std

# Agregar un término de sesgo (bias) al conjunto de características
X_bias = np.c_[np.ones((X.shape[0], 1)), X_normalized]

# Dividir los datos en conjuntos de entrenamiento y prueba manualmente
def manual_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test =        #Completar
# Define stochastic gradient descent with minibatches

def minibatch_stochastic_gradient_descent(X, y, learning_rate=0.0001, n_iterations=100, batch_size=32):
    np.random.seed(semilla)
    theta = np.random.randn(X.shape[1])  # Initialize parameters randomly
    m = len(X)  # Number of training examples
    
    for iteration in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradients = 2 / len(xi) * xi.T.dot(xi.dot(theta) - yi)  # Compute gradients
            theta =#Completar
    return theta

# Train the model using minibatch stochastic gradient descent
theta = #Completar

# Make predictions on the test set
y_pred_normalized = X_test.dot(theta)

# Unnormalize the predictions
y_pred = y_pred_normalized * y.std() + y.mean()

# Calcular métricas de evaluación manualmente
def manual_mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def manual_r2_score(y_true, y_pred):
    numerador = #Completar
    denominador = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (numerador / denominator)

# Calcular métricas de evaluación
mse = manual_mean_squared_error(y_test, y_pred)
r2 = #Completar

# Imprimir métricas de evaluación
print("Mean Squared Error (MSE): {:.2f}".format(mse))
print("Coeficiente de Determinación (R²): {:.2f}".format(r2))
print("Coeficients: ")
print(theta)