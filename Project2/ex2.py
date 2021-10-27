import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import importlib
import functions
import NeuralNetwork
importlib.reload(functions); importlib.reload(NeuralNetwork)
from functions import *
from NeuralNetwork import *

np.random.seed(2405) # Set a random seed

N = 100 #number of datapoints
d = 5 #complexity
x1 = np.random.uniform(0,1,N)
x2 = np.random.uniform(0,1,N)
y = FrankeFunction(x1,x2)
X = createX(x1,x2,d)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) #Split the data into training and test sets



n_features= X_train.shape[1] #numer of features
n_hidden_neurons = 10 # number of hidden neurons
n_outputs = 80
#initialise weights and bias
#hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
bias = 0.01
hidden_bias = np.zeros(n_hidden_neurons) + bias

#output layer
output_weights = np.random.randn(n_hidden_neurons, 1)
output_bias = np.zeros(1) + bias


def feed_forward(X): #feed-forward pass
    z_h = np.matmul(X, hidden_weights) + hidden_bias # weighted sum of inputs to the hidden layer
    a_h = sigmoid(z_h) # activation in the hidden layer
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    y_output= sigmoid(z_o)

    return a_h, y_output


def backpropagation(X, y):
    a_h, y_output = feed_forward(X)
    #error in output layer
    error_output = y_output - y.reshape(y_output.shape)
    #error in hidden layer

    error_hidden = error_output @ output_weights.T * a_h * (1 - a_h)

    # gradients for the output layer
    output_weights_gradient = np.matmul(a_h.T, error_output)
    output_bias_gradient = np.sum(error_output, axis=0)

    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(X.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis=0)

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

"""
lmbd = 0.01
eta = 0.01
for i in range(20000):

    dWo, dBo, dWh, dBh = backpropagation(X_train, y_train)

    # regularization term gradients
    dWo += lmbd * output_weights
    dWh += lmbd * hidden_weights

    # update weights and biases
    output_weights -= eta * dWo
    output_bias -= eta * dBo
    hidden_weights -= eta * dWh
    hidden_bias -= eta * dBh

#print(output_weights)
a_0, y_tilde = feed_forward(X_train)
a_o, y_predict = feed_forward(X_test)
"""
betaOLS = ols(X_train, y_train)
y_tildeOLS = predict(X_train, betaOLS)
y_predictOLS = predict(X_test, betaOLS)

#print(f"MSE Test FFNN: {mse(y_test, y_predict[0,:])}")
#print(f"MSE Test OLS: {mse(y_test, y_predictOLS)}")


FFNN = NeuralNetwork(X_train, y_train, activation_function = leaky_relu)
FFNN.train()
y_tilde = FFNN.predict_reg(X_train)
y_predict = FFNN.predict_reg(X_test)
print(f"MSE Train FFNN: {mse(y_train.reshape(y_tilde.shape), y_tilde)}")
print(f"MSE Train OLS: {mse(y_train, y_tildeOLS)}")

print(f"MSE Test FFNN: {mse(y_test, y_predict[0,:])}")
print(f"MSE Test OLS: {mse(y_test, y_predictOLS)}")
