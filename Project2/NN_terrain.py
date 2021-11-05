import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import importlib
import functions
import NeuralNetworkReg
importlib.reload(functions); importlib.reload(NeuralNetworkReg)
from functions import *
from NeuralNetworkReg import *
from imageio import imread

np.random.seed(2405) # Set a random seed

#Terrain dataset
terrain1 = imread('SRTM_data_Norway_1.tif')
terrain1 = terrain1[:200, :200]
x, y = np.meshgrid(range(terrain1.shape[1]), range(terrain1.shape[0]))
max_y = np.max(y)
x = x / max_y
y = y / max_y
z_terrain = terrain1.flatten().astype(np.float)

X = np.column_stack((x.flatten(), y.flatten()))
X_train, X_test, z_train, z_test = train_test_split(X,z_terrain,test_size=0.2) #Split the data into training and test sets

eta = np.logspace(-4,-3,2)
n_neurons = np.logspace(0,2,3)
n_neurons = np.array([20,50,100])
lmb = 0.01
mse_train = np.zeros((len(eta), len(n_neurons)))
mse_test = np.zeros((len(eta), len(n_neurons)))

actfunc = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "relu": relu,
    "leaky_relu": leaky_relu
}
af = "softmax"

for i,eta_ in enumerate(eta):
    for j,n_  in enumerate(n_neurons):
        NN = NeuralNetwork(X_train, z_train, epochs = 10, batch_size = 25,
            n_categories = 1, eta = eta_, lmbd = 0, n_hidden_neurons = [n_,n_], activation_function = actfunc[af])
        NN.train()
        z_tilde = NN.predict_reg(X_train)
        z_predict = NN.predict_reg(X_test)

        mse_train_ = mse(z_train.reshape(z_tilde.shape), z_tilde)
        mse_test_ = mse(z_test.reshape(z_predict.shape), z_predict)

        mse_train[i,j] = mse_train_
        mse_test[i,j] = mse_test_

        print(f"Eta: {eta_} | # of neurons: {n_}")
        print(f"Training MSE: {mse_train_}")
        print(f"Test MSE: {mse_test_}")
        print("------------------------")

make_heatmap(mse_train, n_neurons, eta, fn = f"train_{af}.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "MSE training set")
make_heatmap(mse_test, n_neurons, eta, fn = f"test_{af}.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "MSE test set")



print(f"MSE Train OLS: {mse(y_train, y_tildeOLS)}")
print(f"MSE Test OLS: {mse(y_test, y_predictOLS)}")
