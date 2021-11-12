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


N = 100 #number of datapoints
d = 1
x1 = np.random.uniform(0,1,N)
x2 = np.random.uniform(0,1,N)
y = FrankeFunction(x1,x2)
noise = np.random.normal(0, 1, size=(y.shape)) #Find random noise
y_noisy =  FrankeFunction(x1,x2) + noise*0.2
X = createX(x1,x2,d)
X_ols  = createX(x1,x2,5)
X_train, X_test, y_train, y_test = train_test_split(X,y_noisy,test_size=0.2) #Split the data into training and test sets
X_train_ols, X_test_ols, y_train_ols, y_test_ols = train_test_split(X,y_noisy, test_size=0.2)

betaOLS = ols(X_train_ols, y_train_ols)
y_tildeOLS = predict(X_train, betaOLS)
y_predictOLS = predict(X_test, betaOLS)

eta = np.logspace(-5,-3,3)
n_neurons = np.logspace(0,2,3)
n_neurons = np.array([1,5,7,10,12,15,20])
lmb = 0.001
mse_train = np.zeros((len(eta), len(n_neurons)))
mse_test = np.zeros((len(eta), len(n_neurons)))

actfunc = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "relu": relu,
    "leaky_relu": leaky_relu
}
af = "leaky_relu"

for i,eta_ in enumerate(eta):
    for j,n_  in enumerate(n_neurons):
        NN = NeuralNetwork(X_train, y_train, epochs = 2000, batch_size = 80,
            n_categories = 1, eta = eta_, lmbd = lmb, n_hidden_neurons = [n_], activation_function = actfunc[af])
        NN.train()
        y_tilde = NN.predict_reg(X_train)
        y_predict = NN.predict_reg(X_test)

        mse_train_ = mse(y_train.reshape(y_tilde.shape), y_tilde)
        mse_test_ = mse(y_test.reshape(y_predict.shape), y_predict)

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
