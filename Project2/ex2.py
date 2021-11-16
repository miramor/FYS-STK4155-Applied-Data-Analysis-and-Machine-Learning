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
from sklearn.neural_network import MLPRegressor



N = 500 #number of datapoints
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

eta = np.logspace(-5,-2,4)
n_neurons = np.logspace(0,2,3)
n_neurons = np.array([1,5,7,10,12,15,20,25])
lmb = 0.001
mse_train = np.zeros((len(eta), len(n_neurons)))
mse_test = np.zeros((len(eta), len(n_neurons)))

actfunc = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "relu": relu,
    "leaky_relu": leaky_relu
}
af = "sigmoid"
"""
for i,eta_ in enumerate(eta):
    for j,n_  in enumerate(n_neurons):
        NN = NeuralNetwork(X_train, y_train, epochs = 3000, batch_size = 50,
            n_categories = 1, eta = eta_, lmbd = lmb, n_hidden_neurons = [n_,n_], activation_function = actfunc[af])
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
make_heatmap(mse_train, n_neurons, eta, fn = f"train_{af}_neur_eta.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "MSE training set")
make_heatmap(mse_test, n_neurons, eta, fn = f"test_{af}_neur_eta.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "MSE test set")



lambdas = np.logspace(-4,-1,4)
mse_train = np.zeros((len(eta), len(lambdas)))
mse_test = np.zeros((len(eta), len(lambdas)))
for i,eta_ in enumerate(eta):
    for j,lmb_  in enumerate(lambdas):
        NN = NeuralNetwork(X_train, y_train, epochs = 3000, batch_size = 50,
            n_categories = 1, eta = eta_, lmbd = lmb_, n_hidden_neurons = [10,10], activation_function = actfunc[af])
        NN.train()
        y_tilde = NN.predict_reg(X_train)
        y_predict = NN.predict_reg(X_test)

        mse_train_ = mse(y_train.reshape(y_tilde.shape), y_tilde)
        mse_test_ = mse(y_test.reshape(y_predict.shape), y_predict)

        mse_train[i,j] = mse_train_
        mse_test[i,j] = mse_test_

        print(f"Eta: {eta_} | lambda: {lmb_}")
        print(f"Training MSE: {mse_train_}")
        print(f"Test MSE: {mse_test_}")
        print("------------------------")

make_heatmap(mse_train, lambdas, eta, fn = f"train_{af}_lambda_eta.pdf",
            xlabel = "Regularization parameter $\lambda$", ylabel = "Learning rate $\eta$", title = "MSE training set")
make_heatmap(mse_test, lambdas, eta, fn = f"test_{af}_lambda_eta.pdf",
            xlabel = "Regularization parameter $\lambda$", ylabel = "Learning rate $\eta$", title = "MSE test set")



"""

print(f"MSE Train OLS: {mse(y_train, y_tildeOLS)}")
print(f"MSE Test OLS: {mse(y_test, y_predictOLS)}")

print(kfold_nn_reg(X, y_noisy, 5, 0.01, 0.001, actfunc[af]))


regr = MLPRegressor(solver = "sgd", random_state=1, hidden_layer_sizes = (25, 25), alpha = 0.01, max_iter=5000).fit(X_train, y_train)
y_pred_reg = regr.predict(X_test)
print(mse(y_test, y_pred_reg))
