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

np.random.seed(2405)
N = 500 #number of datapoints
x1 = np.random.uniform(0,1,N)
x2 = np.random.uniform(0,1,N)
y = FrankeFunction(x1,x2)
noise = np.random.normal(0, 1, size=(y.shape)) #random noise
y_noisy =  FrankeFunction(x1,x2) + noise*0.2
X = createX(x1,x2,1)
X_train, X_test, y_train, y_test = train_test_split(X,y_noisy,test_size=0.2) #Split the data into training and test sets


#Scales data when True. Check scale_data function in functions.py for more scaling options
scale = True
if scale:
    X_train, X_test = scale_data(X_train, X_test)
    y_train, y_test = scale_data(y_train, y_test)


eta = np.logspace(-5,-2,4) #learning rate
n_neurons = np.array([1,5,10,15,20,25]) #number of neurons on each layer
lmb = 0.001 #regularization parameter
epochs = 1000 #number of epochs
n_L = 1 #number of layers


actfunc = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "relu": relu,
    "leaky_relu": leaky_relu
}
#Activation function to be used in the neural networks
af = "relu"


#2D arrays to store MSE and R2 for varieties of hyperparameters
mse_train = np.zeros((len(eta), len(n_neurons)))
mse_test = np.zeros((len(eta), len(n_neurons)))
r2_train = np.zeros((len(eta), len(n_neurons)))
r2_test = np.zeros((len(eta), len(n_neurons)))

#Varying the learning rate and number of neurons in each hidden layer
for i,eta_ in enumerate(eta):
    for j,n_  in enumerate(n_neurons):
        NN = NeuralNetwork(X_train, y_train, epochs = epochs, batch_size = 50,
            n_categories = 1, eta = eta_, lmbd = lmb, n_hidden_neurons = [n_]*n_L, activation_function = actfunc[af])
        NN.train()
        NN.plot_mse()
        y_tilde = NN.predict_reg(X_train)
        y_predict = NN.predict_reg(X_test)

        mse_train_ = mse(y_train.reshape(y_tilde.shape), y_tilde)
        mse_test_ = mse(y_test.reshape(y_predict.shape), y_predict)
        r2_train_ = r2(y_train.reshape(y_tilde.shape), y_tilde)
        r2_test_ = r2(y_test.reshape(y_predict.shape), y_predict)


        mse_train[i,j] = mse_train_
        mse_test[i,j] = mse_test_
        r2_train[i,j] = r2_train_
        r2_test[i,j] = r2_test_


        
        print(f"Eta: {eta_} | # of neurons: {n_}")
        #print(f"Training MSE: {mse_train_}")
        #print(f"Test MSE: {mse_test_}")
        #print(f"Training R2: {r2_train_}")
        #print(f"Test R2: {r2_test_}")
        print("------------------------")
        

#Produces heatmaps with MSE and R2 for both training and test set.
make_heatmap(mse_train, n_neurons, eta, fn = f"mse_train_{af}_L{n_L}_neur_eta.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "MSE training set")
make_heatmap(mse_test, n_neurons, eta, fn = f"mse_test_{af}_L{n_L}_neur_eta.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "MSE test set")
make_heatmap(r2_train, n_neurons, eta, fn = f"r2_train_{af}_L{n_L}_neur_eta.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "R2 training set")
make_heatmap(r2_test, n_neurons, eta, fn = f"r2_test_{af}_L{n_L}_neur_eta.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "R2 test set")



lambdas = np.logspace(-4,-1,4)
mse_train = np.zeros((len(eta), len(lambdas)))
mse_test = np.zeros((len(eta), len(lambdas)))
r2_train = np.zeros((len(eta), len(lambdas)))
r2_test = np.zeros((len(eta), len(lambdas)))

#Varying learning rate and regularization parameter
for i,eta_ in enumerate(eta):
    for j,lmb_  in enumerate(lambdas):
        NN = NeuralNetwork(X_train, y_train, epochs = epochs, batch_size = 50,
            n_categories = 1, eta = eta_, lmbd = lmb_, n_hidden_neurons = [10], activation_function = actfunc[af])
        NN.train()
        y_tilde = NN.predict_reg(X_train)
        y_predict = NN.predict_reg(X_test)

        mse_train_ = mse(y_train.reshape(y_tilde.shape), y_tilde)
        mse_test_ = mse(y_test.reshape(y_predict.shape), y_predict)
        r2_train_ = r2(y_train.reshape(y_tilde.shape), y_tilde)
        r2_test_ = r2(y_test.reshape(y_predict.shape), y_predict)

        mse_train[i,j] = mse_train_
        mse_test[i,j] = mse_test_
        r2_train[i,j] = r2_train_
        r2_test[i,j] = r2_test_

        """
        print(f"Eta: {eta_} | lambda: {lmb_}")
        print(f"Training MSE: {mse_train_}")
        print(f"Test MSE: {mse_test_}")
        print(f"Training R2: {r2_train_}")
        print(f"Test R2: {r2_test_}")
        print("------------------------")
        """

#Produces heatmaps with MSE and R2 for both training and test set.
make_heatmap(mse_train, lambdas, eta, fn = f"mse_train_{af}_L1_lambda_eta.pdf",
            xlabel = "Regularization parameter $\lambda$", ylabel = "Learning rate $\eta$", title = "MSE training set")
make_heatmap(mse_test, lambdas, eta, fn = f"mse_test_{af}_L1_lambda_eta.pdf",
            xlabel = "Regularization parameter $\lambda$", ylabel = "Learning rate $\eta$", title = "MSE test set")
make_heatmap(r2_train, lambdas, eta, fn = f"r2_train_{af}_L1_lambda_eta.pdf",
            xlabel = "Regularization parameter $\lambda$", ylabel = "Learning rate $\eta$", title = "R2 training set")
make_heatmap(r2_test, lambdas, eta, fn = f"r2_test_{af}_L1_lambda_eta.pdf",
            xlabel = "Regularization parameter $\lambda$", ylabel = "Learning rate $\eta$", title = "R2 test set")

