import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import random
import importlib
import functions
import NeuralNetworkClas
importlib.reload(functions); importlib.reload(NeuralNetworkClas)
from functions import *
from NeuralNetworkClas import *


data = load_breast_cancer()
x = data['data']
y = data['target']

#Select features relevant to classification (texture,perimeter,compactness and symmetery)
#and add to input matrix

temp1=np.reshape(x[:,1],(len(x[:,1]),1))
temp2=np.reshape(x[:,2],(len(x[:,2]),1))
X=np.hstack((temp1,temp2))
temp=np.reshape(x[:,5],(len(x[:,5]),1))
X=np.hstack((X,temp))
temp=np.reshape(x[:,8],(len(x[:,8]),1))
X=np.hstack((X,temp))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) #Split the data into training and test sets
y_trainhot = to_categorical_numpy(y_train)

eta = np.logspace(-3,-1,3)
n_neurons = np.logspace(0,3,4)
lmb = 0.01

train_accuracy = np.zeros((len(eta),len(n_neurons)))
test_accuracy = np.zeros_like(train_accuracy)

for i,eta_ in enumerate(eta):
    for j,n_  in enumerate(n_neurons):
        NN = NeuralNetwork(X_train, y_trainhot, epochs = 200,
            n_categories = 2, eta = eta_, n_hidden_neurons = [n_,n_], activation_function = sigmoid)
        NN.train()
        y_tilde = NN.predict(X_train)
        y_predict = NN.predict(X_test)

        train_score = accuracy_score_numpy(y_tilde, y_train)
        test_score = accuracy_score_numpy(y_predict, y_test)
        train_accuracy[i,j] = train_score
        test_accuracy[i,j] = test_score

        print(f"Eta: {eta_} | # of neurons: {n_}")
        print(f"Training accuracy: {accuracy_score_numpy(y_tilde, y_train)}")
        print(f"Test accuracy: {accuracy_score_numpy(y_predict, y_test)}")
        print("------------------------")

make_heatmap(train_accuracy, n_neurons, eta, fn = "train_sigmoid.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "Accuracy score training set")
make_heatmap(test_accuracy, n_neurons, eta, fn = "test_sigmoid.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "Accuracy score test set")
