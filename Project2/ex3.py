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

NN = NeuralNetwork(X_train, y_trainhot,
    n_categories = 2, n_hidden_neurons = [50,20,5], activation_function = softmax)
NN.train()
y_tilde = NN.predict(X_train)
y_predict = NN.predict(X_test)


print(accuracy_score_numpy(y_tilde, y_train))
print(accuracy_score_numpy(y_predict, y_test))
