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

np.random.seed(2405)

#Load the Wisconsin breast cancer data set
data = load_breast_cancer()
x = data['data']
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Split the data into training and test sets
y_trainhot = to_categorical_numpy(y_train)

#Scales data. Can choose between StandardScaler and MinMax scaler
#Can also choose to divide by std or not.
scale = True
if scale:
    X_train, X_test = scale_data(X_train, X_test, scale_type = StandardScaler, with_std=True)



#Arrays used when producing heatmaps
n_neurons = np.array([50,100,150,200,250]) #Number of neurons
n_neurons2= np.array([20,50,70,120,150]) #Number of neurons
eta = np.logspace(-3,-1,3) #learning rate
lambdas = np.logspace(-4,-2,3) #regulularization parameter



lmb = 0.001 #Regularization parameter 
n_hl = 2 #Numer of hidden layers


actfunc = { #dictionary with the function name of the activation functions.
"sigmoid": sigmoid,
"softmax": softmax,
"relu": relu,
"leaky_relu": leaky_relu
}

#Activating function to be used in the neural networks
af = "sigmoid"


#2D arrays for storing prediction accuracies for different hyper parameters.
train_accuracy = np.zeros((len(eta),len(lambdas)))
test_accuracy = np.zeros_like(train_accuracy)

#Varying learning rate and regularization parameter.
for i,eta_ in enumerate(eta):
    for j,lmb_  in enumerate(lambdas):
        NN = NeuralNetwork(X_train, y_trainhot, epochs = 50, batch_size = 10,
            n_categories = 2, eta = eta_, lmbd = lmb_, n_hidden_neurons = [15,15], activation_function = actfunc[af])
        NN.train()
        NN.plot_accuracy() #uncomment to plot train accuracy for each epoch.
        y_tilde = NN.predict(X_train)
        y_predict = NN.predict(X_test)

        train_score = accuracy_score_numpy(y_tilde, y_train)
        test_score = accuracy_score_numpy(y_predict, y_test)
        train_accuracy[i,j] = train_score
        test_accuracy[i,j] = test_score

        print(f"Lambda: {lmb_} | Eta: {eta_}")
        #print(f"Training accuracy: {accuracy_score_numpy(y_tilde, y_train)}")
        #print(f"Test accuracy: {accuracy_score_numpy(y_predict, y_test)}")
        print("------------------------")


#Makes heatmaps
make_heatmap(train_accuracy, eta, lambdas, fn = f"train_{af}_sc{1 if scale else 0}_L{n_hl}_15_lmb_etac.pdf",
            xlabel = "Learning rate $\eta$", ylabel = "Regularization parameter $\lambda$", title = "Accuracy score training set")
make_heatmap(test_accuracy, eta, lambdas, fn = f"test_{af}_sc{1 if scale else 0}_L{n_hl}_15_lmb_etac.pdf",
            xlabel = "Learning rate $\eta$", ylabel = "Regularization parameter $\lambda$", title = "Accuracy score test set")

#Makes a confusion matrix
make_confusion_matrix(y_test, y_predict, fn="cm_heatmap.pdf", title = "Own Neural Network")


#Varying regularization parameter and number of neurons in each layer
train_accuracy = np.zeros((len(lambdas),len(n_neurons)))
test_accuracy = np.zeros_like(train_accuracy)
for i,lmb_ in enumerate(lambdas):
    for j,n_  in enumerate(n_neurons):
        NN = NeuralNetwork(X_train, y_trainhot, epochs = 50, batch_size = 10,
            n_categories = 2, eta = 0.01, lmbd = lmb_, n_hidden_neurons = [n_]*n_hl, activation_function = actfunc[af])
        NN.train()
        #NN.plot_accuracy() #uncomment to plot train accuracy for each epoch.
        y_tilde = NN.predict(X_train)
        y_predict = NN.predict(X_test)

        train_score = accuracy_score_numpy(y_tilde, y_train)
        test_score = accuracy_score_numpy(y_predict, y_test)
        train_accuracy[i,j] = train_score
        test_accuracy[i,j] = test_score

        #print(f"Lambda: {lmb_} | # of neurons: {n_}")
        #print(f"Training accuracy: {accuracy_score_numpy(y_tilde, y_train)}")
        #print(f"Test accuracy: {accuracy_score_numpy(y_predict, y_test)}")
        #print("------------------------")


make_confusion_matrix(y_test, y_predict)

make_heatmap(train_accuracy, n_neurons, lambdas, fn = f"train_{af}_sc{1 if scale else 0}_L{n_hl}_eta001c.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Regularization parameter $\lambda$", title = "Accuracy score training set")
make_heatmap(test_accuracy, n_neurons, lambdas, fn = f"test_{af}_sc{1 if scale else 0}_L{n_hl}_eta001c.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Regularization parameter $\lambda$", title = "Accuracy score test set")




#Varying learning are and number of neurons in each layer
train_accuracy = np.zeros((len(eta),len(n_neurons)))
test_accuracy = np.zeros_like(train_accuracy)
for i,eta_ in enumerate(eta):
    for j,n_  in enumerate(n_neurons):
        NN = NeuralNetwork(X_train, y_trainhot, epochs = 50, batch_size = 10,
            n_categories = 2, eta = eta_, lmbd = lmb, n_hidden_neurons = [n_]*n_hl, activation_function = actfunc[af])
        NN.train()
        #NN.plot_accuracy() #uncomment to plot train accuracy for each epoch.
        y_tilde = NN.predict(X_train)
        y_predict = NN.predict(X_test)

        train_score = accuracy_score_numpy(y_tilde, y_train)
        test_score = accuracy_score_numpy(y_predict, y_test)
        train_accuracy[i,j] = train_score
        test_accuracy[i,j] = test_score

        #print(f"Eta: {eta_} | # of neurons: {n_}")
        #print(f"Training accuracy: {accuracy_score_numpy(y_tilde, y_train)}")
        #print(f"Test accuracy: {accuracy_score_numpy(y_predict, y_test)}")
        #print("------------------------")


make_confusion_matrix(y_test, y_predict)

make_heatmap(train_accuracy, n_neurons, eta, fn = f"train_{af}_sc{1 if scale else 0}_L{n_hl}_c.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "Accuracy score training set")
make_heatmap(test_accuracy, n_neurons, eta, fn = f"test_{af}_sc{1 if scale else 0}_L{n_hl}_c.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "Accuracy score test set")




#SKlearns neural network
from sklearn.neural_network import MLPClassifier
plt.clf()
clf = MLPClassifier(random_state=1, hidden_layer_sizes = (10,5,10), solver = "sgd", activation = "logistic", batch_size = 10, max_iter=150, learning_rate_init = 0.01, alpha = 0.001).fit(X_train, y_train)
pred_nn = clf.predict(X_test)
print("Prediction accuracy SKlearn: ", accuracy_score_numpy(pred_nn, y_test))
make_confusion_matrix(y_test, pred_nn, fn="cm_heatmapsklearn.pdf", title = "SKlearn Neural Network")
cm_nn = confusion_matrix(y_test, pred_nn)
