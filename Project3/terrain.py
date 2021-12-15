from inspect import CO_VARARGS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from tqdm import tqdm
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams["figure.figsize"]=12,12
import importlib
import functions
import NNReg
importlib.reload(functions); importlib.reload(NNReg)
from NNReg import NeuralNetwork
from functions import *
from mlxtend.evaluate import bias_variance_decomp
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

data = imread('SRTM_data_Norway_1.tif') #All data
terrain = data[:200,:200] #Subset
Y = terrain.ravel() #1d array of subset

dim = terrain.shape
x1,x2 = np.meshgrid(range(dim[0]), range(dim[1]))
X1 = x1.ravel()
X2 = x2.ravel()

print(type(X1[0]))
#Scale variables
#X1_scaled, X2_scaled = scale_data(X1,X2)

#Set up design matrix
print(200**10)
X = create_X(X1, X2, 10)

print(np.min(X))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.4)

X_train, X_test = scale_data(X_train, X_test)
Y_train, Y_test = scale_data(Y_train, Y_test)


def Ridge_bs(X_train, X_test, y_train, y_test, nb = 500, plot = False):
    complexity = X_train.shape[1]
    degree = int((-3+np.sqrt(9-8*(1-complexity)))/2)

    MSE_train = np.zeros(degree)
    MSE_test = np.zeros(degree)
    Bias = np.zeros(degree)
    Variance = np.zeros(degree)

    for i in tqdm(range(nb)):
        y_pred_bs = np.zeros((degree, len(y_test), nb))
        X_sample, y_sample  = bootstrap(X_train, y_train)
         
        for d in range(1,degree + 1):
            c = int((d+1)*(d+2)/2)
            regr = Ridge(alpha = 0.001).fit(X_sample[:,:c], y_sample)
            y_tilde = regr.predict(X_sample[:,:c])
            y_predict = regr.predict(X_test[:,:c])
            y_pred_bs[d-1,:,i] = y_predict.reshape(len(y_pred_bs[d-1]))

            MSE_train[d-1] += mse(y_tilde, y_sample)
            MSE_test[d-1] += mse(y_predict, y_test)

    for d in range(degree):
        Variance[d] = np.mean(np.var(y_pred_bs[d], axis = 1))
        Bias[d] = np.mean( ( y_test - np.mean(y_pred_bs[d], axis = 1) )**2 )

    MSE_train /= nb; MSE_test /= nb

    if plot:
        plt.plot(range(1,degree+1), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, 'o-', label = "Variance")
        plt.plot(range(1,degree+1), MSE_train, 'o-', label = "MSE Train")
        plt.plot(range(1,degree+1), MSE_test, 'o-', label = "MSE Test")
        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.title("Ridge Bias-Variance Trade Off")
        plt.ylim(0,0.75)
        plt.legend()
        plt.show()

def Lasso_bs(X_train, X_test, y_train, y_test, nb = 500, plot = False):
    complexity = X_train.shape[1]

    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for c in range(1,complexity+1):
        y_pred_bs = np.zeros((len(y_test), nb))
        for i in range(nb):
            X_sample, y_sample  = bootstrap(X_train[:,:c], y_train)

            regr = Lasso(alpha = 0.0001, max_iter = 5000, fit_intercept=False).fit(X_sample, y_sample)
            y_tilde = regr.predict(X_sample)
            y_predict = regr.predict(X_test[:,:c])
            y_pred_bs[:,i] = y_predict.reshape(len(y_pred_bs))

            MSE_train[c-1] += mse(y_tilde, y_sample)
            MSE_test[c-1] += mse(y_predict, y_test)

        Variance[c-1] = np.mean(np.var(y_pred_bs, axis = 1))
        Bias[c-1] = np.mean( ( y_test - np.mean(y_pred_bs, axis = 1) )**2 )

    MSE_train /= nb; MSE_test /= nb

    if plot:
        plt.plot(range(1,complexity+1), Bias, label = "Bias$^2$")
        plt.plot(range(1,complexity+1), Variance, label = "Variance")
        plt.plot(range(1,complexity+1), MSE_train, label = "MSE Train")
        plt.plot(range(1,complexity+1), MSE_test, label = "MSE Test")
        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.title("Lasso Bias Variance Trade-Off")
        #plt.ylim(0,2)
        plt.legend()
        plt.show()

def OLS_bs(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = X_train.shape[1]
    degree = int((-3+np.sqrt(9-8*(1-complexity)))/2)

    MSE_train = np.zeros(degree)
    MSE_test = np.zeros(degree)
    Bias = np.zeros(degree)
    Variance = np.zeros(degree)

    for i in tqdm(range(nb)):
        y_pred_bs = np.zeros((degree, len(y_test), nb))
        X_sample, y_sample  = bootstrap(X_train, y_train)

        for d in range(1,degree+1):
            c = int((d+1)*(d+2)/2)
            regr = LinearRegression().fit(X_sample[:,:c], y_sample)
            y_tilde = regr.predict(X_sample[:,:c])
            y_predict = regr.predict(X_test[:,:c])
            y_pred_bs[d-1,:,i] = y_predict.reshape(len(y_pred_bs[d-1]))

            MSE_train[d-1] += mse(y_tilde, y_sample)
            MSE_test[d-1] += mse(y_predict, y_test)

    for d in range(degree):
        Variance[d] = np.mean(np.var(y_pred_bs[d], axis = 1))
        Bias[d] = np.mean( ( y_test - np.mean(y_pred_bs[d], axis = 1) )**2 )


    MSE_train /= nb; MSE_test /= nb

    if plot:
        plt.plot(range(1,degree+1), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, 'o-', label = "Variance")
        plt.plot(range(1,degree+1), MSE_train, 'o-', label = "MSE Train")
        plt.plot(range(1,degree+1), MSE_test, 'o-', label = "MSE Test")
        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.title("OLS Bias-Variance Trade Off")
        plt.ylim(0,0.75)
        #plt.xlim(0,40)
        plt.legend()
        plt.show()



def DecisionTree_bs(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = 15

    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for i in tqdm(range(nb)):
        y_pred_bs = np.zeros((complexity, len(y_test), nb))
        X_sample, y_sample  = bootstrap(X_train, y_train)

        for c in range(1,complexity+1):
            regr = DecisionTreeRegressor(max_depth=2*c).fit(X_sample, y_sample)
            print(regr.get_n_leaves())
            y_tilde = regr.predict(X_sample)
            y_predict = regr.predict(X_test)
            y_pred_bs[c-1,:,i] = y_predict.reshape(len(y_pred_bs[c-1]))

            MSE_train[c-1] += mse(y_tilde, y_sample)
            MSE_test[c-1] += mse(y_predict, y_test)

    for c in range(complexity):
        Variance[c] = np.mean(np.var(y_pred_bs[c], axis = 1))
        Bias[c] = np.mean( ( y_test - np.mean(y_pred_bs[c], axis = 1) )**2 )


    MSE_train /= nb; MSE_test /= nb

    if plot:
        plt.plot(range(1,complexity+1), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(1,complexity+1), Variance, 'o-', label = "Variance")
        plt.plot(range(1,complexity+1), MSE_train, 'o-', label = "MSE Train")
        plt.plot(range(1,complexity+1), MSE_test, 'o-', label = "MSE Test")
        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.title("Tree Bias-Variance Trade Off")
        plt.ylim(0,0.1)
        #plt.xlim(0,40)
        plt.legend()
        plt.show()



def NN_bs(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = 20
    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for i in tqdm(range(nb)):
        y_pred_bs = np.zeros((complexity, len(y_test), nb))
        X_sample, y_sample  = bootstrap(X_train, y_train)

        for c in range(1, complexity + 1):
            NN = NeuralNetwork(X_sample, y_sample, epochs = 2000, batch_size = 50, n_categories = 1,
                                eta = 1e-5, lmbd = 0.001, n_hidden_neurons = [c,c], activation_function = relu)
            NN.train()
            y_tilde = NN.predict_reg(X_sample)
            y_predict = NN.predict_reg(X_test)

            y_pred_bs[c-1,:,i] = y_predict.reshape(len(y_pred_bs[c-1]))

            MSE_train[c-1] += mse(y_tilde, y_sample)
            MSE_test[c-1] += mse(y_predict, y_test)

    for c in range(complexity):
        Variance[c] = np.mean(np.var(y_pred_bs[c], axis = 1))
        Bias[c] = np.mean( ( y_test - np.mean(y_pred_bs[c], axis = 1) )**2 )
            
    MSE_train /= nb; MSE_test /= nb

    if plot:
        plt.plot(range(1,complexity+1), Bias, label = "Bias$^2$")
        plt.plot(range(1,complexity+1), Variance, label = "Variance")
        plt.plot(range(1,complexity+1), MSE_train, label = "MSE Train")
        plt.plot(range(1,complexity+1), MSE_test, label = "MSE Test")
        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.title("NN Bias-Variance Trade Off")
        #plt.ylim(0,2)
        #plt.xlim(0,40)
        plt.legend()
        plt.show()
"""
            regr = MLPRegressor(activation = "relu", solver = "sgd", max_iter = 5000, learning_rate_init = 1e-5, 
                                alpha = 0.001, batch_size=20, momentum = 0, hidden_layer_sizes=(c,c)).fit(X_sample, y_sample)

            y_tilde = regr.predict(X_sample)
            y_predict = regr.predict(X_test)
            y_pred_bs[:,i] = y_predict.reshape(len(y_pred_bs))

            #print(y_tilde)
            MSE_train[c-1] += mse(y_tilde, y_sample)
            MSE_test[c-1] += mse(y_predict, y_test)
"""




"""
OLS_bs(X_train, X_test, Y_train, Y_test, nb=10, plot=True)
Ridge_bs(X_train, X_test, Y_train, Y_test, nb=10, plot=True)
#Lasso_bs(X_train, X_test, Y_train, Y_test, nb=1, plot=True)
#NN_bs(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb = 1, plot=True)
DecisionTree_bs(X_train, X_test, Y_train, Y_test, nb = 10, plot=True)
#SVM_bs(X_train , X_test , y_train , y_test , nb = 10, plot=True)
"""
def callOLS():
    OLS_bs(X_train, X_test, Y_train, Y_test, nb=1, plot=True)


def callRidge():
    Ridge_bs(X_train, X_test, Y_train, Y_test, nb=1, plot=True)

def callDT():
    DecisionTree_bs(X_train, X_test, Y_train, Y_test, nb = 1, plot=True)

def callNN():
    NN_bs(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb = 1, plot=True)

def BVT_OLS(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = X_train.shape[1]
    degree = int((-3+np.sqrt(9-8*(1-complexity)))/2)
    Loss = np.zeros(degree)
    Variance = np.zeros(degree)
    Bias = np.zeros(degree)
    
    for d in tqdm(range(1, degree + 1)):

        c = int((d+1)*(d+2)/2)
        regr = LinearRegression()
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train[:,:c], y_train, X_test[:,:c], y_test, loss='mse', num_rounds = nb)

        Loss[d-1] = avg_expected_loss
        Variance[d-1] = avg_var
        Bias[d-1] = avg_bias
    
    if plot:
        plt.plot(range(1,degree+1), Bias, label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, label = "Variance")
        plt.plot(range(1,degree+1), Loss, label = "Loss")
        plt.xlabel("Polynomial Degree")
        plt.ylabel("Error")
        plt.title("OLS Bias-Variance Trade Off")
        plt.ylim(0,0.04)
        #plt.xlim(0,40)
        plt.legend()
        plt.show()

def BVT_Ridge(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = X_train.shape[1]
    degree = int((-3+np.sqrt(9-8*(1-complexity)))/2)
    Loss = np.zeros(degree)
    Variance = np.zeros(degree)
    Bias = np.zeros(degree)
    
    for d in tqdm(range(1, degree + 1)):

        c = int((d+1)*(d+2)/2)
        regr = Ridge(alpha = 0.01)
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train[:,1:c], y_train, X_test[:,1:c], y_test, loss='mse', num_rounds = nb)

        Loss[d-1] = avg_expected_loss
        Variance[d-1] = avg_var
        Bias[d-1] = avg_bias
    
    if plot:
        plt.plot(range(1,degree+1), Bias, label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, label = "Variance")
        plt.plot(range(1,degree+1), Loss, label = "Loss")
        plt.xlabel("Polynomial Degree")
        plt.ylabel("Error")
        plt.title("Ridge Bias-Variance Trade Off")
        plt.ylim(0,0.2)
        #plt.xlim(0,40)
        plt.legend()
        plt.show() 

def BVT_Lasso(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = X_train.shape[1]
    degree = int((-3+np.sqrt(9-8*(1-complexity)))/2)
    Loss = np.zeros(degree)
    Variance = np.zeros(degree)
    Bias = np.zeros(degree)
    
    for d in tqdm(range(1, degree + 1)):

        c = int((d+1)*(d+2)/2)
        regr = Lasso(alpha = 0.01, max_iter=5000, fit_intercept=False)
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train[:,1:c], y_train, X_test[:,1:c], y_test, loss='mse', num_rounds = nb)

        Loss[d-1] = avg_expected_loss
        Variance[d-1] = avg_var
        Bias[d-1] = avg_bias
    
    if plot:
        plt.plot(range(1,degree+1), Bias, label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, label = "Variance")
        plt.plot(range(1,degree+1), Loss, label = "Loss")
        plt.xlabel("Polynomial Degree")
        plt.ylabel("Error")
        plt.title("Lasso Bias-Variance Trade Off")
        plt.ylim(0,0.4)
        #plt.xlim(0,40)
        plt.legend()
        plt.show() 

def BVT_DT(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = 15
    Loss = np.zeros(complexity)
    Variance = np.zeros(complexity)
    Bias = np.zeros(complexity)
    leaf = np.logspace(1,complexity,complexity,base=2).astype(int)
    print(leaf)
    for c in tqdm(range(complexity)):
        regr = DecisionTreeRegressor(max_leaf_nodes=leaf[c])
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train, y_train, X_test, y_test, loss='mse', num_rounds = nb)

        Loss[c] = avg_expected_loss
        Variance[c] = avg_var
        Bias[c] = avg_bias
    
    if plot:
        plt.plot(np.log2(leaf), Bias, label = "Bias$^2$")
        plt.plot(np.log2(leaf), Variance, label = "Variance")
        plt.plot(np.log2(leaf), Loss, label = "Loss")
        plt.xlabel("log2(Leaves)")
        plt.ylabel("Error")
        plt.title("Decision Tree Bias-Variance Trade Off")
        plt.ylim(0,0.05)
        #plt.xlim(0,40)
        plt.legend()
        plt.show() 

def BVT_NN(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = 12
    Loss = np.zeros(complexity)
    Variance = np.zeros(complexity)
    Bias = np.zeros(complexity)
    
    nodes = np.logspace(0,complexity-1, complexity, base = 2).astype(int)
    for c in tqdm(range(complexity)):

        regr = MLPRegressor(learning_rate_init=0.1,hidden_layer_sizes=(nodes[c],nodes[c]) , max_iter=10000)
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train, y_train, X_test, y_test, loss='mse', num_rounds = nb)

        Loss[c] = avg_expected_loss
        Variance[c] = avg_var
        Bias[c] = avg_bias
    
    if plot:
        plt.plot(np.log2(nodes), Bias, label = "Bias$^2$")
        plt.plot(np.log2(nodes), Variance, label = "Variance")
        plt.plot(np.log2(nodes), Loss, label = "Loss")
        plt.xlabel("log2(Nodes)")
        plt.ylabel("Error")
        plt.title("Neural Network Bias-Variance Trade Off")
        plt.ylim(0,0.1)
        #plt.xlim(0,40)
        plt.legend()
        plt.show() 


def BVT_SVM(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = X_train.shape[1]
    degree = int((-3+np.sqrt(9-8*(1-complexity)))/2)
    degree = 10
    Loss = np.zeros(degree)
    Variance = np.zeros(degree)
    Bias = np.zeros(degree)
    gamma = np.logspace(-3,3,degree)

    for d in tqdm(range(1, degree + 1)):
        regr = SVR(gamma = gamma[d-1])
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train, y_train, X_test, y_test, loss='mse', num_rounds = nb)
        #c = int((d+1)*(d+2)/2)
        #regr = SVR(C = 5)
        #avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        #    regr, X_train[:,1:c], y_train, X_test[:,1:c], y_test, loss='mse', num_rounds = nb)

        Loss[d-1] = avg_expected_loss
        Variance[d-1] = avg_var
        Bias[d-1] = avg_bias
    
    if plot:
        plt.plot(np.log(gamma), Bias, label = "Bias$^2$")
        plt.plot(np.log(gamma), Variance, label = "Variance")
        plt.plot(np.log(gamma), Loss, label = "Loss")
        plt.xlabel("log($\gamma$)")
        plt.ylabel("Error")
        plt.title("Support Vector Machine Bias-Variance Trade Off")
        #plt.ylim(0,0.2)
        #plt.xlim(0,40)
        plt.legend()
        plt.show() 



def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

if __name__ == '__main__':
    #__spec__ = None
    #runInParallel(callOLS, callRidge, callDT)
    #OLS_bs(X_train, X_test, Y_train, Y_test, nb=5, plot=True)
    #Ridge_bs(X_train, X_test, Y_train, Y_test, nb=5, plot=True)
    #Lasso_bs(X_train, X_test, Y_train, Y_test, nb=1, plot=True)
    #NN_bs(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb = 1, plot=True)
    #DecisionTree_bs(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb = 5, plot=True)
    #SVM_bs(X_train , X_test , y_train , y_test , nb = 10, plot=True)
    BVT_OLS(X_train, X_test, Y_train, Y_test, nb=5, plot=True)
    #BVT_Ridge(X_train, X_test, Y_train, Y_test, nb=5, plot=True)
    #BVT_Lasso(X_train, X_test, Y_train, Y_test, nb=5, plot=True)
    #BVT_NN(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb=5, plot=True)
    #BVT_SVM(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb=5, plot=True)
    BVT_DT(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb=50, plot=True)
    

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#ax.plot_surface(x1,x2,terrain)
#ax.view_init(20,-20)
#plt.show()



