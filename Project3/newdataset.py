from inspect import CO_VARARGS
from os import replace
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

df = pd.read_csv('./CASP.csv', sep = ",")

features = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']


#Non linear terms
for d in range(2,6):
    for feat in features:
        df.insert(len(df.columns), feat+f'_{d}',df[feat]**d)

X = np.array(df.loc[:, df.columns != "RMSD"])
y = np.array(df.loc[:, df.columns == "RMSD"])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

#X_train, X_test = scale_data(X_train, X_test)
#y_train, y_test = scale_data(y_train.reshape(len(y_train)), y_test.reshape(len(y_test)))



def BVT_OLS(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    degree = X_train.shape[1]

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
        #plt.ylim(0,0.2)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_ols.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show()

def BVT_Ridge(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    degree = X_train.shape[1]
    Loss = np.zeros(degree)
    Variance = np.zeros(degree)
    Bias = np.zeros(degree)
    
    for d in tqdm(range(1, degree + 1)):

        c = int((d+1)*(d+2)/2)
        regr = Ridge(alpha = 0.0005)
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
        #plt.ylim(0,0.02)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_ridge.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show() 

def BVT_Lasso(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    degree = X_train.shape[1]
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
        #plt.ylim(0,0.4)
        #plt.xlim(0,40)
        plt.savefig('bv_tradeoff_lasso.pdf', dpi = 400, bbox_inches = 'tight')
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
        #plt.ylim(0,0.05)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_dt.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show() 

def BVT_NN(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = 10
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
        ##plt.ylim(0,0.2)
        #plt.xlim(0,40)
        plt.savefig('bv_tradeoff_nn.pdf', dpi = 400, bbox_inches = 'tight')
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
    pass
    #__spec__ = None
    #runInParallel(callOLS, callRidge, callDT)
    #OLS_bs(X_train[:,:int(21*20/2)], X_test[:,:int(21*20/2)], Y_train, Y_test, nb=100, plot=True)
    #Ridge_bs(X_train, X_test, Y_train, Y_test, nb=5, plot=True)
    #Lasso_bs(X_train, X_test, Y_train, Y_test, nb=1, plot=True)
    #NN_bs(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb = 1, plot=True)
    #DecisionTree_bs(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb = 5, plot=True)
    #SVM_bs(X_train , X_test , y_train , y_test , nb = 10, plot=True)
    #BVT_OLS(X_train, X_test, y_train, y_test, nb=50, plot=True)
    #BVT_Ridge(X_train, X_test, y_train, y_test, nb=50, plot=True)   
    #BVT_Lasso(X_train, X_test, y_train, y_test, nb=10, plot=True)
    BVT_NN(X_train[:,:9], X_test[:,:9], y_train, y_test, nb=10, plot=True)
    #BVT_SVM(X_train[:,1:3], X_test[:,1:3], y_train, y_test, nb=5, plot=True)
    #BVT_DT(X_train[:,1:3], X_test[:,1:3], y_train, y_test, nb=10, plot=True)