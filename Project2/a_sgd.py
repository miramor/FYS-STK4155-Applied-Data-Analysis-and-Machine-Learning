import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import importlib
import functions
importlib.reload(functions)
from functions import *

np.random.seed(2405) # Set a random seed

N = 500 #number of datapoints
d = 5 #complexity
x1 = np.random.uniform(0,1,N)
x2 = np.random.uniform(0,1,N)
y = FrankeFunction(x1,x2)
noise = np.random.normal(0, 1, size=(y.shape)) #Find random noise
y_noisy =  FrankeFunction(x1,x2) + noise*0.2
X = createX(x1,x2,d)


X_train, X_test, y_train, y_test = train_test_split(X,y_noisy,test_size=0.2) #Split the data into training and test sets

#Scales data if true. See scale_data in functions.py for scaling options
scale = True
if scale:
    X_train, X_test = scale_data(X_train, X_test, scale_type = StandardScaler, with_std=False)
    y_train, y_test = scale_data(y_train, y_test, scale_type = StandardScaler, with_std=False)

theta = np.random.randn(len(X_train[0])) #first guess of beta


def SGDSearch_OLS(learnSced=False, gamma = 0, epochs = 1000):
    """
    Grid search with OLS cost function for number of mini batches and learning rate
    """
    if learnSced:
        print("With learnign schedule")
    nbatches = [1, 5, 20, 50, 100,250]
    etaL = np.logspace(-5,-1,5)
    MSE_train_OLS = np.zeros((len(etaL), len(nbatches)))
    MSE_test_OLS = np.zeros((len(etaL), len(nbatches)))
    print(f"OLS, epochs = {epochs}, learning schedule = {learnSced}, gamma = {gamma}")
    print("Start loop over learning rate and mini-batches")

    for i in range(len(etaL)):
        for j in range(len(nbatches)):
            beta, betaL = SGD(X_train, y_train, M=int((N/nbatches[j])), epochs = epochs, gamma = gamma, beta = theta, gradCostFunc = gradCostOls, eta = etaL[i], LS=learnSced)
            ytilde = predict(X_train,beta)
            ypredict = predict(X_test, beta)
            MSE_train_OLS[i, j] = mse(y_train,ytilde)
            MSE_test_OLS[i, j] = mse(y_test,ypredict)

        print(f"{int((i+1)/len(etaL)*100)} % done")
    make_heatmap(MSE_test_OLS, nbatches, etaL, fn = f"ex1_hm_ols_{epochs}.pdf", title = "MSE as a function different hyperparameters", xlabel = "Number of mini batches", ylabel = "$\eta$")

    minMSE_OLS_train_index = np.argwhere(MSE_train_OLS == np.min(MSE_train_OLS))[0]
    minMSE_OLS_test_index = np.argwhere(MSE_test_OLS == np.min(MSE_test_OLS))[0]

    optimal_eta_train = etaL[minMSE_OLS_train_index[0]]
    optimal_eta_test = etaL[minMSE_OLS_test_index[0]]
    optimal_batch_train = nbatches[minMSE_OLS_train_index[1]]
    optimal_batch_test = nbatches[minMSE_OLS_test_index[1]]

    print(f"Optimal eta train {optimal_eta_train}")
    print(f"Optimal eta test {optimal_eta_test}")
    print(f"Optimal mini batches train {optimal_batch_train}")
    print(f"Optimal mini batches test {optimal_batch_test}")
    print("\n")

    betaSKOLSTr = SklearnSGD(X_train, y_train, penalty = None, eta = optimal_eta_train, epochs=epochs)
    betaSKOLSTe = SklearnSGD(X_train, y_train, penalty = None, eta = optimal_eta_test, epochs=epochs)
    y_tildeSKOLS = predict(X_train,betaSKOLSTr)
    y_predictSKOLS = predict(X_test,betaSKOLSTe)


    print(f"MSE train SGD: {np.min(MSE_train_OLS)}, eta={optimal_eta_train}")
    print(f"MSE train Sklearn: {mse(y_train,y_tildeSKOLS)}, eta={optimal_eta_train}")
    print("\n")
    print(f"MSE test SGD: {np.min(MSE_test_OLS)}, eta={optimal_eta_test}")
    print(f"MSE test Sklearn : {mse(y_test,y_predictSKOLS)}, eta={optimal_eta_test}")

def SGDSearch_Ridge(learnSced=False, gamma = 0, epochs = 1000):
    """
    Grid search with Ridge cost function for number of mini batches and learning rate
    """
    nbatches = [1, 5, 20, 50, 100, 250]
    eta_ = 0.1
    lambdaL = np.logspace(-6,-2,5)

    if learnSced:
        print("With learnign schedule")

    MSE_train_Ridge = np.zeros((len(lambdaL), len(nbatches)))
    MSE_test_Ridge = np.zeros((len(lambdaL), len(nbatches)))
    print("\n")
    print(f"Ridge, epochs = {epochs}, eta = {eta_}, learning schedule = {learnSced}, gamma = {gamma}")
    print("Start loop over hyperparameter and mini-batches")
    for i in range(len(lambdaL)):
        for j in range(len(nbatches)):
            beta, betaL = SGD(X_train, y_train, M=int((N/nbatches[j])), epochs = epochs, beta = theta, gradCostFunc = gradCostRidge, eta = eta_, gamma = gamma, lmb=lambdaL[i], LS=learnSced)
            ytilde = predict(X_train,beta)
            ypredict = predict(X_test, beta)
            MSE_train_Ridge[i, j] = mse(y_train,ytilde)
            MSE_test_Ridge[i, j] = mse(y_test,ypredict)

        print(f"{int((i+1)/len(lambdaL)*100)} % done")

    minMSE_Ridge_train_index = np.argwhere(MSE_train_Ridge == np.min(MSE_train_Ridge))[0]
    minMSE_Ridge_test_index = np.argwhere(MSE_test_Ridge == np.min(MSE_test_Ridge))[0]

    optimal_lmb_train = lambdaL[minMSE_Ridge_train_index[0]]
    optimal_lmb_test = lambdaL[minMSE_Ridge_test_index[0]]
    optimal_batch_train = nbatches[minMSE_Ridge_train_index[1]]
    optimal_batch_test = nbatches[minMSE_Ridge_test_index[1]]
    make_heatmap(MSE_test_Ridge, nbatches, lambdaL, fn = f"ex1_hm_R_{epochs}", xlabel = "Number of mini batches", ylabel = "$\lambda$")
    print("Results for Ridge")
    print(f"Optimal lambda train {optimal_lmb_train}")
    print(f"Optimal lambda test {optimal_lmb_test}")
    print(f"Optimal mini batches train {optimal_batch_train}")
    print(f"Optimal mini batches test {optimal_batch_test}")

    print("\n")
    betaSKRidTr = SklearnSGD(X_train, y_train, penalty = "l2", eta = eta_, epochs=epochs, alpha=optimal_lmb_train)
    betaSKRidTe = SklearnSGD(X_train, y_train, penalty = "l2", eta = eta_, epochs=epochs, alpha=optimal_lmb_test)
    y_tildeSKRid = predict(X_train,betaSKRidTr)
    y_predictSKRid = predict(X_test,betaSKRidTe)

    print(f"MSE train SGD: {np.min(MSE_train_Ridge)}, lmb={optimal_lmb_train}" )
    print(f"MSE train Sklearn: {mse(y_train,y_tildeSKRid)}, eta = {eta_}, lmb={optimal_lmb_train}")
    print("\n")
    print(f"MSE test SGD: {np.min(MSE_test_Ridge)} lmb={optimal_lmb_test}")
    print(f"MSE test Sklearn : {mse(y_test,y_predictSKRid)}, eta={eta_}, lmb={optimal_lmb_test}")

def SGDSearch_Ridge2(learnSced=False, gamma = 0, epochs = 1000):
    """
    Grid search with Ridge cost function for normalization parameter parameter and learning rate
    """
    if learnSced:
        print("With learnign schedule")
    
    eta_ = 0.1
    etaLRid = np.logspace(-4,-1,4)
    lambdaL = np.logspace(-6,-2,5)
    
    MSE_train_Ridge = np.zeros((len(lambdaL), len(etaLRid)))
    MSE_test_Ridge = np.zeros((len(lambdaL), len(etaLRid)))

    print("\n")
    print(f"Ridge, epochs = {epochs}, eta = {eta_}, learning schedule = {learnSced}, gamma = {gamma}")
    print("Start loop over hyperparameter and mini-batches")
    for i in range(len(lambdaL)):
        for j in range(len(etaLRid)):
            beta, betaL = SGD(X_train, y_train, M=5, epochs = epochs, beta = theta, gradCostFunc = gradCostRidge, eta = etaLRid[j], gamma = gamma, lmb=lambdaL[i], LS=learnSced)
            ytilde = predict(X_train,beta)
            ypredict = predict(X_test, beta)
            MSE_train_Ridge[i, j] = mse(y_train,ytilde)
            MSE_test_Ridge[i, j] = mse(y_test,ypredict)

        print(f"{int((i+1)/len(lambdaL)*100)} % done")


    minMSE_Ridge_train_index = np.argwhere(MSE_train_Ridge == np.min(MSE_train_Ridge))[0]
    minMSE_Ridge_test_index = np.argwhere(MSE_test_Ridge == np.min(MSE_test_Ridge))[0]

    optimal_lmb_train = lambdaL[minMSE_Ridge_train_index[0]]
    optimal_lmb_test = lambdaL[minMSE_Ridge_test_index[0]]
    optimal_eta_train = etaLRid[minMSE_Ridge_train_index[1]]
    optimal_eta_test = etaLRid[minMSE_Ridge_test_index[1]]
    make_heatmap(MSE_test_Ridge, etaLRid, lambdaL, title = "MSE as a funciton of different hyperparameters",fn = f"ex1_hm_R_{epochs}.pdf", xlabel = "$\eta$", ylabel = "$\lambda$")
    print("Results for Ridge")
    print(f"Optimal lambda train {optimal_lmb_train}")
    print(f"Optimal lambda test {optimal_lmb_test}")
    print(f"Optimal eta train {optimal_eta_train}")
    print(f"Optimal eta test {optimal_eta_test}")

#Can choose learnSced true to use learning schedule. Gamma is momentum parameter
SGDSearch_OLS(learnSced=False, gamma = 0.01, epochs = 1000)
SGDSearch_Ridge(learnSced=False, gamma = 0.01, epochs = 1000)
SGDSearch_Ridge2(learnSced=False, gamma = 0.01, epochs = 1000)


def runningMSE(gamma=0,epoch=1000, eta = 0.0001, LS = False):
    """
    Calculates the MSE for each epoch.
    """
    beta, betaL = SGD(X_train, y_train, M=5, epochs = epoch, beta = theta, gradCostFunc = gradCostRidge,
        eta = eta, gamma = gamma, lmb=0, LS=LS)
    MSE_train = []
    MSE_test = []
    for b in betaL:
        ytilde = predict(X_train, b)
        ypredict = predict(X_test, b)
        MSE_train.append(mse(y_train,ytilde))
        MSE_test.append(mse(y_test,ypredict))

    return MSE_train, MSE_test



def plot_momLS():
    """
    Plots the MSE for different values of 
    momentum paramater gamma.
    Also with and without a learning schedule.
    """
    epochs = 50; eta = 0.0001
    MSE_train1, MSE_test1 = runningMSE(gamma = 0, epoch = epochs, eta = eta, LS = True)
    MSE_train2, MSE_test2 = runningMSE(gamma = 0, epoch = epochs, eta = eta, LS = False)
    MSE_train3, MSE_test3 = runningMSE(gamma = 0.5, epoch = epochs, eta = eta, LS = True)
    MSE_train4, MSE_test4 = runningMSE(gamma = 0.5, epoch = epochs, eta = eta, LS = False)
    MSE_train5, MSE_test5 = runningMSE(gamma = 0.9, epoch = epochs, eta = eta, LS = True)
    MSE_train6, MSE_test6 = runningMSE(gamma = 0.9, epoch = epochs, eta = eta, LS = False)

    plt.plot(range(epochs), MSE_test1, label = f"$\gamma = 0$, LS=True")
    plt.plot(range(epochs), MSE_test2, label = f"$\gamma = 0$, LS=False")
    plt.plot(range(epochs), MSE_test3, label = f"$\gamma = 0.5$, LS=True")
    plt.plot(range(epochs), MSE_test4, label = f"$\gamma = 0.5$, LS=False")
    plt.plot(range(epochs), MSE_test5, label = f"$\gamma = 0.9$, LS=True")
    plt.plot(range(epochs), MSE_test6, label = f"$\gamma = 0.9$, LS=False")

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Rate of convergence")
    #plt.ylim(0,0.1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"convergenceLS_e{epochs}_eta1emin4.pdf", bbox_inches='tight')
    plt.show()
