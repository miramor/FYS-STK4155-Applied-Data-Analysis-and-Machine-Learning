import sys
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


tts = train_test_split(X,y_noisy,test_size=0.2) #Split the data into training and test sets
theta = np.random.randn(len(X[0])) #first guess of beta
scale =False
if scale: #scala data
    tts[0] =tts[0][:,1:]
    tts[1] =tts[1][:,1:]
    theta = np.random.randn(len(tts[0][0])) #first guess of beta
    scaler = StandardScaler(with_std=False)
    scaler.fit(tts[0])
    tts[0] = scaler.transform(tts[0])
    tts[1] = scaler.transform(tts[1])
    scaler2 = StandardScaler(with_std=False)
    scaler2.fit(tts[2].reshape(-1,1))
    tts[2] = scaler2.transform(tts[2].reshape(-1,1))
    tts[3] = scaler2.transform(tts[3].reshape(-1,1))
    tts[2] =tts[2].flatten()
    tts[3] =tts[3].flatten()
#Regular OLS Regression
betaOLS2 = ols(tts[0], tts[2]) #regular OLS regression
y_tildeOLS2 = predict(tts[0],betaOLS2) #regular OLS regression
y_predictOLS2 = predict(tts[1],betaOLS2)

#OLS
def SGDSearch_OLS(learnSced=False):
    nbatches = [1, 5, 20, 50, 100,250]
    etaL = np.logspace(-5,-1,5)
    epoch = 10000
    MSE_train_OLS = np.zeros((len(etaL), len(nbatches)))
    MSE_test_OLS = np.zeros((len(etaL), len(nbatches)))
    print(f"OLS, epochs = {epoch}, learning schedule = {learnSced}")
    print("Start loop over learning rate and mini-batches")

    for i in range(len(etaL)):
        for j in range(len(nbatches)):
            beta = SGD(tts[0], tts[2], M=int((N/nbatches[j])), epochs = epoch, beta = theta, gradCostFunc = gradCostOls, eta = etaL[i], LS=learnSced)
            ytilde = predict(tts[0],beta)
            ypredict = predict(tts[1], beta)
            MSE_train_OLS[i, j] = mse(tts[2],ytilde)
            MSE_test_OLS[i, j] = mse(tts[3],ypredict)

        print(f"{int((i+1)/len(etaL)*100)} % done")
    make_heatmap(MSE_test_OLS, nbatches, etaL)
    #args = np.argmin(MSE_test_OLS,axis=2)
    #MSE_test_OLS = np.min(MSE_test_OLS,axis=2)
    #MSE_train_OLS = MSE_train_OLS[:,:,args]
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

    betaSKOLSTr = SklearnSGD(tts[0], tts[2], penalty = None, eta = optimal_eta_train, epochs=epoch)
    betaSKOLSTe = SklearnSGD(tts[0], tts[2], penalty = None, eta = optimal_eta_test, epochs=epoch)
    y_tildeSKOLS = predict(tts[0],betaSKOLSTr)
    y_predictSKOLS = predict(tts[1],betaSKOLSTe)


    print(f"MSE train SGD: {np.min(MSE_train_OLS)}, eta={optimal_eta_train}")
    print(f"MSE train Sklearn: {mse(tts[2],y_tildeSKOLS)}, eta={optimal_eta_train}")
    print(f"MSE train Regular OLS: {mse(tts[2],y_tildeOLS2)}")
    print("\n")
    print(f"MSE test SGD: {np.min(MSE_test_OLS)}, eta={optimal_eta_test}")
    print(f"MSE test Sklearn : {mse(tts[3],y_predictSKOLS)}, eta={optimal_eta_test}")
    print(f"MSE test Regular OLS: {mse(tts[3],y_predictOLS2)}")

def SGDSearch_Ridge(learnSced=False):
    epoch = 10000
    nbatches = [1, 5, 20, 50, 100, 250]
    eta_ = 0.01
    etaLRid = np.logspace(-3,-0.8,4)
    lambdaL = np.logspace(-6,-2,5)
    MSE_train_Ridge = np.zeros((len(lambdaL), len(nbatches)))
    MSE_test_Ridge = np.zeros((len(lambdaL), len(nbatches)))
    print("\n")
    print(f"Ridge, epochs = {epoch}, eta = {eta_}, learning schedule = {learnSced}")
    print("Start loop over hyperparameter and mini-batches")
    for i in range(len(lambdaL)):
        for j in range(len(nbatches)):
            beta = SGD(tts[0], tts[2], M=int((N/nbatches[j])), epochs = epoch, beta = theta, gradCostFunc = gradCostRidge, eta = eta_, lmb=lambdaL[i], LS=learnSced)
            ytilde = predict(tts[0],beta)
            ypredict = predict(tts[1], beta)
            MSE_train_Ridge[i, j] = mse(tts[2],ytilde)
            MSE_test_Ridge[i, j] = mse(tts[3],ypredict)

        print(f"{int((i+1)/len(lambdaL)*100)} % done")


    # args = np.argmin(MSE_test_OLS,axis=2)
    # MSE_test_Ridge = np.min(MSE_test_Ridge,axis=2)
    # MSE_train_Ridge = MSE_train_Ridge[:,:,args]
    minMSE_Ridge_train_index = np.argwhere(MSE_train_Ridge == np.min(MSE_train_Ridge))[0]
    minMSE_Ridge_test_index = np.argwhere(MSE_test_Ridge == np.min(MSE_test_Ridge))[0]

    optimal_lmb_train = lambdaL[minMSE_Ridge_train_index[0]]
    optimal_lmb_test = lambdaL[minMSE_Ridge_test_index[0]]
    optimal_batch_train = nbatches[minMSE_Ridge_train_index[1]]
    optimal_batch_test = nbatches[minMSE_Ridge_test_index[1]]
    make_heatmap(MSE_test_Ridge, nbatches, lambdaL)
    print("Results for Ridge")
    print(f"Optimal lambda train {optimal_lmb_train}")
    print(f"Optimal lambda test {optimal_lmb_test}")
    print(f"Optimal mini batches train {optimal_batch_train}")
    print(f"Optimal mini batches test {optimal_batch_test}")

    print("\n")
    betaSKRidTr = SklearnSGD(tts[0], tts[2], penalty = "l2", eta = eta_, epochs=epoch, alpha=optimal_lmb_test)
    betaSKRidTe = SklearnSGD(tts[0], tts[2], penalty = "l2", eta = eta_, epochs=epoch, alpha=optimal_lmb_test)
    y_tildeSKRid = predict(tts[0],betaSKRidTr)
    y_predictSKRid = predict(tts[1],betaSKRidTe)
    betaRidge2 = ridge(tts[0], tts[2], optimal_lmb_test) #regular Ridge regression
    y_tildeRidge2 = predict(tts[0],betaRidge2) #regular Ridge regression
    y_predictRidge2 = predict(tts[1],betaRidge2)

    print(f"MSE train SGD: {np.min(MSE_train_Ridge)}, lmb={optimal_lmb_train}" )
    print(f"MSE train Sklearn: {mse(tts[2],y_tildeSKRid)}, eta = {eta_}, lmb={optimal_lmb_train}")
    print(f"MSE train Regular Ridge: {mse(tts[2],y_tildeRidge2)}")
    print("\n")
    print(f"MSE test SGD: {np.min(MSE_test_Ridge)} lmb={optimal_lmb_test}")
    print(f"MSE test Sklearn : {mse(tts[3],y_predictSKRid)}, eta={eta_}, lmb={optimal_lmb_test}")
    print(f"MSE test Regular Ridge: {mse(tts[3],y_predictRidge2)}")

SGDSearch_OLS(learnSced=False)
#SGDSearch_OLS(learnSced=True)
SGDSearch_Ridge(learnSced=False)
#SGDSearch_Ridge(learnSced=True)
#SklearnSGD følsom bedre når eta = 0.1
