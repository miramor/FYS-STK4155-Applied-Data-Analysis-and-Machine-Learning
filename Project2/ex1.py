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

#Regular OLS Regression
betaOLS2 = ols(tts[0], tts[2]) #regular OLS regression
y_tildeOLS2 = predict(tts[0],betaOLS2) #regular OLS regression
y_predictOLS2 = predict(tts[1],betaOLS2)

OLS = True
Ridge = False
learnSced = True #learning schedule
#OLS
if OLS:
    epochsL = np.arange(5000,10001,1000)
    nbatches = [1, 5, 20, 50, 100,250]
    etaL = np.logspace(-2,-0.8,10)
    MSE_train_OLS = np.zeros((len(epochsL), len(nbatches)))
    MSE_test_OLS = np.zeros((len(epochsL), len(nbatches)))

    print("Start epoch loop OLS")
    for i in range(len(epochsL)):
        for j in range(len(nbatches)):
            beta = SGD(tts[0], tts[2], M=int((N/nbatches[j])), epochs = epochsL[i], beta = theta, gradCostFunc = gradCostOls, eta = 0.1, LS=learnSced)
            ytilde = predict(tts[0],beta)
            ypredict = predict(tts[1], beta)
            MSE_train_OLS[i, j] = mse(tts[2],ytilde)
            MSE_test_OLS[i, j] = mse(tts[3],ypredict)

        print(f"{int((i+1)/len(epochsL)*100)} % done")

    minMSE_OLS_train_index = np.argwhere(MSE_train_OLS == np.min(MSE_train_OLS))[0]
    minMSE_OLS_test_index = np.argwhere(MSE_test_OLS == np.min(MSE_test_OLS))[0]

    optimal_epoch_train = epochsL[minMSE_OLS_train_index[0]]
    optimal_epoch_test = epochsL[minMSE_OLS_test_index[0]]
    optimal_batch_train = nbatches[minMSE_OLS_train_index[1]]
    optimal_batch_test = nbatches[minMSE_OLS_test_index[1]]
    print("Results for OLS")
    print(f"Optimal epoch train {optimal_epoch_train}")
    print(f"Optimal epoch test {optimal_epoch_test}")
    print(f"Optimal mini batches train {optimal_batch_train}")
    print(f"Optimal mini batches test {optimal_batch_test}")


    MSE_train_OLSeta = np.zeros(len(etaL))
    MSE_test_OLSeta = np.zeros(len(etaL))

    print("Start learning rate loop OLS")
    for i in range(len(etaL)): #use optimal values according to test results
        beta = SGD(tts[0], tts[2], M=int((N/optimal_batch_test)), epochs = optimal_epoch_test, beta = theta, gradCostFunc = gradCostOls, eta = etaL[i], LS=learnSced)
        ytilde = predict(tts[0],beta)
        ypredict = predict(tts[1], beta)
        MSE_train_OLSeta[i] = mse(tts[2],ytilde)
        MSE_test_OLSeta[i] = mse(tts[3],ypredict)
        print(f"{int((i+1)/len(etaL)*100)} % done")
    print("\n")
    print(f"Results for epochs = {optimal_epoch_test}, mini batch = {optimal_batch_test} and learning schedule = {learnSced}")
    plotmseLR(MSE_test_OLSeta, etaL, LS=learnSced)

    eta_trainO =etaL[np.argmin(MSE_train_OLSeta)]
    eta_testO = etaL[np.argmin(MSE_test_OLSeta)] #optimal eta value
    betaSKOLSTr = SklearnSGD(tts[0], tts[2], penalty = None, eta = eta_trainO, epochs=optimal_epoch_test)
    betaSKOLSTe = SklearnSGD(tts[0], tts[2], penalty = None, eta = eta_testO, epochs=optimal_epoch_test)
    y_tildeSKOLS = predict(tts[0],betaSKOLSTr)
    y_predictSKOLS = predict(tts[1],betaSKOLSTe)


    print(f"MSE train SGD: {np.min(MSE_train_OLSeta)}, eta={eta_trainO}")
    print(f"MSE train Sklearn: {mse(tts[2],y_tildeSKOLS)}, eta={eta_trainO}")
    print(f"MSE train Regular OLS: {mse(tts[2],y_tildeOLS2)}")
    print("\n")
    print(f"MSE test SGD: {np.min(MSE_test_OLSeta)}, eta={eta_testO}")
    print(f"MSE test Sklearn : {mse(tts[3],y_predictSKOLS)}, eta={eta_testO}")
    print(f"MSE test Regular OLS: {mse(tts[3],y_predictOLS2)}")

if Ridge:
    epochsL = np.arange(2000,6000,1000)
    nbatches = [1, 5, 20, 50, 100]
    #etaLRid = np.logspace(-3,-1,4)
    etaLRid = np.logspace(-3,-0.8,4)
    lambdaL = np.logspace(-4,-5,4)
    #lambdaL = np.logspace(-4,-5,10)
    #Ridge
    MSE_train_Ridge = np.zeros((len(epochsL), len(nbatches)))
    MSE_test_Ridge = np.zeros((len(epochsL), len(nbatches)))
    print("Start epoch loop Ridge")
    for i in range(len(epochsL)):
        for j in range(len(nbatches)):
            beta = SGD(tts[0], tts[2], M=int((N/nbatches[j])), epochs = epochsL[i], beta = theta, gradCostFunc = gradCostRidge, eta = 0.1, lmb=0.001, LS=learnSced)
            ytilde = predict(tts[0],beta)
            ypredict = predict(tts[1], beta)
            MSE_train_Ridge[i, j] = mse(tts[2],ytilde)
            MSE_test_Ridge[i, j] = mse(tts[3],ypredict)

        print(f"{int((i+1)/len(epochsL)*100)} % done")

    minMSE_Ridge_train_index = np.argwhere(MSE_train_Ridge == np.min(MSE_train_Ridge))[0]
    minMSE_Ridge_test_index = np.argwhere(MSE_test_Ridge == np.min(MSE_test_Ridge))[0]

    optimal_epoch_train = epochsL[minMSE_Ridge_train_index[0]]
    optimal_epoch_test = epochsL[minMSE_Ridge_test_index[0]]
    optimal_batch_train = nbatches[minMSE_Ridge_train_index[1]]
    optimal_batch_test = nbatches[minMSE_Ridge_test_index[1]]
    print("Results for Ridge")
    print(f"Optimal epoch train {optimal_epoch_train}")
    print(f"Optimal epoch test {optimal_epoch_test}")
    print(f"Optimal mini batches train {optimal_batch_train}")
    print(f"Optimal mini batches test {optimal_batch_test}")


    MSE_train_REL = np.zeros((len(etaLRid), len(lambdaL))) #Ridge Eta Lambda
    MSE_test_REL = np.zeros((len(etaLRid), len(lambdaL)))

    print("Start learning rate loop Ridge")

    for i in range(len(etaLRid)):
        for j in range(len(lambdaL)):
            beta = SGD(tts[0], tts[2], M=int((N/optimal_batch_test)), epochs = optimal_epoch_test, beta = theta, gradCostFunc = gradCostRidge, eta = etaLRid[i], lmb = lambdaL[i], LS = learnSced)
            ytilde = predict(tts[0],beta)
            ypredict = predict(tts[1], beta)
            MSE_train_REL[i,j] = mse(tts[2],ytilde)
            MSE_test_REL[i,j] = mse(tts[3],ypredict)
        print(f"{int((i+1)/len(etaLRid)*100)} % done")
    print("\n")
    print(f"Results for epochs = {optimal_epoch_test}, mini batch = {optimal_batch_test} and learning schedule = {learnSced}")
    plotmseREL(MSE_test_REL, lambdaL, etaLRid, LS=learnSced)


    minMSE_REL_train_index = np.argwhere(MSE_train_REL == np.min(MSE_train_REL))[0]
    minMSE_REL_test_index = np.argwhere(MSE_test_REL == np.min(MSE_test_REL))[0]

    etaRid_trainO =etaLRid[minMSE_REL_train_index[0]]
    etaRid_testO = etaLRid[minMSE_REL_test_index[0]] #optimal eta value
    lmbTrain = lambdaL[minMSE_REL_train_index[1]]
    lmbTest = lambdaL[minMSE_REL_test_index[1]]
    betaSKRidTr = SklearnSGD(tts[0], tts[2], penalty = "l2", eta = etaRid_trainO, epochs=optimal_epoch_test, alpha=lmbTrain)
    betaSKRidTe = SklearnSGD(tts[0], tts[2], penalty = "l2", eta = etaRid_testO, epochs=optimal_epoch_test, alpha=lmbTest)
    y_tildeSKRid = predict(tts[0],betaSKRidTr)
    y_predictSKRid = predict(tts[1],betaSKRidTe)
    betaRidge2 = ridge(tts[0], tts[2], lmbTrain) #regular Ridge regression
    y_tildeRidge2 = predict(tts[0],betaRidge2) #regular Ridge regression
    y_predictRidge2 = predict(tts[1],betaRidge2)
    print(f"MSE train SGD: {np.min(MSE_train_REL)}, eta={etaRid_trainO}, lmb={lmbTrain}" )
    print(f"MSE train Sklearn: {mse(tts[2],y_tildeSKRid)}, eta={etaRid_trainO}, lmb={lmbTrain}")
    print(f"MSE train Regular Ridge: {mse(tts[2],y_tildeRidge2)}")
    print("\n")
    print(f"MSE test SGD: {np.min(MSE_test_REL)}, eta={etaRid_testO}, lmb={lmbTest}")
    print(f"MSE test Sklearn : {mse(tts[3],y_predictSKRid)}, eta={etaRid_testO}, lmb={lmbTest}")
    print(f"MSE test Regular Ridge: {mse(tts[3],y_predictRidge2)}")
#SklearnSGD følsom bedre når eta = 0.1
