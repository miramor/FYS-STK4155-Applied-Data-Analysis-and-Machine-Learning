import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import importlib
import functions
importlib.reload(functions)
from functions import *

np.random.seed(2405) # Set a random seed

N = 500 #number of datapoints
d = 5 #complexity
M = 100
epochs = 5000
eta = 0.1
x1 = np.random.uniform(0,1,N)
x2 = np.random.uniform(0,1,N)
y = FrankeFunction(x1,x2)
X = createX(x1,x2,d)

tts = train_test_split(X,y,test_size=0.2) #Split the data into training and test sets
theta = np.random.randn(len(X[0])) #first guess of beta
betaOLS1 = SGD(tts[0], tts[2], M=M, epochs = epochs, beta = theta, gradCostFunc = gradCostOls, eta = eta)
betaOLS2 = ols(tts[0], tts[2]) #regular OLS regression
betaSKOLS = SklearnSGD(tts[0], tts[2], penalty = None, eta = eta, epochs=epochs)

lmb = 0.0004
betaRidge1 = SGD(tts[0], tts[2], M=M, epochs = epochs, beta = theta, gradCostFunc = gradCostRidge, eta = eta, lmb = lmb)
betaRidge2 = ridge(tts[0], tts[2], lmb)
betaSKRidge = SklearnSGD(tts[0], tts[2], penalty = "l2", eta=eta, epochs=epochs, alpha = lmb)

#Predicted y-values from training data
y_tildeOLS1 = predict(tts[0],betaOLS1) #SGD
#y_tildetOLS2 = predict(tts[0],betaOLS2) #normal OLS regression
y_tildeSKOLS = predict(tts[0],betaSKOLS)

y_tildeRidge1 = predict(tts[0],betaRidge1) #SGD
#y_tildeRidge2 = predict(tts[0],betaRidge2) #normal Ridge regression
y_tildeSKRidge = predict(tts[0],betaSKRidge)

#Predicted y-values from testing data
y_predictOLS1 = predict(tts[1],betaOLS1)
y_predictOLS2 = predict(tts[1],betaOLS2)
y_predictSKOLS = predict(tts[1],betaSKOLS)

y_predictRidge1 = predict(tts[1],betaRidge1)
y_predictRidge2 = predict(tts[1],betaRidge2)
y_predictSKRidge = predict(tts[1],betaSKRidge)

#y_predictRidge = predict(tts[0],beta2)
print(mse(tts[2],y_tildeOLS1), mse(tts[2],y_tildeSKOLS))
print(mse(tts[2],y_tildeRidge1), mse(tts[2],y_tildeSKRidge))

print(mse(tts[3],y_predictOLS1), mse(tts[3],y_predictSKOLS))
print(mse(tts[3],y_predictRidge1), mse(tts[3],y_predictSKRidge))

epochsL = np.arange(5000,10001,1000)
nbatches = [1, 5, 20, 50, 100,250]
MSE_train_OLS = np.zeros((len(epochsL), len(nbatches)))
MSE_test_OLS = np.zeros((len(epochsL), len(nbatches)))

#OLS
"""
print("Start epoch loop OLS")
for i in range(len(epochsL)):
    for j in range(len(nbatches)):
        beta = SGD(tts[0], tts[2], M=int((N/nbatches[j])), epochs = epochsL[i], beta = theta, gradCostFunc = gradCostOls, eta = 0.1)
        ytilde = predict(tts[0],beta)
        ypredict = predict(tts[1], beta)
        MSE_train_OLS[i, j] = mse(tts[2],ytilde)
        MSE_test_OLS[i, j] = mse(tts[3],ypredict)

    print(f"{int((i+1)/len(epochsL)*100)} % done")

minMSE_OLS_train_index = np.argwhere(MSE_train_OLS == np.min(MSE_train_OLS))[0]
minMSE_OLS_test_index = np.argwhere(MSE_train_OLS == np.min(MSE_train_OLS))[0]

optimal_epoch_train = epochsL[minMSE_OLS_train_index[0]]
optimal_epoch_test = epochsL[minMSE_OLS_test_index[0]]
optimal_batch_train = nbatches[minMSE_OLS_train_index[1]]
optimal_batch_test = nbatches[minMSE_OLS_test_index[1]]
print("Results for OLS")
print(f"Optimal epoch train {optimal_epoch_train}")
print(f"Optimal epoch test {optimal_epoch_test}")
print(f"Optimal mini batches train {optimal_batch_train}")
print(f"Optimal mini batches test {optimal_batch_test}")

etaL = np.logspace(-2,-0.8,20)

MSE_train_OLSeta = np.zeros(len(etaL))
MSE_test_OLSeta = np.zeros(len(etaL))

print("Start learning rate loop OLS")
for i in range(len(etaL)):
    beta = SGD(tts[0], tts[2], M=int((N/optimal_batch_test)), epochs = optimal_epoch_test, beta = theta, gradCostFunc = gradCostOls, eta = etaL[i])
    ytilde = predict(tts[0],beta)
    ypredict = predict(tts[1], beta)
    MSE_train_OLSeta[i] = mse(tts[2],ytilde)
    MSE_test_OLSeta[i] = mse(tts[3],ypredict)
    print(f"{int((i+1)/len(etaL)*100)} % done")
plotmseLR(MSE_test_OLSeta, etaL)
"""
#Ridge
MSE_train_Ridge = np.zeros((len(epochsL), len(nbatches)))
MSE_test_Ridge = np.zeros((len(epochsL), len(nbatches)))
print("Start epoch loop Ridge")
for i in range(len(epochsL)):
    for j in range(len(nbatches)):
        beta = SGD(tts[0], tts[2], M=int((N/nbatches[j])), epochs = epochsL[i], beta = theta, gradCostFunc = gradCostRidge, eta = 0.1, lmb=0.001)
        ytilde = predict(tts[0],beta)
        ypredict = predict(tts[1], beta)
        MSE_train_Ridge[i, j] = mse(tts[2],ytilde)
        MSE_test_Ridge[i, j] = mse(tts[3],ypredict)

    print(f"{int((i+1)/len(epochsL)*100)} % done")

minMSE_Ridge_train_index = np.argwhere(MSE_train_Ridge == np.min(MSE_train_Ridge))[0]
minMSE_Ridge_test_index = np.argwhere(MSE_train_Ridge == np.min(MSE_train_Ridge))[0]

optimal_epoch_train = epochsL[minMSE_Ridge_train_index[0]]
optimal_epoch_test = epochsL[minMSE_Ridge_test_index[0]]
optimal_batch_train = nbatches[minMSE_Ridge_train_index[1]]
optimal_batch_test = nbatches[minMSE_Ridge_test_index[1]]
print("Results for Ridge")
print(f"Optimal epoch train {optimal_epoch_train}")
print(f"Optimal epoch test {optimal_epoch_test}")
print(f"Optimal mini batches train {optimal_batch_train}")
print(f"Optimal mini batches test {optimal_batch_test}")

etaLRid = np.logspace(-2,-0.8,20)
lambdaL = np.logspace(-4,-1,20)
MSE_train_REL = np.zeros((len(etaLRid), len(lambdaL))) #Ridge Eta Lambda
MSE_test_REL = np.zeros((len(etaLRid), len(lambdaL)))

print("Start learning rate loop Ridge")

for i in range(len(etaLRid)):
    for j in range(len(lambdaL)):
        beta = SGD(tts[0], tts[2], M=int((N/optimal_batch_test)), epochs = optimal_epoch_test, beta = theta, gradCostFunc = gradCostRidge, eta = etaLRid[i], lmb = lambdaL[i])
        ytilde = predict(tts[0],beta)
        ypredict = predict(tts[1], beta)
        MSE_train_REL[i,j] = mse(tts[2],ytilde)
        MSE_test_REL[i,j] = mse(tts[3],ypredict)
    print(f"{int((i+1)/len(etaLRid)*100)} % done")


plotmseREL(MSE_test_REL, etaLRid, lambdaL)
