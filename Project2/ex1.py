import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import importlib
import functions
importlib.reload(functions)
from functions import *

#np.random.seed(2405) # Set a random seed

N = 500 #number of datapoints
d = 5 #complexity
x1 = np.random.uniform(0,1,N)
x2 = np.random.uniform(0,1,N)
y = FrankeFunction(x1,x2)
X = createX(x1,x2,d)

tts = train_test_split(X,y,test_size=0.2) #Split the data into training and test sets

theta = np.random.randn(len(X[0])) #first guess of beta
betaOLS1 = SGD(tts[0], tts[2], M=100, n_epochs = 10000, beta = theta, gradCostFunc = gradCostOls, eta = 0.1)
betaOLS2 = ols(tts[0], tts[2])

lmb = 0.0004
betaRidge1 = SGD(tts[0], tts[2], M=100, n_epochs = 10000, beta = theta, gradCostFunc = gradCostRidge, eta = 0.1, lmb = lmb)
betaRidge2 = ridge(tts[0], tts[2], lmb)

y_predictOLS1 = predict(tts[1],betaOLS1)
y_predictOLS2 = predict(tts[1],betaOLS2)

y_predictRidge1 = predict(tts[1],betaRidge1)
y_predictRidge2 = predict(tts[1],betaRidge2)

#y_predictRidge = predict(tts[0],beta2)
print(mse(tts[3],y_predictOLS1), mse(tts[3],y_predictOLS2))
print(mse(tts[3],y_predictRidge1), mse(tts[3],y_predictRidge2))
