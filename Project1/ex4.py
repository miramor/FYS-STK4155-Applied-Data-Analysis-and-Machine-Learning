import numpy as np
from sklearn.model_selection import train_test_split
import random
import importlib
import functions
importlib.reload(functions)
from functions import *

np.random.seed(2405)

N = 500
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)

z = FrankeFunction(x, y)
complex = 13 #complexity of model
X = create_X(x,y,complex)
noise = np.random.normal(0, 1, size=(z.shape))
z_noisy = FrankeFunction(x, y) + noise*0.2

tts = train_test_split(X,z_noisy,test_size=0.2) #Train test split

compl = [3,4,5,6,7,8, 9, 10]
nlambda = 100
lambda_values = np.logspace(-6,-2,nlambda)
print(lambda_values)
mse_test_ridge = np.zeros((len(compl), len(lambda_values)))
mse_train_ridge = np.zeros((len(compl), len(lambda_values)))
r2_test_ridge = np.zeros((len(compl), len(lambda_values)))
r2_train_ridge = np.zeros((len(compl), len(lambda_values)))
"""
for j in range(n_bs): #looping through bootstrap samples
    X_sample, z_sample = bootstrap(tts[0],tts[2])
    tts2 = [X_sample, tts[1], z_sample, tts[3]]
    for i in range(complex): #looping through complexity of model
"""
for i in range(len(compl)):
    for j in range(len(lambda_values)):
        mse_train_ridge[i,j], r2_train_ridge[i,j], mse_test_ridge[i,j], r2_test_ridge[i,j] = evaluate_method(ridge,
        tts, lmb = lambda_values[j], d=compl[i], scale = True)


plot_mse(mse_train_ridge, mse_test_ridge, method_header = "Ridge", lambdas = lambda_values, plot_complexity = True, complexities = compl)
