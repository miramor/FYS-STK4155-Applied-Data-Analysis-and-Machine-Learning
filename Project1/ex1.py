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
# Make data.
#x = np.arange(0, 1, 0.001)
#y = np.arange(0, 1, 0.001)

#x1, y1 = np(x,y)
z = FrankeFunction(x, y)
#z = FrankeFunction(x, y)
complex = 13 #complexity of model
X = create_X(x,y,complex)

test_train_l = train_test_split(X,z,test_size=0.2)
#Exercise 1
print(f"OLS: {evaluate_method(ols, test_train_l, scale = False, d = 5)}")

noise = np.random.normal(0, 1, size=(z.shape))
z_noisy = FrankeFunction(x, y) + noise*0.2
test_train_l_noise = train_test_split(X,z_noisy,test_size=0.2)
print(f"OLS with noise: {evaluate_method(ols, test_train_l_noise, scale = False, d = 5)}")
variance_beta = var_beta(test_train_l_noise[0])
beta_l = ols(test_train_l_noise[0], test_train_l_noise[2])
confidence_interval = ci(beta_l, variance_beta, N)
