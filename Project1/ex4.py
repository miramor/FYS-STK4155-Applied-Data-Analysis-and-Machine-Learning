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
complex = 10 #complexity of model
X = create_X(x,y,complex)
noise = np.random.normal(0, 1, size=(z.shape))
z_noisy = FrankeFunction(x, y) + noise*0.2

tts = train_test_split(X,z_noisy,test_size=0.2) #Train test split

compl = [3,4,5,6,7,8, 9, 10]
nlambda = 200
lambda_values = np.logspace(-10,-0.5,nlambda)
print(lambda_values)
mse_test_ridge = np.zeros((len(compl), len(lambda_values)))
mse_train_ridge = np.zeros((len(compl), len(lambda_values)))
r2_test_ridge = np.zeros((len(compl), len(lambda_values)))
r2_train_ridge = np.zeros((len(compl), len(lambda_values)))

for i in range(len(compl)):
    for j in range(len(lambda_values)):
        mse_train_ridge[i,j], r2_train_ridge[i,j], mse_test_ridge[i,j], r2_test_ridge[i,j] = evaluate_method(ridge,
        tts, lmb = lambda_values[j], d=compl[i])


plot_mse(mse_train_ridge, mse_test_ridge, method_header = "ridge", lambdas = lambda_values, plot_complexity = True, complexities = compl)


#Bias variance trade-off:
n_bs = 100

mse_test = np.zeros((complex-2, n_bs)) #for storing bootstrap samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
mse_train = np.zeros((complex-2, n_bs))
r2_test = np.zeros((complex-2, n_bs))
r2_train = np.zeros((complex-2, n_bs))

for j in range(n_bs): #looping through bootstrap samples
    X_sample, z_sample = bootstrap(tts[0],tts[2])
    tts2 = [X_sample, tts[1], z_sample, tts[3]]
    for i in range(complex-2): #looping through complexity of model
        min_mse_index = np.argmin(mse_test_ridge[i])
        lmb_optimal = lambda_values[min_mse_index]
        mse_train[i,j], r2_train[i,j], mse_test[i,j], r2_test[i,j] = evaluate_method(ridge, tts2, d = i+3, lmb = lmb_optimal)

mean_mse_train = np.mean(mse_train, axis = 1) #calculating mean of MSE of all bootstrap samples
mean_mse_test = np.mean(mse_test, axis = 1)
mean_r2_train = np.mean(r2_train, axis = 1)
mean_r2_test = np.mean(r2_test, axis = 1)

plot_mse(mean_mse_train, mean_mse_test, method_header = "Bootstrap ridge", plot_complexity = True, complexities = compl)

min_mse_index = np.argmin(mse_test_ridge[-1])
lmb_optimal = lambda_values[min_mse_index]
ridge_eval = evaluate_method(ridge, tts, d = 4, lmb = lmb_optimal)
print(f"MSE for best ridge model: {ridge_eval[2]:.5f}")
