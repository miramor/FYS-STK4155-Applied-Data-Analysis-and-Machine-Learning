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

compl = [3,4,5,6,7,8,9,10]
nlambda = 20
lambda_values = np.logspace(-5,-3,nlambda) #[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1]
mse_test_lasso = np.zeros((len(compl), len(lambda_values)))
mse_train_lasso = np.zeros((len(compl), len(lambda_values)))
r2_test_lasso = np.zeros((len(compl), len(lambda_values)))
r2_train_lasso = np.zeros((len(compl), len(lambda_values)))




for i in range(len(compl)):
    for j in range(len(lambda_values)):
        mse_train_lasso[i,j], r2_train_lasso[i,j], mse_test_lasso[i,j], r2_test_lasso[i,j] = evaluate_method(lasso,
        tts, lmb = lambda_values[j], d=compl[i], scale = False)


min_mse_index = np.argmin(mse_test_lasso[-1])
lmb_optimal = lambda_values[min_mse_index]

#Bias variance trade-off:
n_bs = 100

mse_test = np.zeros((complex, n_bs)) #for storing bootstrap samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
mse_train = np.zeros((complex, n_bs))
r2_test = np.zeros((complex, n_bs))
r2_train = np.zeros((complex, n_bs))

for j in range(n_bs): #looping through bootstrap samples
    X_sample, z_sample = bootstrap(tts[0],tts[2])
    tts2 = [X_sample, tts[1], z_sample, tts[3]]
    for i in range(complex): #looping through complexity of model
        mse_train[i,j], r2_train[i,j], mse_test[i,j], r2_test[i,j] = evaluate_method(lasso, tts2, scale = False, d = i+1, lmb = lmb_optimal)

mean_mse_train = np.mean(mse_train, axis = 1) #calculating mean of MSE of all bootstrap samples
mean_mse_test = np.mean(mse_test, axis = 1)
mean_r2_train = np.mean(r2_train, axis = 1)
mean_r2_test = np.mean(r2_test, axis = 1)

plot_mse(mse_train_lasso, mse_test_lasso, method_header = 'Lasso', plot_complexity = True, lambdas = lambda_values, complexities = compl)

plot_mse(mean_mse_train, mean_mse_test, method_header = "Bootstrap lasso")
