import numpy as np
from sklearn.model_selection import train_test_split
import random
import importlib
import functions
importlib.reload(functions)
from functions import *
"""
Exercise 2:
Running OLS on the Franke Function as a function of the complexity of the model and using bootstrap as the resampling method
-->Plot mean MSE as function of complexity
Running OLS on the Franke Function for a given complexity and as a function of the size of the data set (number of datapoints)
-->Plot mean MSE as a function of datapoints
"""

np.random.seed(2405)

N = 500 #Number of datapoints
x = np.random.uniform(0, 1, N) #Randomly creates N x's
y = np.random.uniform(0, 1, N) #Randomly creates N y's
z = FrankeFunction(x, y) #Find the true FrankeFunction values to the given x and y
complex = 11 #complexity of model
X = create_X(x,y,complex)
noise = np.random.normal(0, 1, size=(z.shape))
z_noisy = FrankeFunction(x, y) + noise*0.2

tts = train_test_split(X,z_noisy,test_size=0.1) #Train test split

n_bs = 500 #number of bootstrap cycles
mse_test = np.zeros((complex, n_bs)) #for storing bootstrap samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
mse_train = np.zeros((complex, n_bs))
r2_test = np.zeros((complex, n_bs))
r2_train = np.zeros((complex, n_bs))

#Bootstrap and plotting MSE vs complexity

for j in range(n_bs): #looping through bootstrap samples
    X_sample, z_sample = bootstrap(tts[0],tts[2]) #Find the sample
    tts2 = [X_sample, tts[1], z_sample, tts[3]]
    for i in range(complex): #looping through complexity of model
        mse_train[i,j], r2_train[i,j], mse_test[i,j], r2_test[i,j], beta_l = evaluate_method(ols, tts2, scale = False, d = i+1, return_beta = True)
        l = int((i+1+1)*(i+1+2)/2)
        z_predict = predict(tts2[1][:,:l], beta_l)
        bias_l[i,j] = bias(z_predict, tts2[3])
        variance_l[i,j] = variance(z_predict)

mean_mse_train = np.mean(mse_train, axis = 1) #calculating mean of MSE of all bootstrap samples
mean_mse_test = np.mean(mse_test, axis = 1)
mean_r2_train = np.mean(r2_train, axis = 1)
mean_r2_test = np.mean(r2_test, axis = 1)
mean_bias = np.mean(bias_l, axis = 1)
mean_variance = np.mean(variance_l, axis = 1)


plot_mse(mean_mse_train, mean_mse_test, method_header = "Bootstrap")

compl_optimal_index = np.argmin(mean_mse_test)
compl_optimal = compl_optimal_index + 1
ols_eval = evaluate_method(ols, tts, d = compl_optimal)
print(f"MSE for best OLS model: {ols_eval[2]:.5f}")



#Bootstrap and plot MSE vs # datapoints
n_points = np.arange(100,2001,50)

mse_test_n = np.zeros((len(n_points), n_bs)) #for storing bootstrap samples' MSE for varying sample size (rows:sample size, columns:bootstrap sample)
mse_train_n = np.zeros((len(n_points), n_bs))
r2_test_n = np.zeros((len(n_points), n_bs))
r2_train_n = np.zeros((len(n_points), n_bs))


for i in range(len(n_points)): #looping through different sample sizes
    x = np.random.uniform(0, 1, n_points[i]) #Select new datapoints for each n_points
    y = np.random.uniform(0, 1, n_points[i])
    noise = np.random.normal(0, 1, size=(x.shape))
    X_data = create_X(x,y,4)
    z_data = FrankeFunction(x, y) + noise*0.2
    for j in range(n_bs): #looping through different bootstrap cycles
        X_sample, z_sample = bootstrap(X_data,z_data)
        tts = train_test_split(X_sample,z_sample,test_size=0.2)
        mse_train_n[i,j], r2_train_n[i,j], mse_test_n[i,j], r2_test_n[i,j] = evaluate_method(ols, tts, scale = False, d = 4)


mean_mse_train_n = np.mean(mse_train_n, axis = 1) #calculating mean of MSE of all bootstrap samples
mean_mse_test_n = np.mean(mse_test_n, axis = 1)
mean_r2_train_n = np.mean(r2_train_n, axis = 1)
mean_r2_test_n = np.mean(r2_test_n, axis = 1)

plot_mse(mean_mse_train_n, mean_mse_test_n, method_header = "Bootstrap", plot_complexity = False, complexities = n_points)
