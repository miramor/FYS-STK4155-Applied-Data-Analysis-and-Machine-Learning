import numpy as np
from sklearn.model_selection import train_test_split
import random
from imageio import imread
import matplotlib.pyplot as plt
import importlib
import functions
importlib.reload(functions)
from functions import *
"""
Exercise 6:
Running OLS, Ridge and Lasso on real terrain data
Finding mean MSE using bootstrap
"""

np.random.seed(2405)
labelsize=21
ticksize = 19

terrain1 = imread('SRTM_data_Norway_1.tif')
terrain1 = terrain1[:200, :200]
x, y = np.meshgrid(range(terrain1.shape[1]), range(terrain1.shape[0]))
max_y = np.max(y)
x = x / max_y
y = y / max_y
z_terrain = terrain1.flatten().astype(np.float)

complexity = 50

X = create_X(x.flatten(),y.flatten(), complexity)
tts = train_test_split(X, z_terrain, test_size = 0.2)


#print(x.shape)
#print(y.shape)
#print(terrain1)
#print(terrain1.shape)





# Show the terrain
plt.figure()
plt.title('Top view of real Terrain data', fontsize = labelsize)
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X', fontsize = labelsize)
plt.ylabel('Y', fontsize = labelsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x*max_y, y*max_y, terrain1)
ax.set_xlabel("x", fontsize = labelsize)
ax.set_ylabel("y", fontsize = labelsize)
ax.set_zlabel('Altitude', fontsize = labelsize, rotation = 60)
ax.xaxis.set_tick_params(labelsize= ticksize, pad=-5)
ax.yaxis.set_tick_params(labelsize= ticksize, pad=-5)
ax.zaxis.set_tick_params(labelsize= ticksize, pad=-5)
ax.set_title("Real Terrain data", fontsize = labelsize)
plt.show()


#Bootstrap and plotting MSE vs complexity

#OLS
n_bs = 5
complexities = [5,10,15,20,25,30,35]
mse_test = np.zeros((len(complexities), n_bs)) #for storing bootstrap evaluations
mse_train = np.zeros((len(complexities), n_bs))
r2_test = np.zeros((len(complexities), n_bs))
r2_train = np.zeros((len(complexities), n_bs))

for j in range(n_bs):
    X_sample, z_sample = bootstrap(tts[0],tts[2])
    tts2 = [X_sample, tts[1], z_sample, tts[3]]
    for i in range(len(complexities)): #looping through complexity of model
        mse_train[i], r2_train[i], mse_test[i], r2_test[i] = evaluate_method(ols, tts2, scale = False, d = complexities[i])


mean_mse_train = np.mean(mse_train, axis = 1) #calculating mean of MSE of all bootstrap samples
mean_mse_test = np.mean(mse_test, axis = 1)

compl_optimal_index = np.argmin(mean_mse_test)
compl_optimal_ols = complexities[compl_optimal_index]
ols_eval = evaluate_method(ols, tts, d = compl_optimal_ols)
mse_ols = ols_eval[2]
print(f"Optimal complexity for OLS: {compl_optimal_ols}")
print(f"MSE for best OLS model: {mse_ols:.5f}")



#Ridge
nlambda = 20
lambda_values = np.logspace(-11,-9,nlambda)
mse_test_ridge = np.zeros((len(complexities), len(lambda_values)))
mse_train_ridge = np.zeros((len(complexities), len(lambda_values)))
r2_test_ridge = np.zeros((len(complexities), len(lambda_values)))
r2_train_ridge = np.zeros((len(complexities), len(lambda_values)))

for i in range(len(complexities)):
    for j in range(len(lambda_values)):
        mse_train_ridge[i,j], r2_train_ridge[i,j], mse_test_ridge[i,j], r2_test_ridge[i,j] = evaluate_method(ridge,
        tts, lmb = lambda_values[j], d=complexities[i])


mse_test = np.zeros((len(complexities), n_bs)) #for storing bootstrap samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
mse_train = np.zeros((len(complexities), n_bs))
r2_test = np.zeros((len(complexities), n_bs))
r2_train = np.zeros((len(complexities), n_bs))

for j in range(n_bs): #looping through bootstrap samples
    X_sample, z_sample = bootstrap(tts[0],tts[2])
    tts2 = [X_sample, tts[1], z_sample, tts[3]]
    for i in range(len(complexities)): #looping through complexity of model
        min_mse_index = np.argmin(mse_test_ridge[i])
        lmb_optimal = lambda_values[min_mse_index]
        mse_train[i,j], r2_train[i,j], mse_test[i,j], r2_test[i,j] = evaluate_method(ridge, tts2, d = complexities[i], lmb = lmb_optimal)

mean_mse_train = np.mean(mse_train, axis = 1) #calculating mean of MSE of all bootstrap samples
mean_mse_test = np.mean(mse_test, axis = 1)
mean_r2_train = np.mean(r2_train, axis = 1)
mean_r2_test = np.mean(r2_test, axis = 1)




compl_optimal_index = np.argmin(mean_mse_test)
min_mse_index = np.argmin(mse_test_ridge[compl_optimal_index])
compl_optimal_ridge =complexities[compl_optimal_index]
lmb_optimal_ridge = lambda_values[min_mse_index]
ridge_eval = evaluate_method(ridge, tts, d = compl_optimal_ridge, lmb = lmb_optimal_ridge)
mse_ridge = ridge_eval[2]
print(f"Optimal lambda for ridge: {lmb_optimal_ridge} | Optimal complexity: {compl_optimal_ridge}")
print(f"MSE for best ridge model: {mse_ridge:.5f}")

#Lasso
nlambda = 20
lambda_values = np.logspace(-10,-6,nlambda)
mse_test_lasso = np.zeros(len(lambda_values))
mse_train_lasso = np.zeros(len(lambda_values))
r2_test_lasso = np.zeros(len(lambda_values))
r2_train_lasso = np.zeros(len(lambda_values))

for j in range(len(lambda_values)):
    mse_train_lasso[j], r2_train_lasso[j], mse_test_lasso[j], r2_test_lasso[j] = evaluate_method(lasso,
    tts, lmb = lambda_values[j], d=20)


mse_test = np.zeros(n_bs) #for storing bootstrap samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample
mse_train = np.zeros(n_bs)
r2_test = np.zeros(n_bs)
r2_train = np.zeros(n_bs)

for j in range(n_bs): #looping through bootstrap samples
    X_sample, z_sample = bootstrap(tts[0],tts[2])
    tts2 = [X_sample, tts[1], z_sample, tts[3]]
    min_mse_index = np.argmin(mse_test_lasso)
    lmb_optimal = lambda_values[min_mse_index]
    mse_train[j], r2_train[j], mse_test[j], r2_test[j] = evaluate_method(lasso, tts2, d = 20, lmb = lmb_optimal)

mean_mse_train = np.mean(mse_train) #calculating mean of MSE of all bootstrap samples
mean_mse_test = np.mean(mse_test)
mean_r2_train = np.mean(r2_train)
mean_r2_test = np.mean(r2_test)




min_mse_index = np.argmin(mse_test_lasso)
lmb_optimal_lasso = lambda_values[min_mse_index]
lasso_eval = evaluate_method(lasso, tts, d = 35, lmb = lmb_optimal_lasso)
mse_lasso = lasso_eval[2]
print(f"Optimal lambda for lasso: {lmb_optimal} | Optimal complexity: {35}")
print(f"MSE for best lasso model: {mse_lasso:.5f}")



#Selecting the best model:

#Plot predictions for the best model
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
a,b,c,d,beta_terrain = evaluate_method(best_method, tts, scale = False, d = compl_optimal_ridge, return_beta = True)
l = int((25+1)*(25+2)/2)
z_terrain = predict(X[:,:l], beta_terrain).reshape(200,200)
ax.plot_surface(x*max_y, y*max_y, z_terrain)
ax.set_xlabel("x", fontsize = labelsize)
ax.set_ylabel("y", fontsize = labelsize)
ax.set_zlabel('Altitude', fontsize = labelsize, rotation=60)
ax.xaxis.set_tick_params(labelsize= ticksize, pad=-5)
ax.yaxis.set_tick_params(labelsize= ticksize, pad=-5)
ax.zaxis.set_tick_params(labelsize= ticksize, pad=-5)
ax.set_title("Prediction of real Terrain data", fontsize = labelsize)
plt.show()

plt.figure()
plt.imshow(z_terrain, cmap='gray')
plt.title("Prediction of real Terrain data", fontsize = labelsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.xlabel('X', fontsize = labelsize)
plt.ylabel('Y', fontsize = labelsize)
plt.show()
