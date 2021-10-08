import numpy as np
from sklearn.model_selection import train_test_split
import random
from imageio import imread
import matplotlib.pyplot as plt
import importlib
import functions
importlib.reload(functions)
from functions import *

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



complexities = [5,10,15,20,25,30,35]
n_bs = 100 #number of bootstrap cycles
mse_test = np.zeros((len(complexities), n_bs)) #for storing bootstrap samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
mse_train = np.zeros((len(complexities), n_bs))
r2_test = np.zeros((len(complexities), n_bs))
r2_train = np.zeros((len(complexities), n_bs))



#Bootstrap and plotting MSE vs complexity

complexities = [5,10,15,20,25,30,35,40,45,50]
mse_test = np.zeros(len(complexities))
mse_train = np.zeros(len(complexities))
r2_test = np.zeros(len(complexities))
r2_train = np.zeros(len(complexities))
for i in range(len(complexities)): #looping through complexity of model
    mse_train[i], r2_train[i], mse_test[i], r2_test[i] = evaluate_method(ols, tts, scale = False, d = complexities[i])


plt.plot(complexities, mse_train)
plt.plot(complexities, mse_test)
plt.xlabel("Complexity", fontsize = labelsize)
plt.ylabel("MSE", fontsize = labelsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.title("Analysis of complexity OLS", fontsize = labelsize)
plt.grid()
plt.show()


#OLS
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
a,b,c,d,beta_terrain = evaluate_method(ols, tts, scale = False, d = 25, return_beta = True)
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

ols_eval = evaluate_method(ols, tts, scale = False, d = 25)



#Ridge
compl = [15,17,19,21,23,25]
nlambda = 20
lambda_values = np.logspace(-11,-9,nlambda)
mse_test_ridge = np.zeros((len(compl), len(lambda_values)))
mse_train_ridge = np.zeros((len(compl), len(lambda_values)))
r2_test_ridge = np.zeros((len(compl), len(lambda_values)))
r2_train_ridge = np.zeros((len(compl), len(lambda_values)))
"""
for j in range(n_bs): #looping through bootstrap samples
    X_sample, z_sample = bootstrap(tts[0],tts[2])
    tts2 = [X_sample, tts[1], z_sample, tts[3]]
    for i in range(complex): #looping through complexity of model
for i in range(len(compl)):
    for j in range(len(lambda_values)):
        mse_train_ridge[i,j], r2_train_ridge[i,j], mse_test_ridge[i,j], r2_test_ridge[i,j] = evaluate_method(ridge,
        tts, lmb = lambda_values[j], d=compl[i], scale = False)


plot_mse(mse_train_ridge, mse_test_ridge, method_header = "ridge_terrain", lambdas = lambda_values, plot_complexity = True, complexities = compl)
min_mse_index = np.argmin(mse_test_ridge[-1])
lmb_optimal = lambda_values[min_mse_index]
print(f"Optimal lambda: {lmb_optimal}")
ridge_eval = evaluate_method(ridge, tts, scale = False, d = 25, lmb = lmb_optimal)
print(f"MSE for best OLS model: {ols_eval[2]:.5f}")
print(f"MSE for best ridge model: {ridge_eval[2]:.5f}")

#print(f"Ridge: {evaluate_method(ridge, test_train_l, scale = True, d = 5)}")
"""
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
_,_,_,_,beta_terrain = evaluate_method(ridge, tts, scale = True, d = complexity, lmb = lambda_values[] return_beta = True)
z_terrain = predict(X, beta_terrain).reshape(200,200)
ax.plot_surface(x*max_y, y*max_y, z_terrain)
plt.show()

plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(z_terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""

#Lasso
lambda_values = np.logspace(-6,-2,50)
mse_test_lasso = np.zeros((len(compl), len(lambda_values)))
mse_train_lasso = np.zeros((len(compl), len(lambda_values)))
r2_test_lasso = np.zeros((len(compl), len(lambda_values)))
r2_train_lasso = np.zeros((len(compl), len(lambda_values)))

for i in range(len(compl)):
    for j in range(len(lambda_values)):
        mse_train_lasso[i,j], r2_train_lasso[i,j], mse_test_lasso[i,j], r2_test_lasso[i,j] = evaluate_method(lasso,
        tts, lmb = lambda_values[j], d=compl[i], scale = False)


plot_mse(mse_train_lasso, mse_test_lasso, method_header = 'Lasso_Terrain',
    plot_complexity = True, lambdas = lambda_values, complexities = compl)


"""
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
a,b,c,d,beta_terrain = evaluate_method(lasso, train_test_terrain, scale = True, d = complexity, return_beta = True)
z_terrain = predict(X, beta_terrain).reshape(200,200)
ax.plot_surface(x*max_y, y*max_y, z_terrain)
plt.show()

plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(z_terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""
