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

terrain1 = imread('SRTM_data_Norway_1.tif')
terrain1 = terrain1[:200, :200]
x, y = np.meshgrid(range(terrain1.shape[1]), range(terrain1.shape[0]))
max_y = np.max(y)
x = x / max_y
y = y / max_y
z_terrain = terrain1.flatten().astype(np.float)

complexity = 20

X = create_X(x.flatten(),y.flatten(), complexity)
tts = train_test_split(X, z_terrain, test_size = 0.01)
print(f"OLS terrain: {evaluate_method(ols, tts, scale = True, d = complexity)}")


#print(x.shape)
#print(y.shape)
#print(terrain1)
#print(terrain1.shape)





# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x*max_y, y*max_y, terrain1)

plt.show()

"""
#OLS
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
a,b,c,d,beta_terrain = evaluate_method(ols, tts, scale = True, d = complexity, return_beta = True)
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
#Ridge
compl = [15,17,19,21,23,25]
nlambda = 10
lambda_values = np.logspace(-9,-7.5,nlambda)
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
        tts, lmb = lambda_values[j], d=compl[i], scale = True)
"""

#plot_mse(mse_train_ridge, mse_test_ridge, method_header = "Ridge_Terrain", lambdas = lambda_values, plot_complexity = True, complexities = compl)
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
