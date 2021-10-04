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
x, y = np.meshgrid(range(terrain1.shape[1]), range(terrain1.shape[0]))
z_terrain = terrain1.flatten().astype(np.float)
X = create_X(x.flatten(),y.flatten(), 5)
train_test_terrain = train_test_split(X, z_terrain, test_size = 0.2)
print(f"OLS terrain: {evaluate_method(ols, train_test_terrain, scale = True, d = 5)}")


print(x.shape)
print(y.shape)
print(terrain1)
print(terrain1.shape)
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
