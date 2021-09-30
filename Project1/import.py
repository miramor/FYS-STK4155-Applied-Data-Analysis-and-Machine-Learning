from functions import FrankeFunction
import numpy as np
import importlib
import test_func
importlib.reload(test_func)
from test_func import a


N = 500
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
# Make data.
#x = np.arange(0, 1, 0.001)
#y = np.arange(0, 1, 0.001)

#x1, y1 = np(x,y)
print(a(2))
