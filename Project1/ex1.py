import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import random

def mse(y, y_model):
    n = len(y)
    mean_se = np.sum((y-y_model)**2)
    return mean_se/n

def r2(y, y_model):
    n = len(y)
    return 1 - n*mse(y,y_model)/np.sum((y-np.mean(y))**2)

def create_X(x, y, n ):
    if len(x.shape) > 1:
    	x = np.ravel(x)
    	y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
    	q = int((i)*(i+1)/2)
    	for k in range(i+1):
    		X[:,q+k] = (x**(i-k))*(y**k)

    return X


def ols(X,z):
    return np.linalg.pinv(X.T @ X) @ X.T @ z

def predict(X, beta):
    return X @ beta

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def evaluate_method(method, X, z, scale):
    X = X[:,1:] #Remove first column
    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)

    if scale: #scala data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler2 = StandardScaler()
        scaler2.fit(z_train)
        z_train = scaler2.transform(z_train)
        z_test = scaler2.transform(z_test)


    beta = method(X_train,z_train)
    print(beta.shape)
    z_tilde = predict(X_train, beta)
    z_predict = predict(X_test, beta)

    mse_tilde = mse(z_train, z_tilde)
    mse_predict= mse(z_test, z_predict)

    r2_tilde = r2(z_train, z_tilde)
    r2_predict = r2(z_test, z_predict)

    return mse_tilde, r2_tilde, mse_predict, r2_predict


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x1, y1 = np.meshgrid(x,y)
z = FrankeFunction(x1, y1)
X = create_X(x,y,5)
print(z.shape)
print(evaluate_method(ols,X,z, scale = True))


plt.show()











"""
bootstraps = 10
mse_tilde = np.zeros(bootstraps)
r2_tilde = np.zeros(bootstraps)
mse_predict = np.zerosn(bootstraps)
r2_predict = np.zeros(bootstraps)


for i in range(bootstraps):
    mse_tilde[i], r2_tilde[i], mse_predict[i], r2_predict[i] = evaluate_method(ols,X,z)
"""
