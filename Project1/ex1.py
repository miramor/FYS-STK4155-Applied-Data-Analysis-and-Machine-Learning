import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import random
np.random.seed(2405)
def mse(z, z_model):
    n = len(z)
    mean_se = np.sum((z-z_model)**2)
    return mean_se/n

def r2(z, z_model):
    n = len(y)
    return 1 - n*mse(z,z_model)/np.sum((z-np.mean(z))**2)

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



def evaluate_method(method, scale):
    #X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
    global X_train, X_test, z_train, z_test

    if scale: #scala data
        scaler = StandardScaler()
        #scaler = MaxAbsScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler2 = StandardScaler()
        #scaler2 = MaxAbsScaler()
        scaler2.fit(z_train)#.reshape(-1,1))
        z_train = scaler2.transform(z_train)#.reshape(-1,1))
        z_test = scaler2.transform(z_test)#.reshape(-1,1))
        # scaler2.fit(z_train)
        # z_train = scaler2.transform(z_train)
        # z_test = scaler2.transform(z_test)


    beta = method(X_train,z_train)
    print(beta.shape)
    z_tilde = predict(X_train, beta)
    z_predict = predict(X_test, beta)

    mse_tilde = mse(z_train, z_tilde)
    mse_predict= mse(z_test, z_predict)

    r2_tilde = r2(z_train, z_tilde)
    r2_predict = r2(z_test, z_predict)

    return mse_tilde, r2_tilde, mse_predict, r2_predict

N = 1000
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
# Make data.
#x = np.arange(0, 1, 0.001)
#y = np.arange(0, 1, 0.001)

#x1, y1 = np.meshgrid(x,y)
z = FrankeFunction(x, y)
#z = FrankeFunction(x, y)
X = create_X(x,y,5)

X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)


print(z.shape)
print(f"OLS: {evaluate_method(ols, scale = False)}")

noise = np.random.normal(0, 1, size=(z.shape))
z_noisy = FrankeFunction(x, y) + noise*0.1
print(f"OLS with noise: {evaluate_method(ols, scale = False)}")

# Task 2
complex = 15 #complexity of model
mse_test = np.zeros(complex)
mse_train = np.zeros(complex)
r2_test = np.zeros(complex)
r2_train = np.zeros(complex)



for i in range(complex):
    mse_train[i], r2_train[i], mse_test[i], r2_test[i] = evaluate_method(ols, scale = False)

labelsize=18
plt.plot(range(complex), mse_train, label="MSE train")
plt.plot(range(complex), mse_test, label="MSE test")
plt.legend()
plt.ylabel("MSE", fontsize=labelsize)
plt.xlabel("Complexity", fontsize=labelsize)
plt.title("Mean Squared Error", fontsize=labelsize)
plt.savefig("MSE_complex.png")
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
