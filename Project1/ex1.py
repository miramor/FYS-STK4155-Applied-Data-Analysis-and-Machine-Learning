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



def evaluate_method(method, train_test_l, scale, d):
    X_train, X_test, z_train, z_test = train_test_l
    l = int((d+1)*(d+2)/2)
    X_train = X_train[:,:l]
    X_test = X_test[:,:l]

    if scale: #scala data
        scaler = StandardScaler()
        #scaler = MaxAbsScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler2 = StandardScaler()
        #scaler2 = MaxAbsScaler()
        scaler2.fit(z_train.reshape(-1,1))
        z_train = scaler2.transform(z_train.reshape(-1,1))
        z_test = scaler2.transform(z_test.reshape(-1,1))
        # scaler2.fit(z_train)
        # z_train = scaler2.transform(z_train)
        # z_test = scaler2.transform(z_test)


    beta = method(X_train,z_train)
    #print(beta.shape)
    z_tilde = predict(X_train, beta)
    z_predict = predict(X_test, beta)

    mse_tilde = mse(z_train, z_tilde)
    mse_predict= mse(z_test, z_predict)

    r2_tilde = r2(z_train, z_tilde)
    r2_predict = r2(z_test, z_predict)

    return mse_tilde, r2_tilde, mse_predict, r2_predict

def bootstrap(X,z):
    n = len(z)
    data = np.random.randint(0,n,n)
    X_new = X[data] #random chosen columns for new design matrix
    z_new = z[data]
    return X_new, z_new





N = 1000
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
# Make data.
#x = np.arange(0, 1, 0.001)
#y = np.arange(0, 1, 0.001)

#x1, y1 = np.meshgrid(x,y)
z = FrankeFunction(x, y)
#z = FrankeFunction(x, y)
complex = 15 #complexity of model
X = create_X(x,y,complex)

test_train_l = train_test_split(X,z,test_size=0.2)
#Exercise 1
print(z.shape)
print(f"OLS: {evaluate_method(ols, test_train_l, scale = False, d = 5)}")

noise = np.random.normal(0, 1, size=(z.shape))
z_noisy = FrankeFunction(x, y) + noise*0.2
test_train_l_noise = train_test_split(X,z_noisy,test_size=0.2)
print(f"OLS with noise: {evaluate_method(ols, test_train_l_noise, scale = False, d = 5)}")

#Exercise 2
n_bs = 300 #number of bootstrap cycles
mse_test = np.zeros((complex, n_bs)) #for storing bootstrap samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
mse_train = np.zeros((complex, n_bs))
r2_test = np.zeros((complex, n_bs))
r2_train = np.zeros((complex, n_bs))

#Bootstrap and plotting MSE vs complexity
for j in range(n_bs): #looping through bootstrap samples
    X_sample, z_sample = bootstrap(X,z_noisy)
    tts = train_test_split(X_sample,z_sample,test_size=0.2)
    for i in range(complex): #looping through complexity of model
        mse_train[i,j], r2_train[i,j], mse_test[i,j], r2_test[i,j] = evaluate_method(ols, tts, scale = True, d = i+1)

mean_mse_train = np.mean(mse_train, axis = 1) #calculating mean of MSE of all bootstrap samples
mean_mse_test = np.mean(mse_test, axis = 1)
mean_r2_train = np.mean(r2_train, axis = 1)
mean_r2_test = np.mean(r2_test, axis = 1)

labelsize=18
plt.plot(range(1,complex+1), mean_mse_train, label="MSE train")
plt.plot(range(1,complex+1), mean_mse_test, label="MSE test")
plt.legend(fontsize=labelsize)
plt.ylabel("MSE", fontsize=labelsize)
plt.xlabel("Complexity", fontsize=labelsize)
plt.title("Mean Squared Error", fontsize=labelsize)
plt.savefig("MSE_complex.png")
plt.show()
"""
#Bootstrap and plot MSE vs # datapoints
n_points = np.arange(100,10001,100)

mse_test_n = np.zeros((len(n_points), n_bs)) #for storing bootstrap samples' MSE for varying sample size (rows:sample size, columns:bootstrap sample)
mse_train_n = np.zeros((len(n_points), n_bs))
r2_test_n = np.zeros((len(n_points), n_bs))
r2_train_n = np.zeros((len(n_points), n_bs))


for i in range(len(n_points)): #looping through different sample sizes
    X_data = X[:n_points[i]]
    z_data = z_noisy[:n_points[i]]
    X_sample, z_sample = bootstrap(X_data,z_data)
    tts = train_test_split(X_sample,z_sample,test_size=0.2)
    for j in range(n_bs): #looping through different bootstrap cycles
        mse_train_n[i,j], r2_train_n[i,j], mse_test_n[i,j], r2_test_n[i,j] = evaluate_method(ols, tts, scale = False, d = 4)


mean_mse_train_n = np.mean(mse_train_n, axis = 1) #calculating mean of MSE of all bootstrap samples
mean_mse_test_n = np.mean(mse_test_n, axis = 1)
mean_r2_train_n = np.mean(r2_train_n, axis = 1)
mean_r2_test_n = np.mean(r2_test_n, axis = 1)

print(mean_mse_test_n)
labelsize=18
plt.plot(n_points, mean_mse_train_n, label="MSE train")
plt.plot(n_points, mean_mse_test_n, label="MSE test")
plt.legend(fontsize=labelsize)
plt.ylabel("MSE", fontsize=labelsize)
plt.xlabel("# Datapoints", fontsize=labelsize)
plt.title("Mean Squared Error", fontsize=labelsize)
plt.savefig("MSE_datapoints.png")
plt.show()
"""

#Exercise 3, K-fold

def kfold(X,z,k):
    #Exercise 2
    mse_test = np.zeros((complex, k)) #for storing bootstrap samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
    mse_train = np.zeros((complex, k))
    r2_test = np.zeros((complex, k))
    r2_train = np.zeros((complex, k))

    n = len(X)
    split = int(n/k)
    for j in range(k): #looping through bootstrap samples
        X_test = X[j*split:(j+1)*split]
        X_train = X[-j*split:(j+1)*split]
        for i in range(complex): #looping through complexity of model
            mse_train[i,j], r2_train[i,j], mse_test[i,j], r2_test[i,j] = evaluate_method(ols, tts, scale = True, d = i+1)

    mean_mse_train = np.mean(mse_train, axis = 1) #calculating mean of MSE of all bootstrap samples
    mean_mse_test = np.mean(mse_test, axis = 1)
    mean_r2_train = np.mean(r2_train, axis = 1)
    mean_r2_test = np.mean(r2_test, axis = 1)
