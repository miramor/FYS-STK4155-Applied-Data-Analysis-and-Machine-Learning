import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Lasso
from sklearn import linear_model
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


def ols(X, z):
    return np.linalg.pinv(X.T @ X) @ X.T @ z

def ridge(X, z, lmb):
    return np.linalg.pinv(X.T @ X + lmb*np.identity(n)) @ X.T @ z

def lasso(X, z, lmb):
    reg = Lasso(alpha=lmb, fit_intercept=True)
    reg.fit(X, z)
    return np.array(reg.coef_)

def predict(X, beta):
    return X @ beta

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def kfold(X,z,k, plot = False):
    #Exercise 2
    mse_test = np.zeros((complex, k)) #for storing kfold samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
    mse_train = np.zeros((complex, k))
    r2_test = np.zeros((complex, k))
    r2_train = np.zeros((complex, k))
    n = len(X)
    split = int(n/k)
    for j in range(k):
        if j == k-1:
            X_train = X[:j*split]
            X_test = X[j*split:]
            z_train = z[:j*split]
            z_test = z[j*split:]
        else:
            X_train = np.concatenate((X[:(j)*split], X[(j+1)*split:]), axis = 0)
            X_test = X[j*split:(j+1)*split]
            z_train = np.concatenate((z[:(j)*split], z[(j+1)*split:]))
            z_test = z[j*split:(j+1)*split]
        tts = [X_train, X_test, z_train, z_test]
        for i in range(complex): #looping through complexity of model
            mse_train[i,j], r2_train[i,j], mse_test[i,j], r2_test[i,j] = evaluate_method(ols, tts, scale = True, d = i+1)

    mean_mse_train = np.mean(mse_train, axis = 1) #calculating mean of MSE of all kfold samples
    mean_mse_test = np.mean(mse_test, axis = 1)
    mean_r2_train = np.mean(r2_train, axis = 1)
    mean_r2_test = np.mean(r2_test, axis = 1)
    if plot:
        plot_mse(mean_mse_train, mean_mse_test, method_header = "k-fold")

def evaluate_method(method, train_test_l, d, scale = True, lmb = False):
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

    if lmb != False:
        beta = method(X_train, z_train, lmb)
    else:
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

def plot_mse(mse_train, mse_test, method_header = '', plot_complexity = True, lambdas = False, complexities = False):
    labelsize=18
    degree = mse_train.shape[0]

    if type(lambdas) != type(False):
        n_lmd = len(lambdas)
        for i in range(degree):
            plt.plot(lambdas, mse_test[i], label = f'Complexity: {complexities[i]}')

    else:
        if plot_complexity:
            plt.plot(range(1,degree+1), mse_train, label="MSE train")
            plt.plot(range(1,degree+1), mse_test, label="MSE test")
            plt.xlabel("Complexity", fontsize=labelsize)
        else:
            plt.plot(n_points, mse_train, label="MSE train")
            plt.plot(n_points, mse_test, label="MSE test")
            plt.xlabel("# Datapoints", fontsize=labelsize)
    plt.legend(fontsize=labelsize)
    plt.ylabel("MSE", fontsize=labelsize)
    plt.title(f"Mean Squared Error {method_header}", fontsize=labelsize)
    plt.savefig(f"MSE_datapoints_{method_header}.png")
    plt.legend()
    plt.show()



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

print(f"OLS: {evaluate_method(ols, test_train_l, scale = False, d = 5)}")

noise = np.random.normal(0, 1, size=(z.shape))
z_noisy = FrankeFunction(x, y) + noise*0.2
test_train_l_noise = train_test_split(X,z_noisy,test_size=0.2)
print(f"OLS with noise: {evaluate_method(ols, test_train_l_noise, scale = False, d = 5)}")

#Exercise 2
n_bs = 100 #number of bootstrap cycles
mse_test = np.zeros((complex, n_bs)) #for storing bootstrap samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
mse_train = np.zeros((complex, n_bs))
r2_test = np.zeros((complex, n_bs))
r2_train = np.zeros((complex, n_bs))

#Bootstrap and plotting MSE vs complexity

tts = train_test_split(X,z_noisy,test_size=0.2)
for j in range(n_bs): #looping through bootstrap samples
    X_sample, z_sample = bootstrap(tts[0],tts[2])
    tts2 = [X_sample, tts[1], z_sample, tts[3]]
    for i in range(complex): #looping through complexity of model
        mse_train[i,j], r2_train[i,j], mse_test[i,j], r2_test[i,j] = evaluate_method(ols, tts2, scale = True, d = i+1)

mean_mse_train = np.mean(mse_train, axis = 1) #calculating mean of MSE of all bootstrap samples
mean_mse_test = np.mean(mse_test, axis = 1)
mean_r2_train = np.mean(r2_train, axis = 1)
mean_r2_test = np.mean(r2_test, axis = 1)

plot_mse(mean_mse_train, mean_mse_test, method_header = "bootstrap")


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
        mse_train_n[i,j], r2_train_n[i,j], mse_test_n[i,j], r2_test_n[i,j] = evaluate_method(ols, tts, scale = True, d = 4)


mean_mse_train_n = np.mean(mse_train_n, axis = 1) #calculating mean of MSE of all bootstrap samples
mean_mse_test_n = np.mean(mse_test_n, axis = 1)
mean_r2_train_n = np.mean(r2_train_n, axis = 1)
mean_r2_test_n = np.mean(r2_test_n, axis = 1)

plot_mse(mean_mse_train_n, mean_mse_test_n, method_header = "bootstrap", plot_complexity = False)



#Exercise 3, K-fold
kfold(X, z_noisy, 5, plot = True)

nlambdas = 10
lambdas_values = [0.001,0.005,0.01,0.05,0.1,0.5]#np.logspace(-4,1,nlambdas)
print(lambdas_values)
compl = [2,3,4,5,6,7,8]
mse_test_lasso = np.zeros((len(compl), len(lambdas_values)))
mse_train_lasso = np.zeros((len(compl), len(lambdas_values)))
r2_test_lasso = np.zeros((len(compl), len(lambdas_values)))
r2_train_lasso = np.zeros((len(compl), len(lambdas_values)))

for i in range(len(compl)):
    for j in range(len(lambdas_values)):
        mse_train_lasso[i,j], mse_test_lasso[i,j], r2_train_lasso[i,j], r2_test_lasso[i,j] = evaluate_method(lasso, tts, lmb = lambdas_values[j], d=compl[i])
print(np.min(mse_test_lasso))
plot_mse(mse_train_lasso, mse_test_lasso, method_header = 'lasso', plot_complexity = True, lambdas = lambdas_values, complexities = compl)
