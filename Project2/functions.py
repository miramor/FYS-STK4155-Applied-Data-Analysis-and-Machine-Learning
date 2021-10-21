import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns

plt.style.use("seaborn")
plt.rcParams["font.family"] = "Times New Roman"; plt.rcParams['axes.titlesize'] = 18; plt.rcParams['axes.labelsize'] = 18; plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 15


def SGD(X, y, M, epochs, gradCostFunc, beta, eta, lmb = None): #Stochastic Gradient Descent
    n = len(X) #number of datapoints
    m = int(n/M) #number of mini-batch cycles (M: size of batch)
    for epoch in range(epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index*M:(random_index+1)*M]
            yi = y[random_index*M:(random_index+1)*M]
            if lmb is None:
                gradients = gradCostFunc(xi, yi, beta)
            else:
                gradients = gradCostFunc(xi, yi, beta, lmb)
            #eta = learningSchedule(epoch*m+i)
            beta = beta - eta*gradients
    return beta

def gradCostRidge(X, y, beta, lmb): #returns gradient of Ridge cost function
    n = len(X)
    return 2/n * X.T @ (X @ beta - y) + 2*lmb*beta

def gradCostOls(X, y, beta): #returns gradient of OLS cost function
    n = len(X)
    return 2/n * X.T @ (X @ beta - y)

def learningSchedule(t): #Returns learning rate eta
    t0, t1 = 5, 50
    return t0/(t+t1)

def mse(y, y_model): #Calculates the MSE for a model
    n = len(y)
    mean_se = np.sum((y-y_model)**2)
    return mean_se/n

def r2(y, y_model): #Calculates the R2 score for a model
    n = len(y)
    return 1 - n*mse(y,y_model)/np.sum((y-np.mean(z))**2)

def createX(x, y, n): #Creates design matrix with data x,y and complexity n
    if len(x.shape) > 1:
    	x = np.ravel(x)
    	y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) #Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
    	q = int((i)*(i+1)/2)
    	for k in range(i+1):
    		X[:,q+k] = (x**(i-k))*(y**k)

    return X

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def predict(X, beta):
    return X @ beta

def ols(X, y): #Finds optimal beta for the Ordinary Least Squares method
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def ridge(X, y, lmb): #Finds optimal beta for Ridge
    return np.linalg.pinv(X.T @ X + lmb*np.identity(X.shape[1])) @ X.T @ y

def SklearnSGD(X, y, epochs, penalty, eta, alpha = 0):
    sgdreg = SGDRegressor(max_iter=epochs, penalty = penalty,
                          eta0 = eta, learning_rate = 'constant', alpha = alpha, fit_intercept = False)
    sgdreg.fit(X, y)
    return sgdreg.coef_

def plotmseLR(MSE, LR):
    plt.plot(LR, MSE)
    plt.title("Mean squared error as a funciton of the learning rate")
    plt.xlabel("$\eta$")
    plt.ylabel("MSE")
    plt.savefig("MSELearningRate.pdf")
    plt.show()

def plotmseREL(MSE,LR,lmb):
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis")
    ax.set_title("MSE Ridge as a function of the learning rate and hyperparameter")
    ax.set_xticks(LR)
    ax.set_yticks(lmb)
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("$\lambda$")
    plt.savefig("HeatMapMSE_REL.pdf")
    plt.show()
