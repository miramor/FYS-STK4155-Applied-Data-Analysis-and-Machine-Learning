import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib
from NeuralNetworkReg import NeuralNetwork
plt.style.use("seaborn")
sns.set(font_scale=1.5)
plt.rcParams["font.family"] = "Times New Roman"; plt.rcParams['axes.titlesize'] = 21; plt.rcParams['axes.labelsize'] = 18; plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 18

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

def gradcostMSE(y_model, y): #returns gradient of OLS cost function
    n = len(y_model)
    return -2/n * y_model * (y_model - y)


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
    plt.savefig("MSELearningRate.pdf", bbox_inches='tight')
    plt.show()

def plotmseREL(MSE,LR,lmb):
    fig, ax = plt.subplots()
    x_vals = []
    y_vals = []
    for i in range(len(LR)):
        x_vals.append(np.format_float_scientific(LR[i], precision=1))
        y_vals.append(np.format_float_scientific(lmb[i], precision=1))
    sns.heatmap(MSE, annot=True, ax=ax, xticklabels=x_vals, yticklabels=y_vals, cmap="viridis")
    #fig, ax = plt.subplots(figsize = (10, 10))
    #sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Mean squared error as a function of the learning rate and hyperparameter")
    #ax.set_xticks(LR)
    #ax.set_yticks(lmb)
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("$\lambda$")
    plt.savefig("HeatMapMSE_REL.pdf", bbox_inches='tight')
    plt.show()

def make_heatmap(z,x,y, fn = "defaultheatmap.pdf", title = "", xlabel = "", ylabel = "" ):
    plt.clf()
    fig, ax = plt.subplots()
    x_vals = []
    y_vals = []
    for i in range(len(x)):
        x_vals.append(np.format_float_scientific(x[i], precision=1))
    for i in range(len(y)):
        y_vals.append(np.format_float_scientific(y[i], precision=1))
    sns.heatmap(z, annot=True, ax=ax, xticklabels=x, yticklabels=y, cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(fn, bbox_inches='tight')
    plt.show()

def sigmoid(x, derivative = False): #sigmoid as activation Function
    if derivative:
        return sigmoid(x)*(1-sigmoid(x))
    else:
        return 1/(1 + np.exp(-x))


def relu(x, derivative = False):
    if derivative:
        return np.where(x < 0, 0, 1)
    else:
        return np.maximum(x, 0)

def leaky_relu(x, alpha = 0.1, derivative = False):
    if derivative:
        return np.where(x < 0, alpha, 1)
    else:
        return np.where(x < 0, x*alpha, x)

def softmax(x, derivative = False):
    if derivative:
        return softmax(x) * (1 - softmax(x))
    else:
        return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims=True)

def linear(x):
    return x

def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)



def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector


def predict_logistic(x, coef):
    if len(x) == 1:
        y_pred = coef[0]
        for j in range(len(x[0])-1):
            y_pred += coef[j+1]*x[0][j]
        if y_pred < -500:
            return 0
        elif y_pred > 500:
            return 1
        else:
            return 1/(1+np.exp(-y_pred))
    else:
        pred = []
        for i in range(len(x)):
            print(x[i])
            pred.append(round(predict_logistic([x[i]], coef)))
        return pred


def logistic_reg(X_train, y_train, learn_rate, lmb, n_epochs, M):
    n = len(X_train) #number of datapoints
    m = int(n/M) #number of mini-batch cycles (M: size of batch)
    n_coef = len(X_train[0]) + 1
    coef = [0]*n_coef
    for e in range(n_epochs):
        s_error = 0
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_train[random_index:random_index+1][0]
            yi = y_train[random_index:random_index+1][0]
            y_pred = predict_logistic([xi], coef)
            error = y_pred - yi
            s_error += error**2
            coef[0] = coef[0] - learn_rate*error
            for j in range(1,n_coef):
                coef[j] = coef[j] - learn_rate*(error*xi[j-1]+lmb*coef[j])
    return coef


def kfold_nn_reg(X,y,k, lmb, eta, actfunc): #Implements k-fold method for use in logistic regression,  X = X_train, z = z_train
    mse_test = np.zeros(k) #for storing kfold samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
    mse_train = np.zeros(k)
    r2_test = np.zeros(k)
    r2_train = np.zeros(k)
    n = len(X)
    split = int(n/k) #Size of the folds
    mse2 = 0
    
    for j in range(k): #Splits into training and test set
        if j == k-1:
            X_train = X[:j*split]
            X_test = X[j*split:]
            y_train = y[:j*split]
            y_test = y[j*split:]
        else:
            X_train = np.concatenate((X[:(j)*split], X[(j+1)*split:]), axis = 0)
            X_test = X[j*split:(j+1)*split]
            y_train = np.concatenate((y[:(j)*split], y[(j+1)*split:]))
            y_test = y[j*split:(j+1)*split]


        NN = NeuralNetwork(X_train, y_train, epochs = 3000, batch_size = 50,
            n_categories = 1, eta = eta, lmbd = lmb, n_hidden_neurons = [10,10], activation_function = actfunc)
        NN.train()
        y_tilde = NN.predict_reg(X_train)
        y_predict = NN.predict_reg(X_test)
        mse2 += mse(y_test.reshape(y_predict.shape), y_predict)


    return mse2/k
