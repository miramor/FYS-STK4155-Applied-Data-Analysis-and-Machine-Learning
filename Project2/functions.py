import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
np.random.seed(2405) # Set a random seed
plt.style.use("seaborn")
plt.rcParams["font.family"] = "Times New Roman"; plt.rcParams['axes.titlesize'] = 21; plt.rcParams['axes.labelsize'] = 18; plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 18

def SGD(X, y, M, epochs, gradCostFunc, beta, eta, gamma = 0, lmb = None, LS=False): #Stochastic Gradient Descent
    n = len(X) #number of datapoints
    m = int(n/M) #number of mini-batch cycles (M: size of batch)
    v_prev = 0
    betaL = []
    for epoch in range(epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index*M:(random_index+1)*M]
            yi = y[random_index*M:(random_index+1)*M]
            if lmb is None:
                gradients = gradCostFunc(xi, yi, beta)
            else:
                gradients = gradCostFunc(xi, yi, beta, lmb)
            if LS:
                eta = learningSchedule(epoch*m+i)

            v = gamma * v_prev + eta*gradients
            beta = beta - v
            v_prev = v
        betaL.append(beta)

            #beta = beta - eta*gradients


    return beta, np.array(betaL)


def gradCostRidge(X, y, beta, lmb): #returns gradient of Ridge cost function
    n = len(X)
    return 2/n * X.T @ (X @ beta - y) + 2*lmb*beta

def gradCostOls(X, y, beta): #returns gradient of OLS cost function
    n = len(X)
    return 2/n * X.T @ (X @ beta - y)



def learningSchedule(t, t0=5, t1=100): #Returns learning rate eta
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

def SklearnSGD(X, y, epochs, penalty, eta, alpha = 0, tol=1e-4):
    sgdreg = SGDRegressor(max_iter=epochs, penalty = penalty,
                          eta0 = eta, learning_rate = 'constant', alpha = alpha, fit_intercept = False, tol=tol)
    sgdreg.fit(X, y)
    return sgdreg.coef_

def plotmseLR(MSE, LR, LS=False):
    plt.plot(LR, MSE)
    plt.title("Mean squared error as a function of the learning rate")
    plt.xlabel("$\eta$")
    plt.ylabel("$MSE_{Test}$")
    if LS:
        fn = "MSELearningRate_LS.pdf"
    else:
        fn ="MSELearningRate.pdf"
    plt.savefig(fn, bbox_inches='tight')
    plt.show()

def plotmseREL(MSE,x,y, LS=False):
    fig, ax = plt.subplots()
    x_vals = []
    y_vals = []
    for i in range(len(x)):
        x_vals.append(np.format_float_scientific(x[i], precision=1))
    for i in range(len(y)):
        y_vals.append(np.format_float_scientific(y[i], precision=1))
    sns.heatmap(MSE, annot=True, ax=ax, xticklabels=x_vals, yticklabels=y_vals, cmap="viridis")
    ax.set_title("Mean squared error as a function of the learning rate and hyperparameter")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    if LS:
        fn = "HeatMapMSE_REL_LS.pdf"
    else:
        fn ="HeatMapMSE_REL.pdf"
    plt.savefig(fn, bbox_inches='tight')
    plt.show()

def make_heatmap(z,x,y, fn = "defaultheatmap.pdf", title = "", xlabel = "", ylabel = "", with_precision=False):
    fig, ax = plt.subplots()
    x_vals = []
    y_vals = []
    for i in range(len(x)):
        x_vals.append(np.format_float_scientific(x[i], precision=1))
    for i in range(len(y)):
        y_vals.append(np.format_float_scientific(y[i], precision=1))
    if with_precision:
        sns.heatmap(z, annot=True, ax=ax, xticklabels=x_vals, yticklabels=y_vals, cmap="viridis")
    else:
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
        return np.exp(x) / np.sum(np.exp(x), keepdims=True)

def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)



def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector


def predict_logistic(X, coef):
    y_pred = X @ coef
    if max(abs(y_pred)) < 500:
        return np.around(1/(1+np.exp(-y_pred)))
    else:
        if type(y_pred) == type(1.5):
            if y_pred > 500:
                return 1
            else:
                return 0
        else:
            for i in range(len(y_pred)):
                if y_pred[i] > 500:
                    y_pred[i] = 1
                elif y_pred[i] < -500:
                    y_pred[i] = 0
                else:
                    y_pred[i] = np.around(1/(1+np.exp(-y_pred[i])))
            return y_pred


def gradLogistic(X, y, coef, lmbd):
    m = len(y)
    grad = (predict_logistic(X, coef) - y) @ X + lmbd * coef
    grad = (1/m)*grad
    return grad



def logistic_reg(X_train, y_train, learn_rate, lmb, n_epochs, M):
    n = len(X_train) #number of datapoints
    n_coef = len(X_train[0])
    coef = np.zeros(n_coef)
    coef = SGD(X_train, y_train, M, n_epochs, gradLogistic, coef, learn_rate, lmb)
    #print(coef)
    return coef

def kfold_logistic(X,y,k, lmbd, eta, n_epochs, sklearn = False): #Implements k-fold method for use in logistic regression,  X = X_train, z = z_train
    mse_test = np.zeros(k) #for storing kfold samples' MSE for varying complexity (rows:complexity, columns:bootstrap sample)
    mse_train = np.zeros(k)
    r2_test = np.zeros(k)
    r2_train = np.zeros(k)
    n = len(X)
    split = int(n/k) #Size of the folds
    accuracy = 0
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
        if sklearn:
            clf = SGDClassifier(loss="log", penalty="l2", learning_rate = "constant", eta0 = eta, alpha = lmbd, max_iter=n_epochs).fit(X_train, y_train)
            pred_sklearn = clf.predict(X_test)
            accuracy += accuracy_score_numpy(pred_sklearn, y_test)
        else:
            coef = logistic_reg(X_train, y_train, eta, lmbd, n_epochs, 10)
            test_pred = predict_logistic(X_test, coef)
            accuracy += accuracy_score_numpy(test_pred, y_test)


    return accuracy


    return mse2/k

def scale_data(X1,X2, scale_type = StandardScaler, with_std=False):
    try:
        X1 =X1[:,1:]
        X2 =X2[:,1:]
        scaler = StandardScaler(with_std=with_std)
        scaler.fit(X1)
        X1 = scaler.transform(X1)
        X2 = scaler.transform(X2)
    except:
        scaler = StandardScaler(with_std=False)
        scaler.fit(X1.reshape(-1,1))
        X1 = scaler.transform(X1.reshape(-1,1))
        X2 = scaler.transform(X2.reshape(-1,1))
        X1 =X1.flatten()
        X2 =X2.flatten()

    return X1, X2
