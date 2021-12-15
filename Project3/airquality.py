from inspect import CO_VARARGS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import importlib
import functions
import NNReg
importlib.reload(functions); importlib.reload(NNReg)
from NNReg import NeuralNetwork
from functions import *
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
#plt.rcParams["figure.figsize"]=12,12


def scale_data(X1,X2, scale_type = MinMaxScaler):
    try:
        X1 =X1[:,:]
        X2 =X2[:,:]
        scaler = scale_type()
        scaler.fit(X1)
        X1 = scaler.transform(X1)
        X2 = scaler.transform(X2)
    except:
        scaler = scale_type()
        scaler.fit(X1.reshape(-1,1))
        X1 = scaler.transform(X1.reshape(-1,1))
        X2 = scaler.transform(X2.reshape(-1,1))
        X1 =X1.flatten()
        X2 =X2.flatten()

    return X1, X2

def bootstrap(X,z): #Resamples with replacement
    n = len(z)
    data = np.random.randint(0,n,n)
    X_new = X[data] #random chosen columns for new design matrix
    z_new = z[data]
    return X_new, z_new
    
def mse(x1, x2):
    return np.mean((x1-x2)**2)


df = pd.read_csv('./AirQualityUCI.csv', sep = ";")
df.drop(['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], axis = 1,inplace = True)
df.dropna(inplace = True)


commas = ["C6H6(GT)", "T", "RH", "AH", "CO(GT)"]
for col in commas:
    df[col] = df[col].str.replace(',', '.').astype(float)


for col in df.columns:
    df.drop(df.index[df[col] == -200], inplace=True)


#response  'NOx(GT)'
print(len(df.columns),df.head())
features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
        'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

#Non linear terms
for d in range(2,6):
    for feat in features:
        df.insert(len(df.columns), feat+f'_{d}',df[feat]**d)

X = df.loc[:, df.columns != "NOx(GT)"]
y = df.loc[:, df.columns == "NOx(GT)"]

#sns.heatmap(df.corr(),cmap="OrRd",annot=True)
#plt.show()

def Ridge_bs(X_train, X_test, y_train, y_test, nb = 500, plot = False):
    complexity = X_train.shape[1]

    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for c in range(1,complexity+1):
        y_pred_bs = np.zeros((len(y_test), nb))
        for i in range(nb):
            X_sample, y_sample  = bootstrap(X_train[:,:c], y_train)

            regr = Ridge(alpha = 0.01).fit(X_sample, y_sample)
            y_tilde = regr.predict(X_sample)
            y_predict = regr.predict(X_test[:,:c])
            y_pred_bs[:,i] = y_predict.reshape(len(y_pred_bs))

            MSE_train[c-1] += mse(y_tilde, y_sample)
            MSE_test[c-1] += mse(y_predict, y_test)

        Variance[c-1] = np.mean(np.var(y_pred_bs, axis = 1))
        Bias[c-1] = np.mean( ( y_test - np.mean(y_pred_bs, axis = 1) )**2 )

    MSE_train /= nb; MSE_test /= nb

    if plot:
        plt.plot(range(1,complexity+1), Bias, label = "Bias$^2$")
        plt.plot(range(1,complexity+1), Variance, label = "Variance")
        plt.plot(range(1,complexity+1), MSE_train, label = "MSE Train")
        plt.plot(range(1,complexity+1), MSE_test, label = "MSE Test")
        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.title("Ridge Bias Variance Trade-Off")
        plt.grid()
        plt.ylim(0,0.005)
        #plt.xlim
        plt.legend()
        plt.show()

def OLS_bs(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = X_train.shape[1]

    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for c in range(1,complexity+1):
        y_pred_bs = np.zeros((len(y_test), nb))
        for i in range(nb):
            X_sample, y_sample  = bootstrap(X_train[:,:c], y_train)

            regr = LinearRegression().fit(X_sample, y_sample)
            y_tilde = regr.predict(X_sample)
            y_predict = regr.predict(X_test[:,:c])
            y_pred_bs[:,i] = y_predict.reshape(len(y_pred_bs))

            MSE_train[c-1] += mse(y_tilde, y_sample)
            MSE_test[c-1] += mse(y_predict, y_test)

        Variance[c-1] = np.mean(np.var(y_pred_bs, axis = 1))
        Bias[c-1] = np.mean( ( y_test - np.mean(y_pred_bs, axis = 1) )**2 )


    MSE_train /= nb; MSE_test /= nb

    if plot:
        plt.plot(range(1,complexity+1), Bias, label = "Bias$^2$")
        plt.plot(range(1,complexity+1), Variance, label = "Variance")
        plt.plot(range(1,complexity+1), MSE_train, label = "MSE Train")
        plt.plot(range(1,complexity+1), MSE_test, label = "MSE Test")
        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.title("OLS Bias Variance Trade-Off")
        plt.ylim(0,0.005)
        #plt.xlim(0,40)
        plt.legend()
        plt.show()

def DecisionTree_bs(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = 20

    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for c in tqdm(range(1,complexity+1)):
        y_pred_bs = np.zeros((len(y_test), nb))
        for i in range(nb):
            X_sample, y_sample  = bootstrap(X_train, y_train)

            regr = DecisionTreeRegressor(max_leaf_nodes=2+(c-1)*5).fit(X_sample, y_sample)
            print(regr.get_n_leaves())

            y_tilde = regr.predict(X_sample)
            y_predict = regr.predict(X_test)
            y_pred_bs[:,i] = y_predict.reshape(len(y_pred_bs))
            MSE_train[c-1] += mse(y_tilde, y_sample)
            MSE_test[c-1] += mse(y_predict, y_test)

        Variance[c-1] = np.mean(np.var(y_pred_bs, axis = 1))
        Bias[c-1] = np.mean( ( y_test - np.mean(y_pred_bs, axis = 1) )**2 )


    MSE_train /= nb; MSE_test /= nb

    if plot:
        plt.plot(range(1,complexity+1), Bias, label = "Bias$^2$")
        plt.plot(range(1,complexity+1), Variance, label = "Variance")
        plt.plot(range(1,complexity+1), MSE_train, label = "MSE Train")
        plt.plot(range(1,complexity+1), MSE_test, label = "MSE Test")
        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.title("Tree Bias Variance Trade-Off")
        plt.grid()
        #plt.ylim(0,3)
        #plt.xlim(0,40)
        plt.legend()
        plt.show()

def NN_bs(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = 25
    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for c in range(1, complexity + 1):
        y_pred_bs = np.zeros((len(y_test), nb))
        for i in range(nb):
            X_sample, y_sample  = bootstrap(X_train, y_train)

            NN = NeuralNetwork(X_sample, y_sample, epochs = 2000, batch_size = 50, n_categories = 1,
                                eta = 1e-4, lmbd = 0.001, n_hidden_neurons = [c,c], activation_function = relu)
            NN.train()
            y_tilde = NN.predict_reg(X_sample)
            y_predict = NN.predict_reg(X_test)

            y_pred_bs[:,i] = y_predict.reshape(len(y_pred_bs))

            MSE_train[c-1] += mse(y_tilde, y_sample)
            MSE_test[c-1] += mse(y_predict, y_test)

        Variance[c-1] = np.mean(np.var(y_pred_bs, axis = 1))
        Bias[c-1] = np.mean( ( y_test - np.mean(y_pred_bs, axis = 1) )**2 )


    MSE_train /= nb; MSE_test /= nb

    if plot:
        plt.plot(range(1,complexity+1), Bias, label = "Bias$^2$")
        plt.plot(range(1,complexity+1), Variance, label = "Variance")
        plt.plot(range(1,complexity+1), MSE_train, label = "MSE Train")
        plt.plot(range(1,complexity+1), MSE_test, label = "MSE Test")
        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.title("NN Bias Variance Trade-Off")
        plt.grid()
        #plt.xlim(0,40)
        plt.legend()
        plt.show()


params = {'activation': ['relu'],
            'alpha': [0.001,0.0001,0.01],
            'max_iter': [150],
            'batch_size': [100,50,20],
         # 'hidden_layer_sizes': [(10,), (100,), (500,), (1000,), (10,10)]  ,     #Define number of neurons per layer,
          'hidden_layer_sizes': [(i) for i in range(1,21)], #(50,50), (60,60), (70,70)
                                # (80,80), (90,90), (100,100), (120,120), (140,140), (160,160), (180,180), (200,200)]  ,     #Define number of neurons per layer,
          'solver': ['adam'],#, 'sgd', 'lbfgs'],
          'beta_1': [0.75, 0.8, 0.85, 0.9, 0.95],
          'beta_2': [0.9, 0.95, 0.999],
          'momentum': [0.1, 0.3, 0.5, 0.7, 0.9],
          'learning_rate_init': [0.1, 0.01, 0.001]
         }


def algorithm_pipeline(X_train, X_val, y_train, model, params_, cv=5, search_mode="GridSearchSV", n_iterations = 0, scoring_fit="accuracy" ):
    if search_mode =="GridSearchSV":
        gs = GridSearchCV(estimator=model, param_grid=params_, cv = cv, n_jobs=-1, scoring=scoring_fit, verbose=2)
    elif search_mode == "RandomizedSearchCV":
        gs = RandomizedSearchCV(estimator = model, param_distributions=params_, n_jobs = -1, cv=cv, scoring = scoring_fit, verbose = 2, n_iter=100)
    fitted_model = gs.fit(X_train, y_train)
    pred = fitted_model.predict(X_val)
    #pred = np.argmax(fitted_model.predict(X_val), axis=-1)
    #pred = (fitted_model.predict(X_val) > 0.5).astype("int32")
    return fitted_model, pred

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size = 0.2)
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
y_train_scaled, y_test_scaled = scale_data(y_train, y_test)

OLS_bs(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, nb = 100, plot=True)
Ridge_bs(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, nb=100, plot=True)
NN_bs(X_train_scaled[:,:12], X_test_scaled[:,:12], y_train_scaled, y_test_scaled, nb = 10, plot=True)
DecisionTree_bs(X_train_scaled[:,1:3], X_test_scaled[:,1:3], y_train_scaled, y_test_scaled, nb = 10, plot=True)









params = {'activation': ['relu'],
            'alpha': [0.0001],
            'max_iter': [2000],
            'batch_size': [20],
         # 'hidden_layer_sizes': [(10,), (100,), (500,), (1000,), (10,10)]  ,     #Define number of neurons per layer,
          'hidden_layer_sizes': [(i) for i in range(1,21)],
                                # (80,80), (90,90), (100,100), (120,120), (140,140), (160,160), (180,180), (200,200)]  ,     #Define number of neurons per layer,
          'solver': ['adam'],
          'learning_rate_init': [0.1,0.001,0.0001, 1e-5],
          'momentum': [0,0.3,0.8]
         }

model = MLPRegressor()
#gs = GridSearchCV(estimator=model, param_grid = params, n_jobs = -1)
#gs.fit(X_train_scaled, y_train_scaled)

#print(f"Best: {gs.best_score_:.3f} using {gs.best_params_} ")



def Perform(clf, plot = False, param_=None):
    print(f"Best: {clf.best_score_:.3f} using {clf.best_params_} ")
    #print("Best: %f using %s" %(clf.best_score_, clf.best_params_))
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    params = clf.cv_results_['params']
    d = pd.DataFrame(params)
    d['Mean'] = means
    d['STD. Dev'] = stds
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))
    print(d.to_latex(index=False, float_format="%.3g"))
