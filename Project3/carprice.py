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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams["figure.figsize"]=12,12
import importlib
import functions
import NNReg
importlib.reload(functions); importlib.reload(NNReg)
from NNReg import NeuralNetwork
from functions import *
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

cars = pd.read_csv('./CarPrice_Assignment.csv')
CompanyName = cars["CarName"].apply(lambda x: x.split(" ")[0])

#Insert into dataframe
cars.insert(3, "CompanyName", CompanyName)
cars.drop(["CarName"], axis=1, inplace = True)
cars.drop(['car_ID'],axis=1,inplace=True)

#Check for spelling errors in CompanyName
cars.CompanyName.unique()

#Correct the spelling errors
cars = cars.replace(to_replace = "maxda", value = "mazda")
cars = cars.replace(to_replace = "Nissan", value = "nissan")
cars = cars.replace(to_replace = "porcshce", value = "porsche")
cars = cars.replace(to_replace = "toyouta", value = "toyota")
cars = cars.replace(to_replace = "vokswagen", value = "volkswagen")
cars = cars.replace(to_replace = "vw", value = "volkswagen")

"""
#Look at correlation
sns.heatmap(cars.corr(),cmap="OrRd",annot=True)
plt.show()
"""

#Keep only variables with high correlation to price
cars = cars.drop(["peakrpm", "compressionratio", "stroke", "carheight", "symboling"],axis=1)

#Look at correlation between some other variables
vars1 = ['wheelbase', 'carlength', 'carwidth','curbweight']
vars2 = ['citympg','highwaympg']
vars3 = ['enginesize','boreratio','horsepower']
"""
sns.heatmap(cars.filter(vars1).corr(),cmap="OrRd",annot=True)
plt.show()
sns.heatmap(cars.filter(vars2).corr(),cmap="OrRd",annot=True)
plt.show()
sns.heatmap(cars.filter(vars3).corr(),cmap="OrRd",annot=True)
plt.show()
"""
#We only need one of those variables that are highly correlated
cars.drop(["citympg"], axis=1, inplace = True)
cars.drop(['wheelbase'],axis=1,inplace=True)
cars.drop(['carlength'],axis=1,inplace=True)
cars.drop(['carwidth'],axis=1,inplace=True)
cars.drop(['horsepower'],axis=1,inplace=True)

cars = pd.get_dummies(cars)


#Keeping only Buick
remove = ['CompanyName_alfa-romero', 'CompanyName_audi', 'CompanyName_bmw','CompanyName_chevrolet', 'CompanyName_dodge',
       'CompanyName_honda', 'CompanyName_isuzu', 'CompanyName_jaguar',
       'CompanyName_mazda', 'CompanyName_mercury', 'CompanyName_mitsubishi',
       'CompanyName_nissan', 'CompanyName_peugeot', 'CompanyName_plymouth',
       'CompanyName_porsche', 'CompanyName_renault', 'CompanyName_saab',
       'CompanyName_subaru', 'CompanyName_toyota', 'CompanyName_volkswagen',
       'CompanyName_volvo',]
cars.drop(remove, axis = 1, inplace = True)


#Keeping only fuelsystem with high correlation to price
remove = ['fuelsystem_1bbl', 'fuelsystem_4bbl',
       'fuelsystem_idi', 'fuelsystem_mfi',
       'fuelsystem_spdi', 'fuelsystem_spfi']
cars.drop(remove, axis = 1, inplace = True)

#Remove all engine types, none with >0.5 correlation to price
remove = ['enginetype_dohc', 'enginetype_dohcv', 'enginetype_l', 'enginetype_ohc', 'enginetype_ohcf',
       'enginetype_ohcv', 'enginetype_rotor']
cars.drop(remove, axis = 1, inplace = True)

#Remove all cylinders, none with >0.5 correlation to price
remove = ['cylindernumber_eight', 'cylindernumber_five',
       'cylindernumber_four', 'cylindernumber_six', 'cylindernumber_three',
       'cylindernumber_twelve', 'cylindernumber_two']
cars.drop(remove, axis = 1, inplace = True)

#Remove all car types as well
remove = ['carbody_convertible', 'carbody_hardtop',
       'carbody_hatchback', 'carbody_sedan', 'carbody_wagon']
cars.drop(remove, axis = 1, inplace = True)

#Removing the rest of the variables without high corr to price
remove = ['fueltype_diesel', 'fueltype_gas',
       'aspiration_std', 'aspiration_turbo', 'doornumber_four',
       'doornumber_two', 'drivewheel_4wd','enginelocation_front', 'enginelocation_rear',]
cars.drop(remove, axis = 1, inplace = True)
"""
sns.heatmap(cars.corr(),cmap="OrRd",annot=True)
plt.show()
"""

#Make dataframes for non linear relationships
nonlin = ['curbweight', 'enginesize', 'boreratio', 'highwaympg']

cars2 = cars.copy()
for feat in nonlin:
    cars2.insert(len(cars2.columns),feat + "2", cars[feat]**2)

cars3 = cars2.copy()
for feat in nonlin:
    cars3.insert(len(cars3.columns),feat + "3", cars[feat]**3)

cars4 = cars3.copy()
for feat in nonlin:
    cars4.insert(len(cars4.columns),feat + "4", cars[feat]**4)

cars5 = cars4.copy()
for feat in nonlin:
    cars5.insert(len(cars5.columns),feat + "5", cars[feat]**5)

def scale_data(X1,X2, scale_type = StandardScaler):
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


#Make list with features sorted by complexity
features = []
features += nonlin
features += ['CompanyName_buick', 'drivewheel_fwd', 'drivewheel_rwd',
       'fuelsystem_2bbl', 'fuelsystem_mpfi']

for i in range(2,6):
    for j in nonlin:
        features.append(j+str(i))

X = cars5.loc[:, cars5.columns != "price"]
y = cars5.loc[:, cars5.columns == "price"]

#Train test split
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size = 0.2)

#Scale data
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
y_train_scaled, y_test_scaled = scale_data(y_train, y_test)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

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

            regr = Ridge(alpha = 0.0001).fit(X_sample, y_sample)
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
        #plt.ylim(0,0.4)
        plt.legend()
        plt.show()

def Lasso_bs(X_train, X_test, y_train, y_test, nb = 500, plot = False):
    complexity = X_train.shape[1]

    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for c in range(1,complexity+1):
        y_pred_bs = np.zeros((len(y_test), nb))
        for i in range(nb):
            X_sample, y_sample  = bootstrap(X_train[:,:c], y_train)

            regr = Lasso(alpha = 0.0001, max_iter = 5000, fit_intercept=False).fit(X_sample, y_sample)
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
        plt.title("Lasso Bias Variance Trade-Off")
        #plt.ylim(0,2)
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
        #plt.ylim(0,0.4)
        #plt.xlim(0,40)
        plt.legend()
        plt.show()



def DecisionTree_bs(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = 20

    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for c in range(1,complexity+1):
        y_pred_bs = np.zeros((len(y_test), nb))
        for i in range(nb):
            X_sample, y_sample  = bootstrap(X_train, y_train)

            regr = DecisionTreeRegressor(max_depth=3*c).fit(X_sample, y_sample)
            #print(regr.get_n_leaves())

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
        plt.ylim(0,2)
        #plt.xlim(0,40)
        plt.legend()
        plt.show()



def NN_bs(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = 20
    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for c in range(1, complexity + 1):
        y_pred_bs = np.zeros((len(y_test), nb))
        for i in range(nb):
            X_sample, y_sample  = bootstrap(X_train, y_train)

            NN = NeuralNetwork(X_sample, y_sample, epochs = 2000, batch_size = 50, n_categories = 1,
                                eta = 1e-5, lmbd = 0.001, n_hidden_neurons = [c,c], activation_function = relu)
            NN.train()
            y_tilde = NN.predict_reg(X_sample)
            y_predict = NN.predict_reg(X_test)

            y_pred_bs[:,i] = y_predict.reshape(len(y_pred_bs))

            MSE_train[c-1] += mse(y_tilde, y_sample)
            MSE_test[c-1] += mse(y_predict, y_test)
            
            """
            regr = MLPRegressor(activation = "relu", solver = "sgd", max_iter = 5000, learning_rate_init = 1e-5, 
                                alpha = 0.001, batch_size=20, momentum = 0, hidden_layer_sizes=(c,c)).fit(X_sample, y_sample)

            y_tilde = regr.predict(X_sample)
            y_predict = regr.predict(X_test)
            y_pred_bs[:,i] = y_predict.reshape(len(y_pred_bs))

            #print(y_tilde)
            MSE_train[c-1] += mse(y_tilde, y_sample)
            MSE_test[c-1] += mse(y_predict, y_test)
            """

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
        plt.ylim(0,2)
        #plt.xlim(0,40)
        plt.legend()
        plt.show()

def SVM_bs(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    complexity = 6

    MSE_train = np.zeros(complexity)
    MSE_test = np.zeros(complexity)
    Bias = np.zeros(complexity)
    Variance = np.zeros(complexity)

    for c in range(1,complexity+1):
        y_pred_bs = np.zeros((len(y_test), nb))
        for i in range(nb):
            X_sample, y_sample  = bootstrap(X_train, y_train)

            regr = SVR(gamma = 'auto', epsilon = 0.1, kernel = 'poly', C = 1, degree = c).fit(X_sample, y_sample)
            #print(regr.get_n_leaves())

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
        plt.title("SVM Bias Variance Trade-Off")
        #plt.ylim(0,4)
        #plt.xlim(0,40)
        plt.legend()
        plt.show()



OLS_bs(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, nb=50, plot=True)
Ridge_bs(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, nb=50, plot=True)
#Lasso_bs(X_train_scaled[:,:9], X_test_scaled[:,:9], y_train_scaled, y_test_scaled, nb=50, plot=True)
#NN_bs(X_train_scaled[:,:9], X_test_scaled[:,:9], y_train_scaled, y_test_scaled, nb = 5, plot=True)
#DecisionTree_bs(X_train_scaled[:,:9], X_test_scaled[:,:9], y_train_scaled, y_test_scaled, nb = 5, plot=True)
#SVM_bs(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, nb = 10, plot=True)


paramsNN = {'activation': ['relu'],
            'alpha': [0.0001],
            'max_iter': [2000],
            'batch_size': [20],
         # 'hidden_layer_sizes': [(10,), (100,), (500,), (1000,), (10,10)]  ,     #Define number of neurons per layer,
          'hidden_layer_sizes': [(100,60,40)],
                                # (80,80), (90,90), (100,100), (120,120), (140,140), (160,160), (180,180), (200,200)]  ,     #Define number of neurons per layer,
          'solver': ['adam'],
          'learning_rate_init': [0.001]
         }

paramsDT = {
    'splitter': ['best', 'random'],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'max_leaf_nodes': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
}

paramsRF = {
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'max_leaf_nodes': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
}
paramsRidge = {
    'alpha': [1,0.1,0.01,0.001,0.0001,0.00001],

}
paramsOLS = {
    'fit_intercept': [True,False],
}

paramsSVM = {
    'kernel': ['poly'], #, 'rbf'],
    'degree': [1,2,3,4,5,6,7,8,9,10],
    'C': [0.5, 1, 2, 5, 10, 15],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.01, 0.1]
}

#model = SVR()
#gs = GridSearchCV(estimator=model, param_grid = paramsSVM, n_jobs = -1)
#gs.fit(X_train_scaled[:,:9], y_train_scaled)
#print(f"Best: {gs.best_score_:.3f} using {gs.best_params_} ")
