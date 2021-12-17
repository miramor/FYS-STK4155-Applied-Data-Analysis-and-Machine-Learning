import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import importlib
import functions
import NNReg
importlib.reload(functions); importlib.reload(NNReg)
from functions import *
from NNReg import *
from imageio import imread
from sklearn.neural_network import MLPRegressor

np.random.seed(2405)
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

features = []
features += nonlin
features += ['CompanyName_buick', 'drivewheel_fwd', 'drivewheel_rwd',
       'fuelsystem_2bbl', 'fuelsystem_mpfi']

X = cars5.loc[:, cars5.columns != "price"]
y = cars5.loc[:, cars5.columns == "price"]

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size = 0.2)

#Scale data
X_train, X_test = scale_data(X_train, X_test)
y_train, y_test = scale_data(y_train, y_test)

eta = np.logspace(-5,-2,4)
n_neurons = np.logspace(0,2,3)
n_neurons = np.array([1,5,10,15,20,25])
lmb = 0.001
epochs = 5000
n_L = 2
mse_train = np.zeros((len(eta), len(n_neurons)))
mse_test = np.zeros((len(eta), len(n_neurons)))
r2_train = np.zeros((len(eta), len(n_neurons)))
r2_test = np.zeros((len(eta), len(n_neurons)))


actfunc = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "relu": relu,
    "leaky_relu": leaky_relu
}
af = "sigmoid"

for i,eta_ in enumerate(eta):
    for j,n_  in enumerate(n_neurons):
        NN = NeuralNetwork(X_train, y_train, epochs = epochs, batch_size = 50,
            n_categories = 1, eta = eta_, lmbd = lmb, n_hidden_neurons = [n_]*n_L, activation_function = actfunc[af])
        NN.train()
        y_tilde = NN.predict_reg(X_train)
        y_predict = NN.predict_reg(X_test)

        mse_train_ = mse(y_train.reshape(y_tilde.shape), y_tilde)
        mse_test_ = mse(y_test.reshape(y_predict.shape), y_predict)
        r2_train_ = r2(y_train.reshape(y_tilde.shape), y_tilde)
        r2_test_ = r2(y_test.reshape(y_predict.shape), y_predict)


        mse_train[i,j] = mse_train_
        mse_test[i,j] = mse_test_
        r2_train[i,j] = r2_train_
        r2_test[i,j] = r2_test_

        print(f"Eta: {eta_} | # of neurons: {n_}")
        print(f"Training MSE: {mse_train_}")
        print(f"Test MSE: {mse_test_}")
        print(f"Training R2: {r2_train_}")
        print(f"Test R2: {r2_test_}")
        print("------------------------")
make_heatmap(mse_train, n_neurons, eta, fn = f"mse_train_{af}_L{n_L}_neur_eta.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "MSE training set")
make_heatmap(mse_test, n_neurons, eta, fn = f"mse_test_{af}_L{n_L}_neur_eta.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "MSE test set")
make_heatmap(r2_train, n_neurons, eta, fn = f"r2_train_{af}_L{n_L}_neur_eta.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "R2 training set")
make_heatmap(r2_test, n_neurons, eta, fn = f"r2_test_{af}_L{n_L}_neur_eta.pdf",
            xlabel = "Number of neurons per layer", ylabel = "Learning rate $\eta$", title = "R2 test set")




lambdas = np.logspace(-4,-1,4)
mse_train = np.zeros((len(eta), len(lambdas)))
mse_test = np.zeros((len(eta), len(lambdas)))
r2_train = np.zeros((len(eta), len(lambdas)))
r2_test = np.zeros((len(eta), len(lambdas)))

for i,eta_ in enumerate(eta):
    for j,lmb_  in enumerate(lambdas):
        NN = NeuralNetwork(X_train, y_train, epochs = epochs, batch_size = 50,
            n_categories = 1, eta = eta_, lmbd = lmb_, n_hidden_neurons = [10], activation_function = actfunc[af])
        NN.train()
        y_tilde = NN.predict_reg(X_train)
        y_predict = NN.predict_reg(X_test)

        mse_train_ = mse(y_train.reshape(y_tilde.shape), y_tilde)
        mse_test_ = mse(y_test.reshape(y_predict.shape), y_predict)
        r2_train_ = r2(y_train.reshape(y_tilde.shape), y_tilde)
        r2_test_ = r2(y_test.reshape(y_predict.shape), y_predict)

        mse_train[i,j] = mse_train_
        mse_test[i,j] = mse_test_
        r2_train[i,j] = r2_train_
        r2_test[i,j] = r2_test_

        print(f"Eta: {eta_} | lambda: {lmb_}")
        print(f"Training MSE: {mse_train_}")
        print(f"Test MSE: {mse_test_}")
        print(f"Training R2: {r2_train_}")
        print(f"Test R2: {r2_test_}")
        print("------------------------")

make_heatmap(mse_train, lambdas, eta, fn = f"mse_train_{af}_L1_lambda_eta.pdf",
            xlabel = "Regularization parameter $\lambda$", ylabel = "Learning rate $\eta$", title = "MSE training set")
make_heatmap(mse_test, lambdas, eta, fn = f"mse_test_{af}_L1_lambda_eta.pdf",
            xlabel = "Regularization parameter $\lambda$", ylabel = "Learning rate $\eta$", title = "MSE test set")
make_heatmap(r2_train, lambdas, eta, fn = f"r2_train_{af}_L1_lambda_eta.pdf",
            xlabel = "Regularization parameter $\lambda$", ylabel = "Learning rate $\eta$", title = "R2 training set")
make_heatmap(r2_test, lambdas, eta, fn = f"r2_test_{af}_L1_lambda_eta.pdf",
            xlabel = "Regularization parameter $\lambda$", ylabel = "Learning rate $\eta$", title = "R2 test set")


"""
print("Kfold NN reg MSE: ",kfold_nn_reg(X, y_noisy, 5, 0.01, 0.001, actfunc[af]))


regr = MLPRegressor(solver = "sgd", random_state=1, hidden_layer_sizes = (25, 25), alpha = 0.01, max_iter=5000).fit(X_train, y_train)
y_pred_reg = regr.predict(X_test)
print("MLPRegressor MSE: ",mse(y_test, y_pred_reg))
"""