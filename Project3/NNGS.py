import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.constraints import maxnorm
import importlib
import functions1
importlib.reload(functions1)
from functions1 import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Feed Forward Neural Nework using sklearn's grid search with k=5 folds
"""
np.random.seed(5)

#Read data and remove rows with NaN
df = pd.read_csv("water_potability.csv")
df = df.dropna()
df2 = df.drop(["Potability"], axis=1)
X = np.array(df2) #Design matrix
y = np.array(df["Potability"]) #Targets

#Train-validation-test split
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.40, random_state=42) #Split i 1 og 2
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.50, random_state=42) #Split i 3 og 4

#Combine X_train and X_val for cross validation
X_k = np.vstack((X_train, X_val))
y_k = np.hstack((y_train, y_val))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_k)
X_k = scaler.transform(X_k)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)



#dicitionaries for grid search, serve as argument in algorithm_pipeline
"""
params1 = {'activation': ['relu','logistic'],#, 'tanh', 'logistic', 'identity'],
            'alpha': 10.0 ** -np.arange(7, 8),
            'max_iter': [10,50, 100, 500, 600, 800],
            'batch_size': [50],
          #'hidden_layer_sizes': [(100,), (50,100,), (50,75,100,)],
          'hidden_layer_sizes': [(10,), (100,), (500,), (1000,), (10,10)]  ,
          'solver': ['sgd','adam'],#, 'sgd', 'lbfgs'],
          'beta_1': [0.85],
          'beta_2': [0.9],
          'learning_rate' : ['constant'],#, 'adaptive', 'invscaling'] #only used when solver sgd
          'learning_rate_init': [0.1]
         }

params2 = {'activation': ['relu'],#, 'tanh', 'logistic', 'identity'],
            'alpha': [0.1],
            'max_iter': [500],
            'batch_size': [32],
            'hidden_layer_sizes': [(100,)],
            'solver': ['adam'],#, 'sgd', 'lbfgs'],
            'learning_rate' : ['constant'],#, 'adaptive', 'invscaling']
            'penalty': ["l1","l2"]
         }
"""
params3 = {'activation': ['relu'],
            'alpha': [0.000237],
            'max_iter': [150],
            'batch_size': [50],
          'hidden_layer_sizes': [(10,)],
          'solver': ['adam'],#, 'sgd', 'lbfgs'],
          'beta_1': [0.75, 0.8, 0.85, 0.9, 0.95],
          'beta_2': [0.9, 0.95, 0.999],
          'momentum': [0.1, 0.3, 0.5, 0.7, 0.9],
          'learning_rate_init': [0.1]
         }

def algorithm_pipeline(X_train, X_val, y_train, model, params_, cv=5, search_mode="GridSearchSV", n_iterations = 0, scoring_fit="accuracy" ):
    if search_mode =="GridSearchSV":
        gs = GridSearchCV(estimator=model, param_grid=params_, cv = cv, n_jobs=-1, scoring=scoring_fit, verbose=2)
    elif search_mode == "RandomizedSearchCV":
        gs = RandomizedSearchCV(estimator = model, param_distributions=params_, n_jobs = -1, cv=cv, scoring = scoring_fit, verbose = 2, n_iter=100)
    fitted_model = gs.fit(X_train, y_train)
    #pred = fitted_model.predict(X_val) #use when searching with MLPClassifier
    pred = (fitted_model.predict(X_val) > 0.5).astype("int32") #use when searching with KerasClassifier
    return fitted_model, pred

#search with MLPClassifier
#model_mlp = MLPClassifier(random_state=123)
#model_mlp, pred = algorithm_pipeline(X_k, X_val, y_k, model_mlp, params3, cv=5, search_mode="GridSearchSV", scoring_fit="accuracy")
#Perform(model_mlp)


"""
The grid search is divided into six tests using KerasClassifier. True or False to determine whether a test should be performed.
The tests are not consecutive.
"""
Test1 = True
Test2 = False
Test3 = False
Test4 = False
Test5 = False
Test6 = False

#Test running over different solvers
if Test1:
    print("Test 1")
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam']
    param_grid = dict(optimizer=optimizer)
    def create_model(optimizer):#learn_rate= 0.01, momentum = 0, dropout_rate=0.0):#, init_mode='uniform'):
        opt = {'SGD': SGD(lr=0.1, momentum=0.3, nesterov=True), 'RMSprop': RMSprop(lr=0.1), 'Adagrad':Adagrad(lr=0.1), 'Adam':Adam(lr=0.1)}
        model = Sequential()
        model.add(Dense(10, input_dim=9, activation='relu', kernel_regularizer= l2(0.000237)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=opt[optimizer], metrics=['accuracy'])
        return model
    model_ks = KerasClassifier(build_fn=create_model,epochs=150, batch_size=50, verbose=0)
    model_ks, pred = algorithm_pipeline(X_k, X_val, y_k, model_ks, param_grid, cv=5, scoring_fit="accuracy")
    Perform(model_ks)

#Test running over learning rate and epochs
if Test2:
    print("Test 2")
    learn_rate = [0.001, 0.01, 0.1]
    epochs = [10, 50, 100, 150, 200]
    param_grid = dict(learn_rate=learn_rate,epochs=epochs)
    def create_model(learn_rate=0.01, beta_1=0.8, beta_2=0.99):#learn_rate= 0.01, momentum = 0, dropout_rate=0.0):#, init_mode='uniform'):
        model = Sequential()
        model.add(Dense(10, input_dim=9, activation='relu',kernel_regularizer= l2(0.000237)))
        model.add(Dense(1, activation='sigmoid'))
        #optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999)
        optimizer = Adagrad(lr=learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    model_ks = KerasClassifier(build_fn=create_model,epochs=150, batch_size=50, verbose=0)
    model_ks, pred = algorithm_pipeline(X_k, X_val, y_k, model_ks, param_grid, cv=5, scoring_fit="accuracy")
    Perform(model_ks)

#Test running over kernel_initializer
if Test3:
    print("Test 3")
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    param_grid = dict(init_mode=init_mode)
    def create_model(init_mode="uniform"):#learn_rate= 0.01, momentum = 0, dropout_rate=0.0):#, init_mode='uniform'):
        model = Sequential()
        model.add(Dense(10, input_dim=9, kernel_initializer=init_mode,activation='relu',kernel_regularizer= l2(0.000237)))
        model.add(Dense(1, kernel_initializer=init_mode,activation='sigmoid'))
        optimizer = Adagrad(lr=0.1)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    model_ks = KerasClassifier(build_fn=create_model,epochs=150, batch_size=50, verbose=0)
    model_ks, pred = algorithm_pipeline(X_train, X_val, y_train, model_ks, param_grid, cv=5, scoring_fit="accuracy")
    Perform(model_ks)


#Test running over activation function
if Test4:
    print("Test 4")
    activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
    param_grid = dict(activation=activation)

    def create_model(activation="relu"):#learn_rate= 0.01, momentum = 0, dropout_rate=0.0):#, init_mode='uniform'):
        model = Sequential()
        model.add(Dense(12, input_dim=9, kernel_initializer='uniform',activation=activation,kernel_regularizer= l2(0.000237)))
        model.add(Dense(1, kernel_initializer='uniform',activation='sigmoid'))
        optimizer = Adagrad(lr=0.1)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    model_ks = KerasClassifier(build_fn=create_model,epochs=150, batch_size=50, verbose=0)
    model_ks, pred = algorithm_pipeline(X_train, X_val, y_train, model_ks, param_grid, cv=5, scoring_fit="accuracy")
    Perform(model_ks)


#Test running over weight_constraint and dropout rate
if Test5:
    print("Test 5")
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
    def create_model(dropout_rate=0.0, weight_constraint=0):#learn_rate= 0.01, momentum = 0, dropout_rate=0.0):#, init_mode='uniform'):
        model = Sequential()
        model.add(Dense(12, input_dim=9, kernel_initializer='uniform',activation='relu', kernel_constraint=maxnorm(weight_constraint),kernel_regularizer= l2(0.000237)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, kernel_initializer='uniform',activation='sigmoid'))
        optimizer = Adagrad(lr=0.1)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    model_ks = KerasClassifier(build_fn=create_model,epochs=150, batch_size=50, verbose=0)
    model_ks, pred = algorithm_pipeline(X_train, X_val, y_train, model_ks, param_grid, cv=5, scoring_fit="accuracy")
    Perform(model_ks)


#Test running over neurons, epochs and batch size
if Test6:
    print("Test 6")
    neurons = [1, 5, 10, 15, 20, 25, 30, 100]
    epochs = [10,50,100,150]
    batch_size = [10, 20, 40, 60, 80]
    param_grid = dict(neurons=neurons, batch_size=batch_size, epochs=epochs)
    def create_model(neurons=1):#learn_rate= 0.01, momentum = 0, dropout_rate=0.0):#, init_mode='uniform'):
        model = Sequential()
        model.add(Dense(neurons, input_dim=9, kernel_initializer='uniform',activation='relu', kernel_constraint=maxnorm(2)))
        model.add(Dropout(0.1))
        model.add(Dense(1, kernel_initializer='uniform',activation='sigmoid'))
        optimizer = Adagrad(lr=0.1)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    model_ks = KerasClassifier(build_fn=create_model,epochs=150, batch_size=50, verbose=0)
    model_ks, pred = algorithm_pipeline(X_train, X_val, y_train, model_ks, param_grid, cv=5, scoring_fit="accuracy")
    Perform(model_ks)
