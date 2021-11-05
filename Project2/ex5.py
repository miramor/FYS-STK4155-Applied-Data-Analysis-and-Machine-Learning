from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
import importlib
import functions
importlib.reload(functions)
from functions import *


data = load_breast_cancer()
x = data['data']
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3) #Split the data into training and test sets


coef = logistic_reg(X_train, y_train, 0.001, 0.001, 5000, 10)

y_pred = predict_logistic(X_test, coef)



print(f"Results from own implementation: {accuracy_score_numpy(y_pred, y_test)}")


clf = LogisticRegression(random_state=0, max_iter = 5000).fit(X_train, y_train)

pred_sklearn = clf.predict(X_test)

print(f"Results from sklearn: {accuracy_score_numpy(pred_sklearn, y_test)}")



eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)


logistic_reg_models = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        coef = logistic_reg(X_train, y_train, eta, lmbd, 5000, 10)
        #dnn.fit(X_train, y_train)
        logistic_reg_models[i][j] = coef

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        coef = logistic_reg_models[i][j]

        train_pred = predict_logistic(X_train, coef)
        test_pred = predict_logistic(X_test, coef)

        train_accuracy[i][j] = accuracy_score_numpy(y_train, train_pred)
        test_accuracy[i][j] = accuracy_score_numpy(y_test, test_pred)

make_heatmap(test_accuracy, lmb_vals, eta_vals, fn = f"logistic_reg.pdf",
            xlabel = "lambda values", ylabel = "$\eta$ values", title = "Accuracy score sklearn")
