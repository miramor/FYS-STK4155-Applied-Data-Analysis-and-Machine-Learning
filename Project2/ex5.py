from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import math
import importlib
import functions
importlib.reload(functions)
from functions import *
from sklearn.preprocessing import StandardScaler
np.random.seed(2405) # Set a random seed

#def kFold_score_logistic_reg(X, y):


data = load_breast_cancer()
x = data['data']
y = data['target']
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)




eta_vals = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]#np.logspace(-5, 0, 5)
lmbd_vals = [0, 0.0001, 0.001, 0.01, 0.1, 1] #np.logspace(-5, 0, 5)

n_epochs = 150


test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy_sklearn = np.zeros((len(eta_vals), len(lmbd_vals)))

k=0
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        print(k)
        k+=1
        test_accuracy[i][j] = kfold_logistic(x, y, 5, lmbd_vals[j], eta_vals[i], n_epochs, sklearn = False)
        test_accuracy_sklearn[i][j] = kfold_logistic(x, y, 5, lmbd_vals[j], eta_vals[i], n_epochs, sklearn = True)

make_heatmap(test_accuracy, lmbd_vals, eta_vals, fn = f"logistic_reg.pdf",
            xlabel = "$\lambda$ values", ylabel = "$\eta$ values", title = "Accuracy score logistic regression")

make_heatmap(test_accuracy_sklearn, lmbd_vals, eta_vals, fn = f"logistic_reg_sklearn.pdf",
            xlabel = "$\lambda$ values", ylabel = "$\eta$ values", title = "Accuracy score logistic regression using SKLearn")




X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Split the data into training and test sets

coef = logistic_reg(X_train, y_train, 0.00001, 1, n_epochs, 10)
y_pred = predict_logistic(X_test, coef)

cm_own = confusion_matrix(y_test, y_pred)

clf = SGDClassifier(loss="log", penalty="l2", learning_rate = "constant", eta0 = 0.01, alpha = 0, max_iter=n_epochs).fit(X_train, y_train)
pred_sklearn = clf.predict(X_test)

cm_sklearn = confusion_matrix(y_test, pred_sklearn)

print(y_pred)
print(y_test)


sns.heatmap(cm_own, annot=True, cmap='Blues')
plt.title("Confusion matrix for own implementation")
plt.xlabel("Predicted values")
plt.ylabel("True Values")
plt.savefig("Confusion_matrix_own.pdf", bbox_inches='tight')
plt.show()

sns.heatmap(cm_sklearn, annot=True, cmap='Blues')
plt.title("Confusion matrix for sklearn")
plt.xlabel("Predicted values")
plt.ylabel("True values")
plt.savefig("Confusion_matrix_sklearn.pdf", bbox_inches='tight')
plt.show()
