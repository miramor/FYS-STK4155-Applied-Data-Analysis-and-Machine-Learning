import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from functions1 import *

np.random.seed(5)


plt.style.use("seaborn")
sns.set(font_scale=1.5)
plt.rcParams["font.family"] = "Times New Roman"; plt.rcParams['axes.titlesize'] = 21; plt.rcParams['axes.labelsize'] = 18; plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 18

#Import data
df = pd.read_csv("water_potability[1].csv")
df = df.dropna()

#Extract features
df2 = df.drop(["Potability"], axis=1)
X = np.array(df2)

#Extract the target values
y = np.array(df["Potability"])

#Train-validation-test split
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.40, random_state=42) #Split to train and validation/test
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.50, random_state=42) #Split validation/test into validation and test data

X_k = np.vstack((X_train, X_val)) #used for k-fold cross validation
y_k = np.hstack((y_train, y_val)) #used for k-fold cross validataion

#Scale data:

scaler = StandardScaler() #Choosing StandardScaler for scaling
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

scaler.fit(X_k)
X_k = scaler.transform(X_k)



#Testing Decision tree, Random forrest, Bagging, and Boosting.

#Storing results
accuracy_dt_g = []
accuracy_dt_e = []
accuracy_rf_g = []
accuracy_rf_e = []

#Loop through different max depths
depth = list(range(2,34,2))
for i in depth:
    accuracy_dt_g.append(kfoldOwn(X_k, y_k, 5, DecisionTreeClassifier(random_state=0, max_depth = i)))
    accuracy_rf_g.append(kfoldOwn(X_k, y_k, 5, RandomForestClassifier(random_state=0, max_depth = i)))
    accuracy_dt_e.append(kfoldOwn(X_k, y_k, 5, DecisionTreeClassifier(random_state=0, max_depth = i, criterion="entropy")))
    accuracy_rf_e.append(kfoldOwn(X_k, y_k, 5, RandomForestClassifier(random_state=0, max_depth = i, criterion="entropy")))

#Plot results
plt.plot(depth, accuracy_dt_g, "o-", label="Decision tree (gini)")
plt.plot(depth, accuracy_dt_e, "o-", label="Decision tree (entropy)")
plt.plot(depth, accuracy_rf_g, "o-", label="Random forest (gini)")
plt.plot(depth, accuracy_rf_e, "o-", label="Random forest (entropy)")
plt.title("Depth analysis using K-Fold")
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("dt_rf_depth.pdf", bbox_inches='tight')
plt.show()

#Storing results
accuracy_bagging = []
accuracy_boosting = []

#Loop through different number of estimators
n_e = list(range(20,180,20))
k=0
for j in n_e:
    print(j)
    accuracy_bagging.append([])
    accuracy_boosting.append([])
    #Loop through different number of max depths
    for i in depth[:-2:2]:
        accuracy_bagging[k].append(kfoldOwn(X_k, y_k, 5, BaggingClassifier(n_estimators=j, random_state=0, base_estimator = DecisionTreeClassifier(max_depth=i))))
        accuracy_boosting[k].append(kfoldOwn(X_k, y_k, 5, GradientBoostingClassifier(n_estimators=j, learning_rate=1.0, max_depth=i, random_state=0, criterion = "mse")))
    k+=1


#Make heatmaps
make_heatmap(accuracy_bagging, depth[:-3:2], n_e, "bagging_hm.pdf", "Accuracy Bagging using K-Fold", "Depth", "Number of Estimators", save=True)
make_heatmap(accuracy_boosting, depth[:-3:2], n_e, "boosting_hm.pdf", "Accuracy Boosting using K-Fold", "Depth", "Number of Estimators", save=True)



#Testing with train and validation data:
max_v_rf = max(accuracy_rf_e) #Find max value
rf_depth_idx = accuracy_rf_e.index(max_v_rf) #Find index of max value
clf_rf = RandomForestClassifier(random_state=0, max_depth = depth[rf_depth_idx], criterion="entropy")  #Make random forest classifier
clf_rf.fit(X_train, y_train) #Fit the classifier to the training data
y_pred_rf = clf_rf.predict(X_val) #Predict values
make_confusion_matrix(y_val, y_pred_rf, "cm_rf.pdf", "Best model Random forest", "Predicted label", "True label", save=True)


clf_dt = DecisionTreeClassifier(random_state=0, max_depth = 6, criterion="gini")
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_val)
make_confusion_matrix(y_val, y_pred_dt, "cm_dt.pdf", "Best model Decision tree", "Predicted label", "True label", save=True)


clf_bagging = BaggingClassifier(n_estimators=140, random_state=0, base_estimator = DecisionTreeClassifier(max_depth=18))
clf_bagging.fit(X_train, y_train)
y_pred_bagging = clf_bagging.predict(X_val)
make_confusion_matrix(y_val, y_pred_bagging, "cm_bagging.pdf", "Best model Bagging", "Predicted label", "True label", save=True)


clf_boosting = GradientBoostingClassifier(n_estimators=40, learning_rate=1.0, max_depth=14, random_state=0, criterion = "mse")
clf_boosting.fit(X_train, y_train)
y_pred_boosting = clf_boosting.predict(X_val)
make_confusion_matrix(y_val, y_pred_boosting, "cm_boosting.pdf", "Best model Boosting", "Predicted label", "True label", save=True)







#Testing the best model:
y_pred_test = clf_bagging.predict(X_test)
make_confusion_matrix(y_test, y_pred_test, "cm_test.pdf", "Test data", "Predicted label", "True label", save=True)
