import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib
from sklearn.model_selection import StratifiedKFold
import pandas as pd

plt.style.use("seaborn")
sns.set(font_scale=1.5)
plt.rcParams["font.family"] = "Times New Roman"; plt.rcParams['axes.titlesize'] = 21; plt.rcParams['axes.labelsize'] = 18; plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 18
np.random.seed(5)

def make_heatmap(z,x,y, fn = "defaultheatmap.pdf", title = "", xlabel = "", ylabel = "", with_precision = False, save = False, string=False):
    """
    Makes a heatmap for given x,y and z.
    """
    fig, ax = plt.subplots()
    x_vals = []
    y_vals = []
    if string:
        sns.heatmap(z, annot=True, ax=ax, xticklabels=x, yticklabels=y, cmap='viridis')
        plt.yticks(rotation=0)
    else:
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
    plt.yticks(rotation=0)
    if save:
        plt.savefig(fn, bbox_inches='tight')
    plt.show()

def make_confusion_matrix(y_true,y_predict, fn = "defaultcm.pdf", title = "", xlabel = "", ylabel = "", save = False):
    """
    Makes a cofusion matrix.
    """
    plt.clf()
    cm = confusion_matrix(y_true,y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap = "Blues");  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title(f'Confusion Matrix {title}');
    ax.xaxis.set_ticklabels(["0","1"]); ax.yaxis.set_ticklabels(["0","1"]);
    if save:
        plt.savefig(fn, dpi=400)
    plt.show()

def Performance(model, X_train, y_train, X_val, y_val):
    """
    Evaluating performance of model
    """
    print('Train Accuracy : %.3f'%model.best_estimator_.score(X_train, y_train))
    print('Test Accuracy : %.3f'%model.best_estimator_.score(X_val, y_val))
    print('Best Accuracy Through Grid Search : %.3f'%model.best_score_)
    print('Best Parameters : ', model.best_params_)


def kfoldOwn(X,y,k,model):
    """
    Performing StratifiedKFold for MLPClassifier calculating mean accuracy of the k folds
    """
    accuracy = 0
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state= 7)
    for train, test in kfold.split(X, y):
        clf = model.fit(X[train], y[train])
        pred_sklearn = clf.predict(X[test])
        accuracy += accuracy_score(y[test], pred_sklearn)
    return accuracy/k

def kfoldKeras(X, y, k, model, params):
    """
    Performning StratifiedKFold for KerasClassifier calculating mean accuracy of the k folds
    """
    accuracy = 0
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state= 7)
    for train, test in kfold.split(X, y):
        model.fit(X[train], y[train], epochs=params[0], batch_size=params[1], verbose = params[2])
        accuracy += model.evaluate(X[test], y[test], verbose=0)[1]
    return accuracy/k

def Perform(clf, plot = False, param_=None):
    """
    Evaluating performance of model having done sklearn's gridsearch
    """
    print(f"Best: {clf.best_score_:.3f} using {clf.best_params_} ")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    params = clf.cv_results_['params']
    d = pd.DataFrame(params)
    d['Mean'] = means
    d['STD. Dev'] = stds
    print(d.to_latex(index=False, float_format="%.3g"))
