import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import importlib
import functions1
importlib.reload(functions1)
from functions1 import *
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


"""
Feed Forward Neural Nework with MLPClassifier performing own grid search with StratifiedKFold (k=5)
"""

np.random.seed(5)

df = pd.read_csv("water_potability.csv") #read data
df = df.dropna() #remove all rows with NaN
df2 = df.drop(["Potability"], axis=1) # design matrix
X = np.array(df2)

y = np.array(df["Potability"]) #Targets

#Train-validation-test split
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.40, random_state=42) #Split i 1 og 2
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.50, random_state=42) #Split i 3 og 4

X_k = np.vstack((X_train, X_val)) #used for k-fold cross validation
y_k = np.hstack((y_train, y_val)) #used for k-fold cross validataion

#Scale data:
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scalerV = StandardScaler() #Choosing StandardScaler for scaling
scaler = scalerV
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

scaler.fit(X_k)
X_k = scaler.transform(X_k)



"""
The grid search is divided into four tests. True or False to determine whether a test should be performed.
The tests can be run consecutively using the optimal parameters from the previous test.
"""
Test1=True
Test2 = True
Test3 = True
Test4 = True

#Test running over learning rate and regularization parameter
if Test1:
    lmd_vals = np.logspace(-8,-3,9) #regulularization parameter values
    eta_vals =  [0.001, 0.05, 0.1, 0.15, 0.2] #epoch values
    test_accuracy = np.zeros((len(eta_vals), len(lmd_vals))) #storing accuracy values after k-fold
    epoch = 150 #fixed epoch
    batch_size = 50 #fixed batch size
    n_neurons = (10,) # fixed layer with one neuron
    print("Test 1: running over learning rate and regularization parameter")
    print(f"Fixed parameters: epochs = {epoch}, batch_size = {batch_size}, hidden_layer_size = {n_neurons}")
    for i, eta in enumerate(eta_vals):
        for j, lmd in enumerate(lmd_vals):
            dnn = MLPClassifier(hidden_layer_sizes=(n_neurons), activation='relu',solver="adam",
                                 alpha=lmd, batch_size= batch_size, learning_rate_init=eta, max_iter=epoch, random_state=1, momentum=0, early_stopping=False)
            test_accuracy[i][j] = kfoldOwn(X_k, y_k, 5, dnn)
        print(f"{(i+1)/len(eta_vals)*100} % done")

    make_heatmap(test_accuracy, lmd_vals, eta_vals, ylabel = "Learning rate $\eta$", xlabel = "Regularization parameter $\lambda$", title = "Accuracy cross validation",with_precision = True, fn = "TA_LRL.pdf", save=True )
    #find optimal values
    maxAcc_test_index = np.argwhere(test_accuracy == np.max(test_accuracy))[0]
    optimal_eta_test = eta_vals[maxAcc_test_index[0]]
    optimal_lmd_test = lmd_vals[maxAcc_test_index[1]]
    print(f"Accuracy score {np.max(test_accuracy)}")
    print(f"Optimal eta {optimal_eta_test}")
    print(f"Optimal lambda {optimal_lmd_test}")
    print("\n")

#Test running over learning rate and epochs
if Test2:
    eta_vals =  [0.001, 0.05, 0.1, 0.15]
    epoch_vals=[5,10, 15, 20, 30, 40, 75, 100, 150, 200, 250, 300, 350]
    test_accuracy = np.zeros((len(eta_vals), len(epoch_vals)))
    batch_size = 50
    n_neurons = (10,)
    print("Test 2: running over learning rate and epochs")
    print(f"Fixed parameters: batch_size = {batch_size}, hidden_layer_size = {n_neurons}, regularization paramter = {optimal_lmd_test}")
    test_accuracy = np.zeros((len(eta_vals), len(epoch_vals)))
    for i, eta in enumerate(eta_vals):
        for j, epoch in enumerate(epoch_vals):
            dnn = MLPClassifier(hidden_layer_sizes=((10,)), activation='relu',solver="adam",
                                 alpha=optimal_lmd_test, batch_size= batch_size, learning_rate_init=eta, max_iter=epoch, random_state=1, momentum=0)
            test_accuracy[i][j] =  kfoldOwn(X_k, y_k, 5, dnn)
        print(f"{(i+1)/len(eta_vals)*100} % done")

    make_heatmap(test_accuracy, epoch_vals, eta_vals, ylabel = "Learning rate $\eta$", xlabel = "Epochs", title = "Accuracy score test set", fn = "TA_LRE.pdf", save=True, string=True )
    maxAcc_test_index = np.argwhere(test_accuracy == np.max(test_accuracy))[0]
    optimal_eta_test = eta_vals[maxAcc_test_index[0]]
    optimal_epoch_test = epoch_vals[maxAcc_test_index[1]]
    print(f"Accuracy score test {np.max(test_accuracy)}")
    print(f"Optimal eta test {optimal_eta_test}")
    print(f"Optimal epoch test {optimal_epoch_test}")
    print("\n")
    for i, val in enumerate(eta_vals):
        plt.plot(epoch_vals,test_accuracy[i,:],"o-", label=f"eta={val}")
    plt.title("Accuracy cross validation")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of epochs")
    plt.legend()
    plt.savefig("TA_Epochs.pdf", bbox_inches='tight')
    plt.show()


#optimal_lmd_test = 0.00023713737056616554
#optimal_eta_test = 0.1
#optimal_epoch_test = 150

#Test running over number of hidden layers and activation function
if Test3:
    batch_size = 50
    n_hidden_neurons =  [(1,),(8,), (9,), (10,), (11,), (100,), (120,),(150,), (2,2), (5,5), (10,1,), (2,2,2)]
    activFunc = ["logistic", "relu", "tanh"]
    test_accuracy = np.zeros((len(n_hidden_neurons), len(activFunc)))
    print("Test 3: running over number of hidden layers, activation function")
    print(f"Fixed parameters: batch_size = {batch_size}, epochs = {optimal_epoch_test}, regularization paramter = {optimal_lmd_test}")
    for i, n_neuron in enumerate(n_hidden_neurons):
        for j, acfunc in enumerate(activFunc):
            dnn = MLPClassifier(hidden_layer_sizes=n_neuron, activation=acfunc,solver="adam",
                                alpha=optimal_lmd_test, batch_size= batch_size, learning_rate_init=optimal_eta_test, max_iter=optimal_epoch_test, random_state=1, momentum = 0)
            test_accuracy[i][j] =  kfoldOwn(X_k, y_k, 5, dnn)
        print(f"{(i+1)/len(n_hidden_neurons)*100} % done")

    #make_heatmap(test_accuracy, np.arange(len(activFunc)),np.arange(len(n_hidden_neurons)) , xlabel = "Activation function", ylabel = "Neurons", title = "Accuracy score cross validation", fn = "TA_NAcf.pdf", save=True )
    make_heatmap(test_accuracy, activFunc,n_hidden_neurons , xlabel = "Activation function", ylabel = "Neurons", title = "Accuracy score cross validation", fn = "TA_NAcf.pdf", save=True,string=True )
    maxAcc_test_index = np.argwhere(test_accuracy == np.max(test_accuracy))[0]
    optimal_hneurons = n_hidden_neurons[maxAcc_test_index[0]]
    optimal_acfunc = activFunc[maxAcc_test_index[1]]
    print(f"Accuracy score test {np.max(test_accuracy)}")
    print(f"Optimal hidden neurons test {optimal_hneurons}")
    print(f"Optimal activation function {optimal_acfunc}")
    print("\n")


#optimal_eta_test = 0.1
#optimal_hneurons = (10,)


#Test running over epochs and batch size for different solvers (adam, sgd with and without momentum)
if Test4:
    epoch_vals = [1, 3, 6, 9,12, 15, 20, 25, 30, 35, 50, 100, 150, 200, 250]
    print("Test 4: running over epochs and batch size for different solvers")
    print(f"Fixed parameters: regularization parameter {optimal_lmd_test}, learning reate = {optimal_eta_test}, hidden_layer_size = {optimal_hneurons}")
    batch_vals = [32, 38, 42, 46, 50, 60, 70, 80]
    #batch_vals = [46, 50, 60, 70]
    test_accuracyA = np.zeros((len(epoch_vals), len(batch_vals)))
    test_accuracySGD = np.zeros((len(epoch_vals), len(batch_vals)))
    test_accuracySGDM = np.zeros((len(epoch_vals), len(batch_vals)))
    test_accuracySGDM2 = np.zeros((len(epoch_vals), len(batch_vals)))
    for i, epoch in enumerate(epoch_vals):
        for j, batch in enumerate(batch_vals):
            dnnAdam = MLPClassifier(hidden_layer_sizes=(optimal_hneurons), activation="relu",solver="adam",
                                alpha=optimal_lmd_test, batch_size= batch, learning_rate_init=optimal_eta_test,  max_iter=epoch, momentum=0, random_state=1)
            dnnSGD = MLPClassifier(hidden_layer_sizes=(optimal_hneurons), activation="relu",solver="sgd",
                                alpha=optimal_lmd_test, batch_size= batch, learning_rate_init=optimal_eta_test, max_iter=epoch, momentum=0, random_state=1)
            dnnSGDM = MLPClassifier(hidden_layer_sizes=(optimal_hneurons), activation="relu",solver="sgd",
                                alpha=optimal_lmd_test, batch_size= batch, learning_rate_init=optimal_eta_test,  max_iter=epoch, momentum=0.3, random_state=1)
            dnnSGDM2 = MLPClassifier(hidden_layer_sizes=(optimal_hneurons), activation="relu",solver="sgd",
                                alpha=optimal_lmd_test, batch_size= batch, learning_rate_init=optimal_eta_test,  max_iter=epoch, momentum=0.9, random_state=1)
            test_accuracyA[i][j] =  kfoldOwn(X_k, y_k, 5, dnnAdam)
            test_accuracySGDM2[i][j] =  kfoldOwn(X_k, y_k, 5, dnnSGDM2)
            test_accuracySGD[i][j] =  kfoldOwn(X_k, y_k, 5, dnnSGD)
            test_accuracySGDM[i][j] =  kfoldOwn(X_k, y_k, 5, dnnSGDM)
        print(f"{(i+1)/len(epoch_vals)*100} % done")
    optimal_epoch = []
    optimal_batch = []
    maxAcc_t_index = []
    for test in [test_accuracyA, test_accuracySGDM2, test_accuracySGD, test_accuracySGDM]:
        print(np.max(test))
        maxAcc_test_index = np.argwhere(test == np.max(test))[0]
        maxAcc_t_index.append(maxAcc_test_index)
        optimal_epoch.append(epoch_vals[maxAcc_test_index[0]])
        optimal_batch.append(batch_vals[maxAcc_test_index[1]])
    print(f"optimal epoch Adam: {optimal_epoch[0]}, optimal epoch SGD2: {optimal_epoch[1]}, optimal epoch SGD: {optimal_epoch[2]}, optimal epoch SGDM: {optimal_epoch[3]}" )
    print(f"optimal batch Adam: {optimal_batch[0]}, optimal batch SGD2: {optimal_batch[1]}, optimal batch SGD: {optimal_batch[2]}, optimal batch SGDM: {optimal_batch[3]}")

    plt.plot(epoch_vals,test_accuracyA[:,maxAcc_t_index[0][1]], "o-", label="Adam")
    plt.plot(epoch_vals,test_accuracySGDM2[:,maxAcc_t_index[1][1]], "o-", label="SGD, $\gamma = 0.9$")
    plt.plot(epoch_vals,test_accuracySGD[:,maxAcc_t_index[2][1]],"o-", label="SGD")
    plt.plot(epoch_vals,test_accuracySGDM[:,maxAcc_t_index[3][1]], "o-",label="SGD, $\gamma = 0.3$")
    plt.title("Accuracy cross validation")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of epochs")
    plt.legend()
    plt.savefig("Solver.pdf",bbox_inches='tight')
    plt.show()

#When not running tests use these parameters for MLPClassifier below
#optimal_lmd_test = 0.00023713737056616554
#optimal_eta_test = 0.1
#optimal_epoch_test = 150
#optimal_hneurons = (10,)

#Computing test accuracy of FNN with optimized parameters from grid search
clf = MLPClassifier(hidden_layer_sizes=optimal_hneurons, activation="relu",solver="adam",
                    alpha=optimal_lmd_test, batch_size= 50, learning_rate_init=optimal_eta_test,  max_iter=optimal_epoch_test, momentum=0, random_state=1).fit(X_train, y_train)
y_pred = clf.predict(X_val)
make_confusion_matrix(y_val, y_pred, fn="ConfusionMatrixOwn.pdf", save=True)
print(classification_report(y_val,y_pred))
print(accuracy_score(y_val, y_pred))


