import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import importlib
import functions
importlib.reload(functions)
from functions import *

np.random.seed(2405) # Set a random seed

N = 500 #number of datapoints
d = 5 #complexity
x1 = np.random.uniform(0,1,N)
x2 = np.random.uniform(0,1,N)
y = FrankeFunction(x1,x2)
noise = np.random.normal(0, 1, size=(y.shape)) #Find random noise
y_noisy =  FrankeFunction(x1,x2) + noise*0.2
X = createX(x1,x2,d)

tts = train_test_split(X,y_noisy,test_size=0.2) #Split the data into training and test sets
theta = np.random.randn(len(X[0])) #first guess of beta
#Regular OLS Regression
betaOLS2 = ols(tts[0], tts[2]) #regular OLS regression
y_tildeOLS2 = predict(tts[0],betaOLS2) #regular OLS regression
y_predictOLS2 = predict(tts[1],betaOLS2)




#OLS
def SGDSearch_OLS(tts, theta, learnSced=False, gamma = 0, epochs = 1000, scale = False):
    if scale: #scala data
        tts[0] =tts[0][:,1:]
        tts[1] =tts[1][:,1:]
        theta = np.random.randn(len(tts[0][0])) #first guess of beta
        scaler = StandardScaler(with_std=False)
        scaler.fit(tts[0])
        tts[0] = scaler.transform(tts[0])
        tts[1] = scaler.transform(tts[1])
        scaler2 = StandardScaler(with_std=False)
        scaler2.fit(tts[2].reshape(-1,1))
        tts[2] = scaler2.transform(tts[2].reshape(-1,1))
        tts[3] = scaler2.transform(tts[3].reshape(-1,1))
        tts[2] =tts[2].flatten()
        tts[3] =tts[3].flatten()

    nbatches = [1, 5, 20, 50, 100,250]
    etaL = np.logspace(-5,-1,5)
    MSE_train_OLS = np.zeros((len(etaL), len(nbatches)))
    MSE_test_OLS = np.zeros((len(etaL), len(nbatches)))
    print(f"OLS, epochs = {epochs}, learning schedule = {learnSced}, gamma = {gamma}")
    print("Start loop over learning rate and mini-batches")

    for i in range(len(etaL)):
        for j in range(len(nbatches)):
            beta, betaL = SGD(tts[0], tts[2], M=int((N/nbatches[j])), epochs = epochs, gamma = gamma, beta = theta, gradCostFunc = gradCostOls, eta = etaL[i], LS=learnSced)
            ytilde = predict(tts[0],beta)
            ypredict = predict(tts[1], beta)
            MSE_train_OLS[i, j] = mse(tts[2],ytilde)
            MSE_test_OLS[i, j] = mse(tts[3],ypredict)

        print(f"{int((i+1)/len(etaL)*100)} % done")
    make_heatmap(MSE_test_OLS, nbatches, etaL, fn = f"ex1_hm_ols_{epochs}_sc{1 if scale else 0}.pdf", title = "MSE as a function different hyperparameters", xlabel = "Number of mini batches", ylabel = "$\eta$")
    #args = np.argmin(MSE_test_OLS,axis=2)
    #MSE_test_OLS = np.min(MSE_test_OLS,axis=2)
    #MSE_train_OLS = MSE_train_OLS[:,:,args]
    minMSE_OLS_train_index = np.argwhere(MSE_train_OLS == np.min(MSE_train_OLS))[0]
    minMSE_OLS_test_index = np.argwhere(MSE_test_OLS == np.min(MSE_test_OLS))[0]

    optimal_eta_train = etaL[minMSE_OLS_train_index[0]]
    optimal_eta_test = etaL[minMSE_OLS_test_index[0]]
    optimal_batch_train = nbatches[minMSE_OLS_train_index[1]]
    optimal_batch_test = nbatches[minMSE_OLS_test_index[1]]

    print(f"Optimal eta train {optimal_eta_train}")
    print(f"Optimal eta test {optimal_eta_test}")
    print(f"Optimal mini batches train {optimal_batch_train}")
    print(f"Optimal mini batches test {optimal_batch_test}")
    print("\n")

    betaSKOLSTr = SklearnSGD(tts[0], tts[2], penalty = None, eta = optimal_eta_train, epochs=epochs)
    betaSKOLSTe = SklearnSGD(tts[0], tts[2], penalty = None, eta = optimal_eta_test, epochs=epochs)
    y_tildeSKOLS = predict(tts[0],betaSKOLSTr)
    y_predictSKOLS = predict(tts[1],betaSKOLSTe)


    print(f"MSE train SGD: {np.min(MSE_train_OLS)}, eta={optimal_eta_train}")
    print(f"MSE train Sklearn: {mse(tts[2],y_tildeSKOLS)}, eta={optimal_eta_train}")
    print(f"MSE train Regular OLS: {mse(tts[2],y_tildeOLS2)}")
    print("\n")
    print(f"MSE test SGD: {np.min(MSE_test_OLS)}, eta={optimal_eta_test}")
    print(f"MSE test Sklearn : {mse(tts[3],y_predictSKOLS)}, eta={optimal_eta_test}")
    print(f"MSE test Regular OLS: {mse(tts[3],y_predictOLS2)}")

def SGDSearch_Ridge(tts, theta, learnSced=False, gamma = 0, epochs = 1000, scale = False):
    if scale: #scala data
        tts[0] =tts[0][:,1:]
        tts[1] =tts[1][:,1:]
        theta = np.random.randn(len(tts[0][0])) #first guess of beta
        scaler = StandardScaler(with_std=False)
        scaler.fit(tts[0])
        tts[0] = scaler.transform(tts[0])
        tts[1] = scaler.transform(tts[1])
        scaler2 = StandardScaler(with_std=False)
        scaler2.fit(tts[2].reshape(-1,1))
        tts[2] = scaler2.transform(tts[2].reshape(-1,1))
        tts[3] = scaler2.transform(tts[3].reshape(-1,1))
        tts[2] =tts[2].flatten()
        tts[3] =tts[3].flatten()

    nbatches = [1, 5, 20, 50, 100, 250]
    eta_ = 0.1
    etaLRid = np.logspace(-3,-0.8,4)
    lambdaL = np.logspace(-6,-2,5)
    MSE_train_Ridge = np.zeros((len(lambdaL), len(nbatches)))
    MSE_test_Ridge = np.zeros((len(lambdaL), len(nbatches)))
    print("\n")
    print(f"Ridge, epochs = {epochs}, eta = {eta_}, learning schedule = {learnSced}, gamma = {gamma}")
    print("Start loop over hyperparameter and mini-batches")
    for i in range(len(lambdaL)):
        for j in range(len(nbatches)):
            beta, betaL = SGD(tts[0], tts[2], M=int((N/nbatches[j])), epochs = epochs, beta = theta, gradCostFunc = gradCostRidge, eta = eta_, gamma = gamma, lmb=lambdaL[i], LS=learnSced)
            ytilde = predict(tts[0],beta)
            ypredict = predict(tts[1], beta)
            MSE_train_Ridge[i, j] = mse(tts[2],ytilde)
            MSE_test_Ridge[i, j] = mse(tts[3],ypredict)

        print(f"{int((i+1)/len(lambdaL)*100)} % done")


    # args = np.argmin(MSE_test_OLS,axis=2)
    # MSE_test_Ridge = np.min(MSE_test_Ridge,axis=2)
    # MSE_train_Ridge = MSE_train_Ridge[:,:,args]
    minMSE_Ridge_train_index = np.argwhere(MSE_train_Ridge == np.min(MSE_train_Ridge))[0]
    minMSE_Ridge_test_index = np.argwhere(MSE_test_Ridge == np.min(MSE_test_Ridge))[0]

    optimal_lmb_train = lambdaL[minMSE_Ridge_train_index[0]]
    optimal_lmb_test = lambdaL[minMSE_Ridge_test_index[0]]
    optimal_batch_train = nbatches[minMSE_Ridge_train_index[1]]
    optimal_batch_test = nbatches[minMSE_Ridge_test_index[1]]
    make_heatmap(MSE_test_Ridge, nbatches, lambdaL, fn = f"ex1_hm_R_{epochs}_sc{1 if scale else 0}.pdf", xlabel = "Number of mini batches", ylabel = "$\lambda$")
    print("Results for Ridge")
    print(f"Optimal lambda train {optimal_lmb_train}")
    print(f"Optimal lambda test {optimal_lmb_test}")
    print(f"Optimal mini batches train {optimal_batch_train}")
    print(f"Optimal mini batches test {optimal_batch_test}")

    print("\n")
    betaSKRidTr = SklearnSGD(tts[0], tts[2], penalty = "l2", eta = eta_, epochs=epochs, alpha=optimal_lmb_train)
    betaSKRidTe = SklearnSGD(tts[0], tts[2], penalty = "l2", eta = eta_, epochs=epochs, alpha=optimal_lmb_test)
    y_tildeSKRid = predict(tts[0],betaSKRidTr)
    y_predictSKRid = predict(tts[1],betaSKRidTe)
    betaRidge2 = ridge(tts[0], tts[2], optimal_lmb_test) #regular Ridge regression
    y_tildeRidge2 = predict(tts[0],betaRidge2) #regular Ridge regression
    y_predictRidge2 = predict(tts[1],betaRidge2)

    print(f"MSE train SGD: {np.min(MSE_train_Ridge)}, lmb={optimal_lmb_train}" )
    print(f"MSE train Sklearn: {mse(tts[2],y_tildeSKRid)}, eta = {eta_}, lmb={optimal_lmb_train}")
    print(f"MSE train Regular Ridge: {mse(tts[2],y_tildeRidge2)}")
    print("\n")
    print(f"MSE test SGD: {np.min(MSE_test_Ridge)} lmb={optimal_lmb_test}")
    print(f"MSE test Sklearn : {mse(tts[3],y_predictSKRid)}, eta={eta_}, lmb={optimal_lmb_test}")
    print(f"MSE test Regular Ridge: {mse(tts[3],y_predictRidge2)}")

def SGDSearch_Ridge2(tts, theta, learnSced=False, gamma = 0, epochs = 1000, scale = False):
    if scale: #scala data
        tts[0] =tts[0][:,1:]
        tts[1] =tts[1][:,1:]
        theta = np.random.randn(len(tts[0][0])) #first guess of beta
        scaler = StandardScaler(with_std=True)
        scaler.fit(tts[0])
        tts[0] = scaler.transform(tts[0])
        tts[1] = scaler.transform(tts[1])
        scaler2 = StandardScaler(with_std=True)
        scaler2.fit(tts[2].reshape(-1,1))
        tts[2] = scaler2.transform(tts[2].reshape(-1,1))
        tts[3] = scaler2.transform(tts[3].reshape(-1,1))
        tts[2] =tts[2].flatten()
        tts[3] =tts[3].flatten()
    nbatches = [1, 5, 20, 50, 100, 250]
    eta_ = 0.1
    etaLRid = np.logspace(-4,-1,4)
    lambdaL = np.logspace(-6,-2,5)
    MSE_train_Ridge = np.zeros((len(lambdaL), len(etaLRid)))
    MSE_test_Ridge = np.zeros((len(lambdaL), len(etaLRid)))
    print("\n")
    print(f"Ridge, epochs = {epochs}, eta = {eta_}, learning schedule = {learnSced}, gamma = {gamma}")
    print("Start loop over hyperparameter and mini-batches")
    for i in range(len(lambdaL)):
        for j in range(len(etaLRid)):
            beta, betaL = SGD(tts[0], tts[2], M=5, epochs = epochs, beta = theta, gradCostFunc = gradCostRidge, eta = etaLRid[j], gamma = gamma, lmb=lambdaL[i], LS=learnSced)
            ytilde = predict(tts[0],beta)
            ypredict = predict(tts[1], beta)
            MSE_train_Ridge[i, j] = mse(tts[2],ytilde)
            MSE_test_Ridge[i, j] = mse(tts[3],ypredict)

        print(f"{int((i+1)/len(lambdaL)*100)} % done")

    # args = np.argmin(MSE_test_OLS,axis=2)
    # MSE_test_Ridge = np.min(MSE_test_Ridge,axis=2)
    # MSE_train_Ridge = MSE_train_Ridge[:,:,args]
    minMSE_Ridge_train_index = np.argwhere(MSE_train_Ridge == np.min(MSE_train_Ridge))[0]
    minMSE_Ridge_test_index = np.argwhere(MSE_test_Ridge == np.min(MSE_test_Ridge))[0]

    optimal_lmb_train = lambdaL[minMSE_Ridge_train_index[0]]
    optimal_lmb_test = lambdaL[minMSE_Ridge_test_index[0]]
    optimal_eta_train = etaLRid[minMSE_Ridge_train_index[1]]
    optimal_eta_test = etaLRid[minMSE_Ridge_test_index[1]]
    make_heatmap(MSE_test_Ridge, etaLRid, lambdaL, title = "MSE as a funciton of different hyperparameters",fn = f"ex1_hm_R2_{epochs}_sc{1 if scale else 0}.pdf", xlabel = "$\eta$", ylabel = "$\lambda$")
    print("Results for Ridge")
    print(f"Optimal lambda train {optimal_lmb_train}")
    print(f"Optimal lambda test {optimal_lmb_test}")
    print(f"Optimal eta train {optimal_eta_train}")
    print(f"Optimal eta test {optimal_eta_test}")

#SGDSearch_OLS(tts, theta, learnSced=False, gamma = 0., epochs = 5000)
#SGDSearch_OLS(tts, theta, learnSced=False, gamma = 0., epochs = 5000, scale = True)
#SGDSearch_OLS(tts, learnSced=True)
#SGDSearch_Ridge(tts, theta, learnSced=False, gamma = 0., epochs = 5000)
#SGDSearch_Ridge(tts, theta, learnSced=False, gamma = 0., epochs = 5000, scale = True)
#SGDSearch_Ridge2(tts, theta, learnSced=False, gamma = 0., epochs = 5000)
#SGDSearch_Ridge2(tts, theta, learnSced=False, gamma = 0., epochs = 5000, scale = True)
#SklearnSGD følsom bedre når eta = 0.1



betaOLS, betaL = SGD(tts[0], tts[2], M=5, epochs = 500, beta = theta, gradCostFunc = gradCostOls, eta = 0.1)
betaRidge, betaL = SGD(tts[0], tts[2], M=5, epochs = 500, beta = theta, gradCostFunc = gradCostRidge, eta = 0.1, lmb=0.001)
betaSKOLS = SklearnSGD(tts[0], tts[2], penalty = None, eta = 0.01, epochs=500, alpha=0)
betaSKRidge = SklearnSGD(tts[0], tts[2], penalty = "l2", eta = 0.01, epochs=500, alpha=0.001)

y_predictOLS = predict(tts[1], betaOLS)
y_predictRidge = predict(tts[1], betaRidge)
y_predictSKOLS = predict(tts[1], betaSKOLS)
y_predictSKRidge = predict(tts[1], betaSKRidge)

mseOLS = mse(tts[3], y_predictOLS)
mseRidge = mse(tts[3], y_predictRidge)
mseSKOLS = mse(tts[3], y_predictSKOLS)
mseSKRidge = mse(tts[3], y_predictSKRidge)
print(mseOLS, mseRidge, mseSKOLS, mseSKRidge)


def momentum(gamma=0,epoch=1000, eta = 0.0001, LS = False):
    beta, betaL = SGD(tts[0], tts[2], M=5, epochs = epoch, beta = theta, gradCostFunc = gradCostRidge,
        eta = eta, gamma = gamma, lmb=0, LS=LS)
    MSE_train = []
    MSE_test = []
    for b in betaL:
        ytilde = predict(tts[0], b)
        ypredict = predict(tts[1], b)
        MSE_train.append(mse(tts[2],ytilde))
        MSE_test.append(mse(tts[3],ypredict))

    return MSE_train, MSE_test

def plot_momLS():
    epochs = 50; eta = 0.0001
    MSE_train1, MSE_test1 = momentum(gamma = 0, epoch = epochs, eta = eta, LS = True)
    MSE_train2, MSE_test2 = momentum(gamma = 0, epoch = epochs, eta = eta, LS = False)
    MSE_train3, MSE_test3 = momentum(gamma = 0.5, epoch = epochs, eta = eta, LS = True)
    MSE_train4, MSE_test4 = momentum(gamma = 0.5, epoch = epochs, eta = eta, LS = False)
    MSE_train5, MSE_test5 = momentum(gamma = 0.9, epoch = epochs, eta = eta, LS = True)
    MSE_train6, MSE_test6 = momentum(gamma = 0.9, epoch = epochs, eta = eta, LS = False)

    plt.plot(range(epochs), MSE_test1, label = f"$\gamma = 0$, LS=True")
    plt.plot(range(epochs), MSE_test2, label = f"$\gamma = 0$, LS=False")
    plt.plot(range(epochs), MSE_test3, label = f"$\gamma = 0.5$, LS=True")
    plt.plot(range(epochs), MSE_test4, label = f"$\gamma = 0.5$, LS=False")
    plt.plot(range(epochs), MSE_test5, label = f"$\gamma = 0.9$, LS=True")
    plt.plot(range(epochs), MSE_test6, label = f"$\gamma = 0.9$, LS=False")

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Rate of convergence")
    #plt.ylim(0,0.1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"convergenceLS_e{epochs}_eta1emin4.p1", bbox_inches='tight')
    plt.show()






"""
Gamma 0 epochs 100
0.061179462375961446 Ridge train
0.05790740474536868 OLS train
0.047542734656535524 Ridge test
0.047668818870419606 OLS test

Gamma 0.1 epochs 100
0.06062400643103371 ridge train
0.05798675298564241 ols train
0.04733901014962059 ridge test
0.04783374258247264 ols test

Gamma 0.5 epochs 100
0.05767113921311186 ridge train
0.058868365105594275 OLS train
0.046620279967346016 ridge test
0.04722092080076692 OLS Test

gamma 0.8 epochs 10
0.04909461566508381 ridge test
0.06520061520923416 ridge train
0.05042543547183574 OLS test
0.06544913873697837 OLS train

gamma 0.5 epochs 10
0.053323238707657344 ridge test
0.0708164801106029 ridge train
0.05370406799860736 OLS test
0.07107720124404597 OLS train

Gamma 0.1 epochs 10
0.06250403116843889 ridge test
0.08396226467227269 ridge train
0.05994707123062036 OLS test
0.07641846181389328 OLS train

gamma 0 epochs 10
0.06535732792319413 ridge test
0.08809934325879303 ridge train
0.06050696829465634 ols test
0.07775498206742107 ols train


Ridge Train
0.05523349320751571 gamma 0
0.055235009963888596 gamma 0.01
0.05526179236382237 gamma 0.1
0.05546419167525585 gamma 0.3
0.05510058172755377 gamma 0.6

Ridge test
0.04708050643986462 gamma 0
0.047077590362926004 gamma 0.01
0.047062025132772314 0.1
0.047110286443651506 0.3
0.0465550329469382 gamma 0.6


OLS train
0.054750843415976844 gamma 0
0.0547416152093702 gamma 0.01
0.05465621301158492 gamma 0.1
0.0547416152093702 gamma 0.3
0.054159540079293356 gamma 0.6



OLS test
0.046572729141812076 gamma 0
0.04656653038705307 gamma 0.01
0.046521979629923556 gamma 0.1
0.04656653038705307 gamma 0.3
0.04667897217585048 gamma 0.6


"""
