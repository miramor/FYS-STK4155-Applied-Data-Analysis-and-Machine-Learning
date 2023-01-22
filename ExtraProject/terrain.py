import importlib
import functions
importlib.reload(functions)
from functions import *
np.random.seed(42)


def BVT_OLS(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    #Bias-Variance trade off with OLS
    complexity = X_train.shape[1]
    degree = int((-3+np.sqrt(9-8*(1-complexity)))/2)
    Loss = np.zeros(degree)
    Variance = np.zeros(degree)
    Bias = np.zeros(degree)
    
    for d in tqdm(range(1, degree + 1)):

        c = int((d+1)*(d+2)/2)
        regr = LinearRegression()

        #Calculates the Bias, Variance and Loss with nb bootstrap cycles
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train[:,:c], y_train, X_test[:,:c], y_test, loss='mse', num_rounds = nb)

        Loss[d-1] = avg_expected_loss
        Variance[d-1] = avg_var
        Bias[d-1] = avg_bias
    
    if plot:
        plt.plot(range(1,degree+1), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, 'o-', label = "Variance")
        plt.plot(range(1,degree+1), Loss, 'o-', label = "Loss")
        plt.xlabel("Polynomial Degree")
        #plt.ylabel("Error")
        plt.title("OLS Bias-Variance Trade Off")
        plt.ylim(0,0.2)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_ols.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show()

        plt.plot(range(1,degree+1), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, 'o-', label = "Variance")
        plt.plot(range(1,degree+1), Loss, 'o-', label = "Loss")
        plt.xlabel("Polynomial Degree")
        #plt.ylabel("Error")
        plt.title("OLS Bias-Variance Trade Off")
        plt.ylim(0,0.05)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_ols_close.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show()

def BVT_Ridge(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    #Bias-Variance Trade Off Ridge
    complexity = X_train.shape[1]
    degree = int((-3+np.sqrt(9-8*(1-complexity)))/2)
    Loss = np.zeros(degree)
    Variance = np.zeros(degree)
    Bias = np.zeros(degree)
    
    for d in tqdm(range(1, degree + 1)):

        c = int((d+1)*(d+2)/2)
        regr = Ridge(alpha = 0.0005)

        #Calculates the Bias, Variance and Loss with nb bootstrap cycles
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train[:,1:c], y_train, X_test[:,1:c], y_test, loss='mse', num_rounds = nb)

        Loss[d-1] = avg_expected_loss
        Variance[d-1] = avg_var
        Bias[d-1] = avg_bias
    
    if plot:
        plt.plot(range(1,degree+1), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, 'o-', label = "Variance")
        plt.plot(range(1,degree+1), Loss, 'o-', label = "Loss")
        plt.xlabel("Polynomial Degree")
        #plt.ylabel("Error")
        plt.title("Ridge Bias-Variance Trade Off")
        plt.ylim(0,0.02)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_ridge.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show()
        
        plt.plot(range(1,degree+1), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, 'o-', label = "Variance")
        plt.plot(range(1,degree+1), Loss, 'o-', label = "Loss")
        plt.xlabel("Polynomial Degree")
        #plt.ylabel("Error")
        plt.title("Ridge Bias-Variance Trade Off")
        plt.ylim(0,0.01)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_ridge_close.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show() 

def BVT_Lasso(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    #Bias-Variance Trade Off Lasso
    complexity = X_train.shape[1]
    degree = int((-3+np.sqrt(9-8*(1-complexity)))/2)
    Loss = np.zeros(degree)
    Variance = np.zeros(degree)
    Bias = np.zeros(degree)
    
    for d in tqdm(range(1, degree + 1)):

        c = int((d+1)*(d+2)/2)
        regr = Lasso(alpha = 0.000005, max_iter=1000, tol = 1e-3, fit_intercept=False)

        #Calculates the Bias, Variance and Loss with nb bootstrap cycles
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train[:,1:c], y_train, X_test[:,1:c], y_test, loss='mse', num_rounds = nb)

        Loss[d-1] = avg_expected_loss
        Variance[d-1] = avg_var
        Bias[d-1] = avg_bias
    
    if plot:
        plt.plot(range(1,degree+1), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, 'o-', label = "Variance")
        plt.plot(range(1,degree+1), Loss, 'o-', label = "Loss")
        plt.xlabel("Polynomial Degree")
        #plt.ylabel("Error")
        plt.title("Lasso Bias-Variance Trade Off")
        #plt.ylim(0,0.4)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_lasso.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show() 

        plt.plot(range(1,degree+1), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(1,degree+1), Variance, 'o-', label = "Variance")
        plt.plot(range(1,degree+1), Loss, 'o-', label = "Loss")
        plt.xlabel("Polynomial Degree")
        #plt.ylabel("Error")
        plt.title("Lasso Bias-Variance Trade Off")
        plt.ylim(0,0.02)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_lasso_close.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show() 

def BVT_DT(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    #Bias-Variance Trade Off Decision Tree
    complexity = 20
    Loss = np.zeros(complexity)
    Variance = np.zeros(complexity)
    Bias = np.zeros(complexity)
    for c in range(1, complexity + 1):
        regr_leaf = DecisionTreeRegressor(max_depth=c).fit(X_train, y_train)
        print(f"Max depth: {c} | Number of leaves: {regr_leaf.get_n_leaves()} | Depth: {regr_leaf.get_depth()}")

        regr = DecisionTreeRegressor(max_depth=c)
        #Calculates the Bias, Variance and Loss with nb bootstrap cycles
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train, y_train, X_test, y_test, loss='mse', num_rounds = nb)

        Loss[c-1] = avg_expected_loss
        Variance[c-1] = avg_var
        Bias[c-1] = avg_bias
    
    if plot:
        plt.plot(range(complexity), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(complexity), Variance, 'o-', label = "Variance")
        plt.plot(range(complexity), Loss, 'o-', label = "Loss")
        plt.xlabel("Max depth")
        #plt.ylabel("Error")
        plt.title("Decision Tree Bias-Variance Trade Off")
        #plt.ylim(0,0.05)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_dt.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show() 


        plt.plot(range(complexity), Bias, 'o-', label = "Bias$^2$")
        plt.plot(range(complexity), Variance, 'o-', label = "Variance")
        plt.plot(range(complexity), Loss, 'o-', label = "Loss")
        plt.xlabel("Max depth")
        #plt.ylabel("Error")
        plt.title("Decision Tree Bias-Variance Trade Off")
        plt.ylim(0,0.05)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_dt_close.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show() 

def BVT_NN(X_train, X_test, y_train, y_test, nb = 100, plot = False):
    #Bias-Variance Trade Off Neural Network
    complexity = 11
    n = 20
    Loss = np.zeros(n)
    Variance = np.zeros(n)
    Bias = np.zeros(n)
    neurons = np.logspace(0,complexity-1, n, base = 2).astype(int)
    for c in tqdm(range(n)):

        regr = MLPRegressor(learning_rate_init=0.1,hidden_layer_sizes=(neurons[c],neurons[c]) , max_iter=10000)

        #Calculates the Bias, Variance and Loss with nb bootstrap cycles
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            regr, X_train, y_train, X_test, y_test, loss='mse', num_rounds = nb)

        Loss[c] = avg_expected_loss
        Variance[c] = avg_var
        Bias[c] = avg_bias
    
    if plot:
        plt.plot(np.log2(neurons), Bias, 'o-', label = "Bias$^2$")
        plt.plot(np.log2(neurons), Variance, 'o-', label = "Variance")
        plt.plot(np.log2(neurons), Loss, 'o-', label = "Loss")
        plt.xlabel("log2(neurons per hidden layer)")
        plt.ylabel("Error")
        plt.title("Neural Network Bias-Variance Trade Off")
        plt.ylim(0,0.2)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_nn_.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show()
        
        plt.plot(np.log2(neurons), Bias, 'o-', label = "Bias$^2$")
        plt.plot(np.log2(neurons), Variance, 'o-', label = "Variance")
        plt.plot(np.log2(neurons), Loss, 'o-', label = "Loss")
        plt.xlabel("log2(neurons per hidden layer)")
        plt.ylabel("Error")
        plt.title("Neural Network Bias-Variance Trade Off")
        plt.ylim(0,0.025)
        #plt.xlim(0,40)
        plt.legend()
        plt.savefig('bv_tradeoff_nn_close_.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show() 


def BVT_regterm(X_train, X_test, y_train, y_test, degrees = [20], alphas = [0.01,0.001], nb = 100, plot = False, method = "Ridge"):
    ##Bias-Variance Trade Off with L2 penalty term for Ridge or Lasso

    Loss = np.zeros((len(degrees), len(alphas)))
    Variance = np.zeros((len(degrees), len(alphas)))
    Bias = np.zeros((len(degrees), len(alphas)))

    for j,a in tqdm(enumerate(alphas)):
        for i,d in enumerate(degrees):
            c = int((d+1)*(d+2)/2)
            if method.lower() == "ridge":
                regr = Ridge(alpha = a)
            if method.lower() == "lasso":
                regr = Lasso(alpha = a, max_iter = 5000, tol = 1e-3)

            #Calculates the Bias, Variance and Loss with nb bootstrap cycles
            avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
                regr, X_train[:,1:c], y_train, X_test[:,1:c], y_test, loss='mse', num_rounds = nb)

            Loss[i,j] = avg_expected_loss
            Variance[i,j] = avg_var
            Bias[i,j] = avg_bias
    
    if plot:
        for i,d in enumerate(degrees):
            plt.plot(np.log10(alphas), Bias[i], 'o-', label = f"Bias$^2$ | Degree = {d}")
        for i,d in enumerate(degrees):
            plt.plot(np.log10(alphas), Variance[i], 'o-', label = f"Variance | Degree = {d}")
        
        plt.xlabel(r"$\log_{10}$($\lambda$)")
        plt.ylabel("Error")
        plt.title(f"{method} Bias and Variance")
        #plt.xlim(0,40)
        plt.legend(bbox_to_anchor=(1, 0.7))
        plt.savefig(f'bvt_regterm_{method}.pdf', dpi = 400, bbox_inches = 'tight')
        plt.show()




if __name__ == '__main__':
    #Read data
    data = imread('SRTM_data_Norway_1.tif') #All data
    terrain = data[:50,-50:] #Subset
    Y = terrain.ravel() #1d array of subset
    dim = terrain.shape
    x1,x2 = np.meshgrid(range(dim[0]), range(dim[1]))
    X1 = x1.ravel().astype(np.float)
    X2 = x2.ravel().astype(np.float)

    #Create design matrix of 2D polynomials using X1 and X2
    X = create_X(X1, X2, 50)

    #Split into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

    #Scale data
    X_train, X_test = scale_data(X_train, X_test)
    Y_train, Y_test = scale_data(Y_train, Y_test)


    #BVT_OLS(X_train[:,:int(21*22/2)], X_test[:,:int(21*22/2)], Y_train, Y_test, nb=100, plot=True)
    #BVT_Ridge(X_train, X_test, Y_train, Y_test, nb=20, plot=True)   
    #BVT_DT(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb=100, plot=True)
    #BVT_Lasso(X_train, X_test, Y_train, Y_test, nb=10, plot=True)
    #BVT_NN(X_train[:,1:3], X_test[:,1:3], Y_train, Y_test, nb=10, plot=True)
    #BVT_regterm(X_train, X_test, Y_train, Y_test, degrees = [20,30,40], alphas = np.logspace(-5,0,6), nb=10, plot=True)
    #BVT_regterm(X_train, X_test, Y_train, Y_test, degrees = [20,30,40], alphas = np.logspace(-5,-1,5), method = "Lasso", nb=10, plot=True)
    

    labelsize = 21
    ticksize = 18
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x1,x2,terrain)
    ax.view_init(20,-20)
    ax.set_title("Norwegian Terrain")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel('Altitude', fontsize = labelsize, rotation=60)
    ax.xaxis.set_tick_params(labelsize= ticksize, pad=-5)
    ax.yaxis.set_tick_params(labelsize= ticksize, pad=-5)
    ax.zaxis.set_tick_params(labelsize= ticksize, pad=-5)
    #plt.savefig("NorwegianTerrain.pdf", dpi = 400, bbox_inches = "tight")
    plt.show()

    
