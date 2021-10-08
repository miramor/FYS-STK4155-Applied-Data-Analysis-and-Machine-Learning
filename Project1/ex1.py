import numpy as np
from sklearn.model_selection import train_test_split
import random
import importlib
import functions
importlib.reload(functions)
from functions import *

np.random.seed(2405) # Set a random seed

N = 500 #Number of datapoints
x = np.random.uniform(0, 1, N) #Randomly creates N x's
y = np.random.uniform(0, 1, N) #Randomly creates N y's

z = FrankeFunction(x, y) #Find the true FrankeFunction values to the given x and y

complex = 13 #Choose the highest complexity we will look at

X = create_X(x,y,complex) #Create the design matrix X

test_train_l = train_test_split(X,z,test_size=0.2) #Split the data into training and test sets

#Exercise 1
print(f"OLS: {evaluate_method(ols, test_train_l, scale = False, d = 5)}") #Evaluate the model for complexity 5 without scaling the data
print(f"OLS: {evaluate_method(ols, test_train_l, scale = True, d = 5)}") #Evaluate the model for complexity 5 with scaling the data

noise = np.random.normal(0, 1, size=(z.shape)) #Make random noise
z_noisy = FrankeFunction(x, y) + noise*0.1 #Add noise to the z values of the FrankeFunction

test_train_l_noise = train_test_split(X,z_noisy,test_size=0.2) #Split the noisy data into test and train sets
print(f"OLS with noise: {evaluate_method(ols, test_train_l_noise, scale = False, d = 5)}") #Evaluate the noisy data with degree = 5
print(f"OLS with noise: {evaluate_method(ols, test_train_l_noise, scale = True, d = 5)}") #Evaluate the noisy data without degree = 5

#Calculate the confidence intervals for all beta values
variance_beta = var_beta(test_train_l_noise[0]) #Find the variance of the beta values
beta_l = ols(test_train_l_noise[0], test_train_l_noise[2]) #Calculate the optimal beta values for the given dataset
confidence_interval = ci(beta_l, variance_beta, N) #Find the confidence interval

beta_sd_l = variance_beta*(1.96/np.sqrt(N)) #Find the range of each CI

#Set label and tick size
labelsize=21
ticksize = 19

#Plot the CI's
plt.errorbar(range(len(beta_l)), np.log(abs(beta_l)), np.log(beta_sd_l), linestyle='None', marker = 'o', ecolor = 'red')
plt.title(r"The logarithmic absolute values of $\beta$ and the logarithmic standard deviation", fontsize=labelsize)
plt.xlabel(r"$\beta_i$", fontsize=labelsize)
plt.ylabel(r"log(abs($\beta$))", fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.grid()
plt.show()
