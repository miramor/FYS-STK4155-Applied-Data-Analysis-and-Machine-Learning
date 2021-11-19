# FYS-STK4155 Applied Data Analysis and Machine Learning.  Filer til undervisning og oppgaver. Prosjekter.

## Projects
### Project 1: Regression Analysis and Resampling Methods
In this project we run Ordinary Least Squares, Ridge Regression and Lasso Regression on the Franke Function and Terrain Data. 
We have one file "functions.py" containing all the functions used in this project. We also have a separate file for each exercise. These are named "ex1.py", "ex2.py" etc.. 
The folder UploadedFigures contains all the plots created when running the exercises.

**How to use:**
By simply running the files for each exercise, all of the results in our report are reproduced.

### Project 2: Classification and Regression, from Linear and Logistic Regression to Neural Networks
In this project we have made our own implementation of stochastic gradient descent (SGD) and feed forward neural network using SGD.
We have one file "functions.py" containing a handful of the functions used in this project. 
There is a seperate file for most of the exercises. There are two classes for the neural network code,
one for regression and one for classification.

**How to use:**
By simply running the files for each exercise, some of the results in our report are reproduced when running the files.
#### Stochastic gradient descent for predicting values of the Franke function
Running "a_sgd.py" will show results from three grid searches. One with OLS cost function and varying learning rate and number of mini batches. Two with ridge cost function, varying first learning rate and number of mini batches, and then learning rate and normalization parameter. Some results using SKlearn's SGD are printed as well.
#### Neural Network
##### Regression predicting values of the Franke function
Running "bc_NNregression.py" will show two grid searches where MSE and R2 is presented for both training and test set.
One grid search is with learning rate and number of neurons per layer, and one is with learning rate and regularization parameter.
A lot of plots with MSE as a function of epochs will be shown too, to see the stability performance for different values of hyperparameters.
##### Classification using Wisconsin breast cancer data
Running "d_NNclas.py" presents results from three grid searches when ran.
The prediction accuracy on training data as a function of epochs is also shown for one of the grid searches.
A confusion matrix with results from SKlearns prediction using SKlearn's neural network is also shown.

