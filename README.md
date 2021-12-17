# FYS-STK4155 Applied Data Analysis and Machine Learning.  Filer til undervisning og oppgaver. Prosjekter.

## Projects
### Project 1: Regression Analysis and Resampling Methods
In this project we run Ordinary Least Squares, Ridge Regression and Lasso Regression on the Franke Function and Terrain Data.
We have one file "functions.py" containing all the functions used in this project. We also have a separate file for each exercise. These are named "ex1.py", "ex2.py" etc..
The folder UploadedFigures contains all the plots created when running the exercises.

**How to use:**
By simply running the files for each exercise, all of the results in our report are reproduced.

### Project 2: Classification and Regression, from Linear and Logistic Regression to Neural Networks
In this project we have made our own implementation of stochastic gradient descent (SGD), feed forward neural network and logsitic reggresion, both of which use SGD.
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

#### Logistic regression using Wisconsin breast cancer data
Running "e_logreg.py" presents results from two grid searches when ran, one from own implementation and one from the sklearn implementation.
A confusion matrix with results from SKlearns prediction using SKlearn's SGDClassifier and own implementation is also shown.


## Project 3: Water Quality Predicition

In this project we predict the water-potabilty given the data [Water-Potability](https://www.kaggle.com/adityakadiwal/water-potability).
The data is split into two categories, the water is either potable (1) or non-potable (0). Hence, we have a binary classification problem. Additionally, the data set consits of nine features: pH, hardness [mg/L], solids [ppm], chloramines [ppm], sulfate [mg/L], conductivity [&micro;S/cm], organic carbon [ppm], trihalmoethanes [&micro;g/L], and turbidity [NTU].

There are a total of 3276 samples in this set, but 1265 of these samples have at least one missing feature value.
We have, for this project, decided to just remove these values since we will still have a total of 2011 samples.
Predictions are done using a feed forward neural network, decision trees and random forest with bagging and boosting.

**How to use**: Download the dataset here: [Water-Potability](https://www.kaggle.com/adityakadiwal/water-potability).  
Running `DataAnalysis.py` will show a pie chart of the potable water distribution, a histogram, a correlation matrix and three box plots for the nine features.

### Feed Forward Neural Network
#### Self-implemented Gridsearch
In NNOwn.py, a self-implemented grid search is performed using MLPClassifier.
For each test, the mean accuracy is calculated using stratified kfold with k=5 folds and the results are displayed either in a heatmap or a simple graph.
The tests can be run consecutively, using the optimal parameters from the previous searches.\
Test 1: Run over learning rate and regularization parameters\
Test 2: Run over learning rate and epochs\
Test 3: Run over number of hidden layers and activation function\
Test 4: Run over epochs and batch size for different solvers

Finally, a prediction of the validation-accuracy is performed using the MLPClassifier with optimized parameters.
A confusion matrix and metric scores are also shown.
For plotting, some functions are imported from `functions1.py`.

**How to use**:
The following packages must be installed: sklearn, pandas, numpy, matplotlib and seaborn.\
Simple write `NNOwn.py` into the terminal.\
By default all test will be run.

#### Scikit-learn's Gridsearch
In NNGS.py, sklearn's grid search is used, utilizing either MLPClassifier or KerasClassifer.
Again, the mean accuracy is computed using cross validation with k=5 folds.
The gridsearch using MLP Classifier examines optimized exponential decay rates for the adam solver.
The gridsearch using the KerasClassifier consists in total of six test. The test can be run independently.\
Test 1: Running over different solvers (sgd, RMSprop, adagrad, adam)\
Test 2: Running over learning rate and epochs\
Test 3: Running over kernel initializer\
Test 4: Running over activation functions   
Test 5: Running over weigth constraint and droput rate\
Test 6: Running over neurons, epochs and batch size

Results for each paramter combination are printed to the terminal in latex-format.

**How to use**:
The following packages must be installed: sklearn, keras, pandas, numpy, matplotlib, os and seaborn.\
Simpy write `NNGS.py` into the terminal.\
By default only Test 1 will be run. To run Test 2, change Test 2 = False to True.
Similarly, change boolean for the respective test to run any one of the other tests.
Some of the tests might take a while.

### Decision tree, Random forest, Bagging, and Boosting

**How to use**:
The following packages must be installed: sklearn, pandas, numpy, matplotlib, and seaborn.\
Simpy write `decision_tree_mm.py` into the terminal.\
This will show first a plot of four different decision tree and random forrest methods as a function of the maximum depth. It will also show four heat maps for a grid search for random forest (Gini and entropy), bagging, and boosting.\
Finaly it will show a confusion matrix for the best hyperparameters using the validation data and a confusion matrix with bagging for the test data.

### Additional Exercise: Bias-Variance Tradeoff
In the additional exercise we did a Bias-Variance Tradeoff analysis on terrain data from Norway.
We looked at the trade of for three linear regression models (OLS, Ridge, Lasso), a feed forward neural network, and a decision tree.

The code used is found in "terrain.py" and "functions.py", whereas the first contains the
code with computation and plotting, and functions contains some functions used.
The data was obtained from https://earthexplorer.usgs.gov/, and it contains altitude of terrain. 
"SRTM_data_Norway_1.tif" is the file containing the data.
We looked at a small 50x50 area to reduce computation time.

**How to use**:
Running the file "terrain.py" will plot the surface the models are fitting.
Unncomment the different functions "BVT_\*\*\*" to see various bias-variance tradeoffs.
