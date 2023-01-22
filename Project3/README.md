### Water Quality Predicition

In this project we predict the water-potabilty given the data [Water-Potability](https://www.kaggle.com/adityakadiwal/water-potability).
The data is split into two categories, the water is either potable (1) or non-potable (0). Hence, we have a binary classification problem. Additionally, the data set consits of nine features: pH, hardness [mg/L], solids [ppm], chloramines [ppm], sulfate [mg/L], conductivity [&micro;S/cm], organic carbon [ppm], trihalmoethanes [&micro;g/L], and turbidity [NTU].

There are a total of 3276 samples in this set, but 1265 of these samples have at least one missing feature value.
We have, for this project, decided to just remove these values since we will still have a total of 2011 samples.
Predictions are done using a feed forward neural network, decision trees and random forest with bagging and boosting.

**How to use**: Download the dataset here: [Water-Potability](https://www.kaggle.com/adityakadiwal/water-potability).  
Running `DataAnalysis.py` will show a pie chart of the potable water distribution, a histogram, a correlation matrix and three box plots for the nine features.

### Feed Forward Neural Network
#### Self-implemented Gridsearch
In `NNOwn.py`, a self-implemented grid search is performed using MLPClassifier.
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
Simple write 
```python
NNOwn.py
```
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
Simpy write 
```python
NNGS.py
```
By default only Test 1 will be run. To run Test 2, change Test 2 = False to True.
Similarly, change boolean for the respective test to run any one of the other tests.
Some of the tests might take a while.

### Decision tree, Random forest, Bagging, and Boosting

**How to use**:
The following packages must be installed: sklearn, pandas, numpy, matplotlib, and seaborn.\
Simpy write 
```python
decision_tree_mm.py
```
This will show first a plot of four different decision tree and random forrest methods as a function of the maximum depth. It will also show four heat maps for a grid search for random forest (Gini and entropy), bagging, and boosting.\
Finaly it will show a confusion matrix for the best hyperparameters using the validation data and a confusion matrix with bagging for the test data.

