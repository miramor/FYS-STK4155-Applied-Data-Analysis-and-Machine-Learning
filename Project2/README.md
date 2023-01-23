### Classification and Regression, from Linear and Logistic Regression to Neural Networks
In this project we have made our own implementation of stochastic gradient descent (SGD), feed forward neural network and logsitic reggresion, both of which use SGD.

We have one file `functions.py` containing a handful of the functions used in this project.
There is a seperate file for most of the exercises. There are two classes for the neural network code,
one for regression and one for classification.

**How to use:**
By simply running the files for each exercise, results in our report are reproduced. For example type the following in a Linux/Unix command line

```python
python3 a_sgd.py
```
A summary for each file is given below.

#### Stochastic gradient descent for predicting values of the Franke function

Running `a_sgd.py` will show results from three grid searches. One with OLS cost function and varying learning rate and number of mini batches. Two with ridge cost function, varying first learning rate and number of mini batches, and then learning rate and normalization parameter. Some results using SKlearn's SGD are printed as well. \
Dependencies: sys, sklearn, random, numpy, matplotlib and importlib

#### Neural Network

##### Regression predicting values of the Franke function
Running `bc_NNregression.py` will show two grid searches where MSE and R2 is presented for both training and test set.
One grid search is with learning rate and number of neurons per layer, and one is with learning rate and regularization parameter.
A lot of plots with MSE as a function of epochs will be shown too, to see the stability performance for different values of hyperparameters.
\
Dependencies: sys, sklearn, random, numpy, matplotlib, imageio and importlib

##### Classification using Wisconsin breast cancer data
Running `d_NNclas.py` presents results from three grid searches when ran.
The prediction accuracy on training data as a function of epochs is also shown for one of the grid searches.
A confusion matrix with results from SKlearns prediction using SKlearn's neural network is also shown.
\
Dependencies: sklearn, random, numpy, matplotlib and importlib

#### Logistic regression using Wisconsin breast cancer data
Running `e_logreg.py` presents results from two grid searches when ran, one from own implementation and one from the sklearn implementation.
A confusion matrix with results from SKlearns prediction using SKlearn's SGDClassifier and own implementation is also shown.
\
Dependencies: sklearn, numpy, matplotlib, imageio and importlib