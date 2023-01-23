### Regression Analysis and Resampling Methods
In this project we run Ordinary Least Squares, Ridge Regression and Lasso Regression on the Franke Function and Terrain Data.
We have one file functions.py containing all the functions used in this project. We also have a separate file for each exercise. These are named `ex1.py`, `ex2.py` etc..
The folder UploadedFigures contains all the plots created when running the exercises.

**How to use:**
By simply running the files for each exercise, all of the results in our report are reproduced.
For example type the following in a Linux/Unix command line
```python
python3 ex1.py
```
A summary for each exercise-file is given below. A more comprehensive explanation is given in the report.

Exercise 1:
Running ordinary least squares on the Franke Function printing MSE and R2 for train and test data respectively.\
Dependencies: sklearn, random, numpy and importlib

Exercise 2:
Running OLS on the Franke Function as a function of the complexity of the model and using bootstrap as the resampling method. 
Running OLS on the Franke Function for a given complexity and as a function of the size of the data set (number of datapoints).\
Dependencies: sklearn, random, numpy and importlib

Exercise 3:
Running OLS on the Franke Function as a function of the complexity of the model and using k-fold as a resampling technique.\
Dependencies: sklearn, random, numpy and importlib

Exercise 4:
Running Ridge on the Franke Function for different lambdas as a function of the complexty of the model.\
Dependencies: sklearn, random, numpy and importlib

Exercise 5:
Running Lasso regression on the Franke functionfor for different lambdas as a function of complexity of the model.\
Dependencies: sklearn, random, numpy and importlib

Exercise 6:
Running OLS, Ridge and Lasso on real terrain data.
Finding mean MSE using bootstrap.\
Dependencies: sklearn, random, numpy, imageio, matplotlib and importlib

Supplementary functions:
`functions.py`contains functions needed for exercise 1 to 6\
Dependencies: sklearn, random, numpy, matplotlib, imageio and importlib

