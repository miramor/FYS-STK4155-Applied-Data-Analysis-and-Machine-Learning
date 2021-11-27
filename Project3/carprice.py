import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


cars = pd.read_csv('./CarPrice_Assignment.csv')
print(cars["CarName"])

y = cars["price"]
X = cars.loc[:,cars.columns != "price"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
print(cars.info())
CompanyName = data["CarName"].apply(lambda x: x.split(" ")[0])



regr = RandomForestRegressor(max_depth=1, random_state=0)
regr.fit(X, y)

#print(cars["CarName"].unique())
