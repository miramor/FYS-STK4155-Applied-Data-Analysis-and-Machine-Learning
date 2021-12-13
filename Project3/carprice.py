import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams["figure.figsize"]=12,12

cars = pd.read_csv('./CarPrice_Assignment.csv')
CompanyName = cars["CarName"].apply(lambda x: x.split(" ")[0])

#Insert into dataframe
cars.insert(3, "CompanyName", CompanyName)
cars.drop(["CarName"], axis=1, inplace = True)
cars.drop(['car_ID'],axis=1,inplace=True)

#Check for spelling errors in CompanyName
cars.CompanyName.unique()

#Correct the spelling errors
cars = cars.replace(to_replace = "maxda", value = "mazda")
cars = cars.replace(to_replace = "Nissan", value = "nissan")
cars = cars.replace(to_replace = "porcshce", value = "porsche")
cars = cars.replace(to_replace = "toyouta", value = "toyota")
cars = cars.replace(to_replace = "vokswagen", value = "volkswagen")
cars = cars.replace(to_replace = "vw", value = "volkswagen")

#Look at correlation
sns.heatmap(cars.corr(),cmap="OrRd",annot=True)
plt.show()


#Keep only variables with high correlation to price
cars = cars.drop(["peakrpm", "compressionratio", "stroke", "carheight", "symboling"],axis=1)

#Look at correlation between some other variables
vars1 = ['wheelbase', 'carlength', 'carwidth','curbweight']
vars2 = ['citympg','highwaympg']
vars3 = ['enginesize','boreratio','horsepower']
sns.heatmap(cars.filter(vars1).corr(),cmap="OrRd",annot=True)
plt.show()
sns.heatmap(cars.filter(vars2).corr(),cmap="OrRd",annot=True)
plt.show()
sns.heatmap(cars.filter(vars3).corr(),cmap="OrRd",annot=True)
plt.show()

#We only need one of those variables that are highly correlated
cars.drop(["citympg"], axis=1, inplace = True)
cars.drop(['wheelbase'],axis=1,inplace=True)
cars.drop(['carlength'],axis=1,inplace=True)
cars.drop(['carwidth'],axis=1,inplace=True)
cars.drop(['horsepower'],axis=1,inplace=True)

cars = pd.get_dummies(cars)


#Keeping only Buick
remove = ['CompanyName_alfa-romero', 'CompanyName_audi', 'CompanyName_bmw','CompanyName_chevrolet', 'CompanyName_dodge',
       'CompanyName_honda', 'CompanyName_isuzu', 'CompanyName_jaguar',
       'CompanyName_mazda', 'CompanyName_mercury', 'CompanyName_mitsubishi',
       'CompanyName_nissan', 'CompanyName_peugeot', 'CompanyName_plymouth',
       'CompanyName_porsche', 'CompanyName_renault', 'CompanyName_saab',
       'CompanyName_subaru', 'CompanyName_toyota', 'CompanyName_volkswagen',
       'CompanyName_volvo',]
cars.drop(remove, axis = 1, inplace = True)


#Keeping only fuelsystem with high correlation to price
remove = ['fuelsystem_1bbl', 'fuelsystem_4bbl',
       'fuelsystem_idi', 'fuelsystem_mfi',
       'fuelsystem_spdi', 'fuelsystem_spfi']
cars.drop(remove, axis = 1, inplace = True)

#Remove all engine types, none with >0.5 correlation to price
remove = ['enginetype_dohc', 'enginetype_dohcv', 'enginetype_l', 'enginetype_ohc', 'enginetype_ohcf',
       'enginetype_ohcv', 'enginetype_rotor']
cars.drop(remove, axis = 1, inplace = True)

#Remove all cylinders, none with >0.5 correlation to price
remove = ['cylindernumber_eight', 'cylindernumber_five',
       'cylindernumber_four', 'cylindernumber_six', 'cylindernumber_three',
       'cylindernumber_twelve', 'cylindernumber_two']
cars.drop(remove, axis = 1, inplace = True)

#Remove all car types as well
remove = ['carbody_convertible', 'carbody_hardtop',
       'carbody_hatchback', 'carbody_sedan', 'carbody_wagon']
cars.drop(remove, axis = 1, inplace = True)

#Removing the rest of the variables without high corr to price
remove = ['fueltype_diesel', 'fueltype_gas',
       'aspiration_std', 'aspiration_turbo', 'doornumber_four',
       'doornumber_two', 'drivewheel_4wd','enginelocation_front', 'enginelocation_rear',]
cars.drop(remove, axis = 1, inplace = True)

sns.heatmap(cars.corr(),cmap="OrRd",annot=True)
plt.show()


#Make dataframes for non linear relationships
nonlin = ['curbweight', 'enginesize', 'boreratio', 'highwaympg']

cars2 = cars.copy()
for feat in nonlin:
    cars2.insert(4,feat + "2", cars[feat]**2)

cars3 = cars2.copy()
for feat in nonlin:
    cars3.insert(4,feat + "3", cars[feat]**3)


def scale_data(df_train, df_test):
    #Columns in df
    columns = []
    for i in df_train:
        columns.append(i)

    #Columns to scale
    col_to_scale = columns[:-5]

    scaler = StandardScaler()
    scaler.fit(df_train[col_to_scale])
    train_scaler = scaler.transform(df_train.loc[:,col_to_scale])
    df_train_scaled = df_train.copy()
    df_train_scaled[col_to_scale] = train_scaler

    test_scaler = scaler.transform(df_test.loc[:,col_to_scale])
    df_test_scaled = df_test.copy()
    df_test_scaled[col_to_scale] = test_scaler

    return df_train_scaled, df_test_scaled


#Split into train and test and scale
cars_train, cars_test = train_test_split(cars, test_size = 0.3)
cars_train_scaled, cars_test_scaled = scale_data(cars_train, cars_test)

cars2_train, cars2_test = train_test_split(cars2, test_size = 0.3)
cars2_train_scaled, cars2_test_scaled = scale_data(cars2_train, cars2_test)

cars3_train, cars3_test = train_test_split(cars3, test_size = 0.3)
cars3_train_scaled, cars3_test_scaled = scale_data(cars3_train, cars3_test)

