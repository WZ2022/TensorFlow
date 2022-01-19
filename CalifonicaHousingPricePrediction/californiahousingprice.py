# -*- coding: utf-8 -*-
"""CaliforniaHousingPrice.ipynb

Scikit-Learn provides a package of built-in functions to fetch datasets commonly used by the machine learning community.

https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset

(a) Download the above dataset using sklearn.datasets.fetch_california_housing function.
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from math import sqrt, sin, cos, radians

housing = datasets.fetch_california_housing()

"""(b) From the downloaded dataset, examine the features and define a pandas DataFrame to hold the district records as rows. Add the target median housing price as a new column in the data frame. The shape of the data frame should be (20640, 9)."""

print(housing.DESCR)

housing_df = pd.DataFrame(data=housing.data, columns=housing.feature_names)

housing_df.head()

print(housing.target)

housing_df['MedianPrice'] = housing.target

housing_df.shape

housing_df

"""(c) Plot a histogram for each attribute in the data frame. Use bins=100."""

import matplotlib.pyplot as plt
housing_df.hist(figsize=(12, 12), bins=100)

"""(d) Plot a scatter diagram to show the location (latitude-longitude) of the districts in the records. Use the median housing price as a scale to color the points in the plot.

Seaborn
"""

import seaborn as sns
ax = sns.scatterplot(data=housing_df, x="Longitude", y="Latitude", hue="MedianPrice")
norm = plt.Normalize(housing_df['MedianPrice'].min(), housing_df['MedianPrice'].max())
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
ax.get_legend().remove()
ax.figure.colorbar(sm)
plt.show()

"""Matplotlib"""

plt.figure(figsize=(10, 8))
#plot the data with housing price as scale to color the points
plt.scatter(housing_df['Longitude'],housing_df['Latitude'],
            cmap = 'autumn_r', c = housing_df['MedianPrice'],
            alpha = 0.8)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.show();

"""(e) Write a function to return the distance between two points where the function inputs are the latitude and longitude of the two points."""

# import math

# def degree_conversion(value):
#   return value*111/6378.137

# def distance(lat1, lat2, log1, log2):
#   lat1 = degree_conversion(lat1)
#   lat2 = degree_conversion(lat2)
#   log1 = degree_conversion(log1)
#   log2 = degree_conversion(log2)
#   𝑟 =6378.137
#   distance = 2*r* np.arcsin(math.sqrt(math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((log2-log1)/2)**2))
#   return distance

def distance(point1, point2):
  '''
  Calculate the distance between two points based on distance formula in https://en.wikipedia.org/wiki/Haversine_formula

  Outputs:
    distance : The distance(km) between two points
  '''
  point1_lat, point1_long = point1
  point2_lat, point2_long = point2
  #radius of the Earth
  r = 6378.137
  #convert the degree into radians
  phi_1, phi_2 = radians(point1_lat), radians(point2_lat)
  lambda_1, lambda_2 = radians(point1_long),radians(point2_long)
  #compute the inside square root
  inside_sqrt = sin((phi_2-phi_1)/2)**2 + cos(phi_1)*cos(phi_2)*sin((lambda_2 - lambda_1)/2)**2
  #put all pieces together
  distance = 2 * r * np.arcsin(np.sqrt(inside_sqrt))
  return distance

#An examination: distance from LA to SF
distance((33.93, -118.40), (37.62, -122.38))

"""(f) For each city given in figure 1, compute the distance of the California districts from the city using your function in (e)."""

#create a dataframe that contains the logitude and latitude of cities given in figure 1
data = {'Lat':[33.93, 37.62, 37.37, 32.57, 34.43],
        'Lon':[-118.40, -122.38, -121.92, -116.98, -119.83]}
cities_degree = pd.DataFrame(data, index=['LA', 'SF', 'SJ', 'SD', 'SB'])
# cities_degree = pd.DataFrame(data)
cities_degree

type(housing_df)

dis = dict({})
#use a for loop to generate a new dataframe 
# for i in range(len(housing_df)):
#   for j in range(len(cities_degree)):
#     dis.append(distance(housing_df.loc[i,'Latitude'], cities_degree.loc[j, 'Lat'],
#                         housing_df.loc[i, 'Longitude'], cities_degree.loc[j, 'Lon']))

for city in cities_degree.index:
  #get the city point from cities_degree
  city_point = tuple(cities_degree.loc[city, ['Lat', 'Lon']])
  #print(city_point)
  #apply distance on city_point and housing_df to calculate the distances
  housing_city_dis = housing_df.apply(lambda point: distance(city_point, (point.Latitude, point.Longitude)), axis=1)
  #print(housing_city_dis)
  #organize housing_city_dis into list, dis
  dis['Dist_'+city] = housing_city_dis

dis = pd.DataFrame(dis)
dis

"""(g) Now that you have (f), add a column (with label ‘CityProximity’) to the data frame in part (b) where each number in the column represents the shortest distance of the corresponding district to the given cities in figure 1."""

# shortest_dis = []
# temp = []
# while(dis!=[]):
#   for i in range(5):
#     temp.append(dis.pop(0))
#   shortest_dis.append(min(temp))

# shortest_dis[0:2]

housing_df['CityProximity']=dis.min(axis=1)

housing_df.head()

"""(h) From the data frame prepared in (g), create a new data frame where the median housing price is less than 5.0 and median house age is less than 52.0."""

housing_df1 = housing_df[housing_df['MedianPrice']<5]
housing_df1 = housing_df[housing_df['HouseAge']<52]

len(housing_df1)

"""(i) Split the data frame in (h) into a training and test dataset. Use the first 14000 rows as the training dataset and the rest as the test dataset. Split the target dataset in the same manner. (Note that the data frame in (h) has the target median housing price column which should be removed in the training set)"""

# X = housing_df1[{'AveBedrms', 'MedInc', 'HouseAge', 'AveRooms', 'Population', 
                      #  'AveOccup', 'Latitude', 'Longitude', 'CityProximity'}]
X = housing_df1.drop('median_housing_px',axis=1)
X_train = X[:14000]
X_train.shape

X_test = X[14000:]
X_test.shape

Y = housing_df1['MedianPrice']
Y_train = Y[:14000]
Y_train.shape

Y_test = Y[14000:]
Y_test.shape

"""(j) Use RandomForestRegressor (with 20 estimators) in Scikit-Learn to fit a random forest on the training dataset. Use cross_val_score to compute the average of the root mean squared error (RMSE) on the 5-fold cross-validation.

A standard way to evaluate model accuracy on continuous data is to compare the mean squared error (MSE) of your candidate models.
"""

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=20)
model.fit(X_train, Y_train)

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_root_mean_squared_error')
cv_scores.mean()

"""(k) Use GridSearchCV (with 5-fold cross-validation) to fine tune the hyper-parameters of the Random Forest Regressor model. (You may choose the upper bound for n_estimators to be 50)"""

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators':[10, 20, 30, 40, 50]}
gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
gs.fit(X_train, Y_train)

gs.best_params_

best_model = gs.best_estimator_

"""(l) Evaluate the RMSE of the fine-tuned model on the test dataset."""

best_model.fit(X_test, Y_test)

cv_scores = cross_val_score(best_model, X_test, Y_test, cv=5, scoring='neg_root_mean_squared_error')
cv_scores.mean()
