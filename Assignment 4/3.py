#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:53:00 2018

@author: richa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, read_csv

file= r'/Users/savita/desktop/Concrete_Data.xls'
dataset  = pd.read_excel(file)

x = dataset.iloc[: , :-1].values
y = dataset.iloc[ : , 8].values

from sklearn.cross_validation import train_test_split 
x_train , x_test ,y_train, y_test = train_test_split( x , y , test_size =0.25 , random_state= 0)

from sklearn.linear_model import LinearRegression 
Regressor = LinearRegression()
Regressor.fit(x_train , y_train)


y_pred = Regressor.predict(x_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

