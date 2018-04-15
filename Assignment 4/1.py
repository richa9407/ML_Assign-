#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:44:18 2018

@author: richa
"""
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('/Users/savita/desktop/ThoraricSurgery.csv')

from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
dataset["F.7"] = lb_make.fit_transform(dataset["F.7"])
dataset["F"] = lb_make.fit_transform(dataset["F"])
dataset["F.1"] = lb_make.fit_transform(dataset["F.1"])
dataset["F.2"] = lb_make.fit_transform(dataset["F.2"])
dataset["F.3"] = lb_make.fit_transform(dataset["F.3"])
dataset["F.4"] = lb_make.fit_transform(dataset["F.4"])
dataset["F.5"] = lb_make.fit_transform(dataset["F.5"])
dataset["F.6"] = lb_make.fit_transform(dataset["F.6"])
dataset["T"] = lb_make.fit_transform(dataset["T"])
dataset["T.1"] = lb_make.fit_transform(dataset["T.1"])
dataset["T.2"] = lb_make.fit_transform(dataset["T.2"])
dataset["OC14"] = lb_make.fit_transform(dataset["OC14"])
dataset["DGN2"] = lb_make.fit_transform(dataset["DGN2"])
dataset["PRZ1"] = lb_make.fit_transform(dataset["PRZ1"])

x = dataset.iloc[ : , :-1].values
y = dataset.iloc[: , 16].values

from sklearn.cross_validation import train_test_split 
x_train , x_test , y_train , y_test = train_test_split( x, y , test_size= 0.25 , random_state=0 )

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train= sc.fit_transform(x_train)
x_test = sc. fit_transform(x_test)

from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train , y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred )

