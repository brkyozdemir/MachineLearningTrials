# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:35:45 2019

@author: Berkay
"""

import csv
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('team_1.csv', encoding = 'ISO-8859-1')
test = pd.read_csv('team_2.csv', encoding = 'ISO-8859-1')

X_train = train.iloc[:, [4]].values
y_train = train.iloc[:, [6]].values

X_test = test.iloc[:, [4]].values
y_test = test.iloc[:, [6]].values

def estimate_coef(x,y):
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.sum((x - mean_x) * (y - mean_y))
    var = np.sum((x - mean_x) * (x - mean_x))
    B1 = cov / var
    B2 = mean_y - B1*mean_x
    return (B2,B1)

B_train = estimate_coef(X_train,y_train)
B_test = estimate_coef(X_test, y_test)

def plotPrediction(X,y,b):
    plt.scatter(X,y,s = 32, color = 'red', marker = 'o')
    y_pred = b[0] + b[1]*X
    plt.plot(X, y_pred, color = 'blue')
    plt.xlabel('X', color = 'red')
    plt.ylabel('Predicted y', color= 'blue')
    plt.show()
    
def RSS(X,y,b):
    rss = 0
    for i in range(len(y)):
        y_pred = b[0] + (b[1]*X[i])
        rss += (y[i] - y_pred)**2
    return rss
        

plt.figure(1)
plotPrediction(X_train, y_train, B_test)
plt.figure(2)
plotPrediction(X_test, y_test, B_train)

rss_train = RSS(X_train, y_train, B_test)
rss_test = RSS(X_test, y_test, B_train)

print("RSS of Train set is ", rss_train)
print("RSS of Test set is ", rss_test)














