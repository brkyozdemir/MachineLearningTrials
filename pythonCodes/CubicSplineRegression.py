# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:05:28 2019

@author: Berkay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Grand-slams-men-2013.csv')
x = dataset['NPA.1']
y = dataset['UFE.2']

def cubicSplineRegression(x, y, knotValues):
    
    onesCol = np.ones((1, len(x)))
    x1 = x
    x2 = x**2
    x3 = x**3
    
    x4 = []
    for k in knotValues:
        res = []
        for x in np.nditer(x1):
            equ = x - k
            if (equ < 0):
                equ = 0
            res.append(equ)
        temp = np.array([res]).transpose()
        colx4 = temp**3
        x4.append(colx4)
        
   
    if len(x4) == 3:
        X = np.vstack((onesCol, x1, x2, x3, x4[0].T, x4[1].T, x4[2].T)).T
    if len(x4) == 2:
        X = np.vstack((onesCol, x1, x2, x3, x4[0].T, x4[1].T)).T
    if len(x4) == 1:
        X = np.vstack((onesCol, x1, x2, x3, x4[0].T)).T
        
    B = np.linalg.inv(np.dot(X.T, X))
    B = np.dot(B, X.T)
    B = np.dot(B, y)
    y_pred = np.dot(X,B)
    return y_pred

if __name__ == '__main__':
    
    knotValues1 = [10, 20, 30]
    knotValues2 = [15, 30]
    knotValues3 = [30]
    
    sortedIndex = x.argsort()
    x = x[sortedIndex]
    y = y[sortedIndex]
    
    results1 = cubicSplineRegression(x, y, knotValues1)
    results2 = cubicSplineRegression(x, y, knotValues2)
    results3 = cubicSplineRegression(x, y, knotValues3)
    
    plt.title('Cubic Spline Regression')
    plt.scatter(x, y)
    plt.plot(x, results1, color = 'green', label = '3 knots')
    plt.plot(x, results2, color = 'blue', label = '2 knots')
    plt.plot(x, results3, color = 'red', label = '1 knot')
    plt.xlabel('Net Points Attempted by Player 1')
    plt.ylabel('Unforced Errors Done by Player 2')
    plt.legend()
    plt.show()
            