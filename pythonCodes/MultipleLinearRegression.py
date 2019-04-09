# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:04:20 2019

@author: Berkay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('team.csv', encoding = 'ISO-8859-1')

col_age = dataset.iloc[:,4].values
col_exp = dataset.iloc[:,6].values
col_pow = dataset.iloc[:,7].values
col_sal = dataset.iloc[:,8].values
col_three = dataset.iloc[:, [4,6,7]].values
col_three = np.append(arr = np.ones((18,1)).astype(int), values = col_three, axis = 1)
#col_ones = np.ones((18,))
#X = np.vstack((col_ones, col_age, col_exp, col_pow)).T

Y = col_sal.T

coefficients = np.linalg.inv(np.dot(col_three.T, col_three))
coefficients = np.dot(coefficients, col_three.T)
coefficients = np.dot(coefficients, col_sal)

Y_hat = np.dot(col_three, coefficients)

#coefficients[0]
#coefficients[1]
#coefficients[2]
#coefficients[3]

plt.title("Residual Error Plot")
plt.scatter(Y_hat, Y_hat-col_sal, color = 'red')
plt.hlines(y = 0, xmin = 0, xmax = 20000, linewidth = 2, color = 'blue')
plt.show()







































