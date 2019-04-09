import csv

import numpy as np

import matplotlib.pyplot as plt



def estimate_coef(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    SS_xy = np.sum((x-m_x) * (y-m_y))
    SS_xx = np.sum((x-m_x) * (x-m_x))
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return(b_0, b_1)



def plotLine(x, y, b):
    plt.scatter(x, y, color="black",
                marker="o", s=30)
    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color="orange")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



with open("team.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    x =np.array([])
    y= np.array([])



    for row in csv_reader:
            if line_count == 0:
                line_count = +1
            else:
              line_count += 1
              x = np.append(x ,int(row[4]))
              y= np.append(y ,int(row[6]))

b = estimate_coef(x, y)
print("Estimated coefficients:\nb_0 = {}  \\nb_1 = {}".format(b[0], b[1]))

plotLine(x, y, b)
print(x)
print(y)
