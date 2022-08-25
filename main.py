from operator import truediv
from random import random
import warnings
import numpy as np
import pandas as pd 

# ignore warnings 
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

def initialize_dataset():
    from sklearn.datasets import load_boston

    _load_boston = load_boston()
    xs = _load_boston.data
    ys = _load_boston.target

    data = pd.DataFrame(xs, columns=_load_boston.feature_names)
    # drop some columns remains 2 columns == 2 theta
    data = data.drop(['CRIM', 'ZN', 'DIS', 'LSTAT', 'INDUS', 'CHAS', 'NOX', 'RM', 'RAD', 'PTRATIO', 'B'], axis=1)
    xs = np.copy(data)
    data['SalePrice'] = ys

    return xs, ys

def hypothesis(theta_0, theta, x):
    x_theta = np.dot(theta.transpose(), x)
    return theta_0 + x_theta.sum()

def calc_gradient_0(theta_0, theta, xs, ys):
    grad = 0
    for i in range(len(xs)):
        grad = grad + (hypothesis(theta_0, theta, xs[i]) - ys[i])
    return grad / len(xs)

def calc_gradient_i(theta_0, theta, xs, ys, j):
    grad = 0
    for i in range(len(xs)):
            grad = grad + (hypothesis(theta_0, theta, xs[i]) - ys[i]) * xs[i][j]
    return grad / len(xs)

def compute_cost(theta_0, theta, xs, ys):
    cost = 0
    for i in range(len(xs)):
        error = hypothesis(theta_0, theta, xs[i]) - ys[i]
        cost = cost + error * error
    return cost / (2 * len(xs))

def is_not_small_enough(a, b):
    if abs(a) > 0.0005:
        return True
    
    for B in b:
        if (abs(B) > 0.0005):
            return True
    
    return False

if __name__ == "__main__":
    X, Y = initialize_dataset()

    alpha = 0.00001 # learning rate

    theta_0 = 2 * random() - 1
    theta = np.random.rand(2)
    for theta_i in theta:
        theta_i = 2 * random() - 1

    gradient_0 = calc_gradient_0(theta_0, theta, X, Y)
    gradient = np.zeros(2)
    for i in range(len(gradient)):
        gradient[i] = calc_gradient_i(theta_0, theta, X, Y, i)

    cycle = 1 
    while is_not_small_enough(gradient_0, gradient):
        print('=====CYCLE %d=====' % cycle)
        print("Theta 0: {}, Theta 1: {}, Theta 2: {}".format(theta_0, theta[0], theta[1]))
        print("Gradient 0: {}, Gradient 1: {}, Gradient 2: {}".format(gradient_0, gradient[0], gradient[1]))
        print("Cost: {}".format(compute_cost(theta_0, theta, X, Y)))

        cycle = cycle + 1

        theta_0 = theta_0 - alpha * gradient_0
        theta = theta - alpha * gradient
        gradient_0 = calc_gradient_0(theta_0, theta, X, Y)
        for i in range(len(gradient)):
            gradient[i] = calc_gradient_i(theta_0, theta, X, Y, i)

    print("------------------------")
    print("FOUND THETA 0: {}, THETA 1: {}, THETA 2: {} IS OPTIMUM".format(theta_0,theta[0], theta[1]))
    print("COST: {}".format(compute_cost(theta_0, theta, X, Y)))

#     =====CYCLE 631468=====
# Theta 0: 17.665405611242054, Theta 1: 0.05244614926070475, Theta 2: -0.001701217481131102
# Gradient 0: -1.9654113928291355, Gradient 1: 0.012860517703949841, Gradient 2: 0.0021101252933546743
# Cost: 48.93082392686347