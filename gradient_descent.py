import math
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import params_update


def np_loss_function(w, b, training_data):
    return 1 / len(training_data) * sum(map(lambda data: np.power(data['y'] - w * data['x'] - b, 2), training_data))


def loss_function(w, b, training_data):
    return 1 / len(training_data) * sum(map(lambda data: math.pow(data['y'] - w * data['x'] - b, 2), training_data))


if __name__ == '__main__':

    start = -100
    end = 100
    x_list = random.sample(range(start, end), k=100)
    w_true = 5
    b_true = 135
    training_data = list(map(lambda x: {'x': x, 'y': w_true * x + b_true}, x_list))

    W = np.arange(-500, 500, 0.5)
    B = np.arange(-500, 500, 0.5)
    W, B = np.meshgrid(W, B)
    L = np_loss_function(W, B, training_data)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(W, B, L, cstride=50, rstride=50)

    generator = params_update.ParametersGenerator(500, 500, 25, training_data)
    iterator = iter(generator)
    w_list = []
    b_list = []
    for i in range(1000):
        w, b = next(iterator)
        # print(w,b)
        w_list.append(w)
        b_list.append(b)

    w_np_arr = np.array(w_list)
    b_np_arr = np.array(b_list)
    ax.scatter(w_np_arr, b_np_arr, np_loss_function(w_np_arr, b_np_arr, training_data), color='r', s=20)
    ax.scatter(np.array([w_true]), np.array([b_true]), loss_function(w_true, b_true, training_data), color='y', s=150)

    plt.show()
