import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from sympy import cos, sin


def loss_function(_w1, _w2, x1=1.0, x2=1.5, y_tar=2.0):
    return (1 / 2) * ((np.sin(_w1 * x1) + np.cos(_w2 * x2) + _w2 - y_tar) ** 2)


def viz(current_w1, current_w2, current_loss):
    x = np.linspace(-10, 10, 100)  # Der fuer diese Aufgabe interessante Definitionsbereich liegt zwischen -10 und 10
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    # ax.plot(current_w1, current_w2, Z, marker="o",
    #         markersize=10, markeredgecolor="black", markerfacecolor="white") # TODO fixen
    ax.set_title("Fehlergebirge", fontsize=13)
    ax.set_xlabel('w1', fontsize=11)
    ax.set_ylabel('w2', fontsize=11)
    ax.set_zlabel('E', fontsize=11)
    plt.show()


def derivative(c):
    _w1, _w2, x1, x2, y_tar = sym.symbols('_w1 _w2 x1 x2 y_tar')
    function = (1 / 2) * ((sin(_w1 * x1) + cos(_w2 * x2) + _w2 - y_tar) ** 2)
    return sym.diff(function, sym.symbols(c))


if __name__ == '__main__':
    w1 = -6.5
    w2 = -9.5
    alpha = 0.05

    for x in range(100):
        viz(current_w1=w1, current_w2=w2, current_loss=0.0)

        w1 = w1 - alpha * derivative('_w1')
        w2 = w2 - alpha * derivative('_w2')
