import matplotlib.pyplot as plt
import numpy as np
from sympy import cos, sin


def loss_function(_w1, _w2, _x1, _x2, _y_tar):
    # Fehlerfunktion E aus Aufgabenzettel (Mittlere quadratische Abweichung/Fehler)
    sinval = np.sin(_w1 * _x1)
    cosval = np.cos(_w2 * _x2)

    return (1 / 2) * (sinval + cosval + _w2 - _y_tar) ** 2


def visualize(current_w1, current_w2, _x1, _x2, _y_tar, title):
    _x = np.linspace(-10, 10, 100)  # Der fuer diese Aufgabe interessante Definitionsbereich liegt zwischen -10 und 10
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(_x, y)

    # X und Y auf Z der abbilden -> Fehlerfunktion wird visualiziert.
    Z = loss_function(X, Y, _x1, _x2, _y_tar)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')  # Fehlerfunktion im 3D Raum darstellen

    current_loss = loss_function(float(current_w1), float(current_w2), _x1, _x2, _y_tar)
    ax.plot(current_w1, current_w2, current_loss, marker="o",
            markersize=10, markeredgecolor="black", markerfacecolor="white")  #

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('w1', fontsize=11)
    ax.set_ylabel('w2', fontsize=11)
    ax.set_zlabel('E', fontsize=11)

    plt.show()

    return current_w1, current_w2, current_loss


if __name__ == '__main__':

    # # Datensatz 1
    # w1 = -6.5
    # w2 = -9.5
    # alpha = 0.05

    # Datensatz 2
    w1 = 0.0
    w2 = -0.5
    alpha = 0.05

    w1_start = w1
    w2_start = w2

    # konstant, definiert durch Datensample aus Aufgabe
    x1 = 1.0
    x2 = 1.5
    y_tar = 2.0

    visualize(current_w1=w1, current_w2=w2, _x1=x1, _x2=x2, _y_tar=y_tar, title="Fehlergebirge - Start")

    # Schrittweise Näherung des/eines Minimums
    for x in range(1000):
        # w1 = w1 - alpha * (x1 * (w2 - y_tar + sin(w1 * x1) + cos(w2 * x2)) * cos(w1 * x1))
        w1 = w1 - alpha *(x1 * cos(x1 * w1) * (sin(x1 * w1) - y_tar + cos(w2 * x2) + w2))
        # w2 = w2 - alpha * (0.5 * (-2 * x2 * sin(w2 * x2) + 2) * (w2 - y_tar + sin(w1 * x1) + cos(w2 * x2)))
        w2 = w2 - alpha * (cos(x2 * w2) + w2 - y_tar + sin(w1 * x1)) * (1 - x2 * sin(x2 * w2))

    print('w1 Start: ' + str(w1_start) + '; w1 Ende: ' + str(round(w1, 1)))
    print('w2 Start: ' + str(w2_start) + '; w2 Ende: ' + str(round(w2, 1)))

    current_w1, current_w2, current_loss = visualize(current_w1=w1, current_w2=w2, _x1=x1, _x2=x2, _y_tar=y_tar, title="Fehlergebirge - Ende")
    print('x: ' + str(round(current_w1, 1)) + ' y: ' + str(round(current_w2, 1)) + ' z: ' + str(round(current_loss, 1)))

