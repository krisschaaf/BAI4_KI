from src.classes.Node import *


def heuristic(node) -> int:
    [[y, x]] = np.argwhere(node.state == 1)

    currentValue: int = 1

    while nextValueFound(node.state, y, x, currentValue):
        currentValue += 1
        [[y, x]] = np.argwhere(node.state == currentValue)

    return 15 - currentValue
