import numpy as np


ALLOWED_SPRINGER_STEPS = np.array([
    [-2, -1],
    [-2, +1],
    [-1, -2],
    [+1, -2],
    [-1, +2],
    [+1, +2],
    [+2, -1],
    [+2, +1]]
)

ALLOWED_BLANK_FIELD_STEPS = np.array([
    [0, -1],
    [0, +1],
    [-1, 0],
    [+1, 0]
])


class Node:
    def __init__(self, state, childNodes=None, cost=0):
        if childNodes is None:
            childNodes = []
        self.state = state
        self.childNodes = childNodes
        self.cost = cost


def isTargetState(state: list) -> bool:
    [[y, x]] = np.argwhere(state == 1)

    currentValue: int = 1

    while nextValueFound(state, y, x, currentValue):
        currentValue += 1
        [[y, x]] = np.argwhere(state == currentValue)

        if currentValue == 15:
            return True

    return False


def nextValueFound(state: list, y: int, x: int, currentValue: int) -> bool:
    for allowedSpringerStep in ALLOWED_SPRINGER_STEPS:
        newX: int = x + allowedSpringerStep[1]
        newY: int = y + allowedSpringerStep[0]

        if not (newY > 3 or newY < 0 or newX > 3 or newX < 0):
            intValue: int = state[newY][newX]
            if intValue == currentValue + 1:
                return True

    return False


def generateChildNodes(node: Node) -> list:
    [[y, x]] = np.argwhere(node.state == 0)
    childNodes: list = []

    for allowedBlankFieldStep in ALLOWED_BLANK_FIELD_STEPS:
        newX: int = x + allowedBlankFieldStep[1]
        newY: int = y + allowedBlankFieldStep[0]

        if not (newY > 3 or newY < 0 or newX > 3 or newX < 0):
            newState: list = np.empty_like(node.state)
            newState[:] = node.state
            newState[y][x] = newState[newY][newX]
            newState[newY][newX] = 0

            childNodes.append(Node(newState))

    return childNodes
