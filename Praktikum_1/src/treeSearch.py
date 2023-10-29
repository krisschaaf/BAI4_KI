from collections import deque
import numpy as np
import constants


def execute(node, strategy):

    match strategy:
        case "BFS":
            path = bfs(node)
            print(path)
        case "DFS":
            print("tbd")
        case "IDS":
            print("tbd")
        case "AStar":
            print("tbd")
        case _:
            print("404 Not available!")


# Node int[4][4]
# Muss genau eine 1 enthalten
def isTargetState(state) -> bool:
    [[y, x]] = np.argwhere(state == 1)

    currentValue: int = 1

    while nextValueFound(state, y, x, currentValue):
        currentValue += 1
        [[y, x]] = np.argwhere(state == currentValue)

        if currentValue == 15:
            return True

    return False


def nextValueFound(state, y, x, currentValue):
    for allowedSpringerStep in constants.ALLOWED_SPRINGER_STEPS:
        newX = x + allowedSpringerStep[1]
        newY = y + allowedSpringerStep[0]

        if not (newY > 3 or newY < 0 or newX > 3 or newX < 0):
            intValue = state[newY][newX]
            if intValue == currentValue + 1:
                return True

    return False


class Node:
    def __init__(self, childNodes, state):
        self.state = state
        self.childNodes = childNodes

    def getState(self):
        return self.state

    def getChildNodes(self):
        return self.childNodes

    def addChildNode(self, node):
        self.childNodes.append(node)


# Define a function to generate child nodes for a given node.
def generateChildNodes(node):
    [[y, x]] = np.argwhere(node.state == 0)
    childNodes = []

    for allowedBlankFieldStep in constants.ALLOWED_BLANK_FIELD_STEPS:
        newX = x + allowedBlankFieldStep[1]
        newY = y + allowedBlankFieldStep[0]

        if not (newY > 3 or newY < 0 or newX > 3 or newX < 0):
            newState = np.empty_like(node.state)
            newState[:] = node.state
            newState[y][x] = newState[newY][newX]
            newState[newY][newX] = 0

            childNodes.append(Node([], newState))

    return childNodes


# BFS function with helper functions
def bfs(startNode):
    closedList = set()
    openList = deque([(startNode, [])])

    while openList:
        node, path = openList.popleft()

        if node in closedList:
            continue

        closedList.add(node)
        path.append(node)

        if isTargetState(node.state):
            return path  # Return the path to the target node

        childNodes = generateChildNodes(node)

        for childNode in childNodes:
            if childNode not in closedList:
                openList.append((childNode, path[:]))  # Copy the path for the child node

    return None  # Target node not found
