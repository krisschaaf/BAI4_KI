import numpy as np
import constants


def execute(node, strategy):
    openList = [node]
    closedList = []

    match strategy:
        case "BFS":
            bfs(node, openList, closedList)
        case "DFS":
            print("tbd")
        case "IDS":
            print("tbd")
        case "AStar":
            print("tbd")
        case _:
            print("404 Not available!")


def bfs(node, openList, closedList):
    # List<State> closedList
    # List<List<State>> openList

    if isTargetNode(node):
        return node

    createStatesForNode(node, openList, closedList)


# Node int[4][4]
# Muss genau eine 1 enthalten
def isTargetNode(node) -> bool:
    [[y, x]] = np.argwhere(node == 1)

    currentValue: int = 1

    while nextValueFound(node, y, x, currentValue):
        currentValue += 1
        [[y, x]] = np.argwhere(node == currentValue)

        if currentValue == 15:
            return True

    return False


def nextValueFound(node, y, x, currentValue):
    for allowedSpringerStep in constants.ALLOWED_SPRINGER_STEPS:
        newX = x + allowedSpringerStep[1]
        newY = y + allowedSpringerStep[0]

        if not (newY > 3 or newY < 0 or newX > 3 or newX < 0):
            intValue = node[newY][newX]
            if intValue == currentValue + 1:
                print("Found:")
                print(intValue)
                return True

    return False


# @returns 3d array
# @param openList contains list of possible paths (nodes)
# @param closedList contains all nodes which incident Edges have been checked
def createStatesForNode(node, openList, closedList):
    [[y, x]] = np.argwhere(node == 0)

    for allowedBlankFieldStep in constants.ALLOWED_BLANK_FIELD_STEPS:
        newX = x + allowedBlankFieldStep[1]
        newY = y + allowedBlankFieldStep[0]

        if not (newY > 3 or newY < 0 or newX > 3 or newX < 0):
            newNode = np.empty_like(node)

            # Copy values from the original array to the new array
            newNode[:] = node

            newNode[y][x] = newNode[newY][newX]
            newNode[newY][newX] = 0

            if (not any(np.array_equal(newNode, element) for element in closedList) and
                    not any(np.array_equal(newNode, element) for element in openList)):

                # TODO fix: get list from openlist when last item is node and append to node to that list
                matchingPath = [item for item in openList if item[-1] == node]
                matchingPath[0].append(newNode)

            elif not any(np.array_equal(newNode, element) for element in closedList) and any(
                    np.array_equal(newNode, element) for element in openList):
                # which node hast shortest path, add that one, delete other one
                print("tbd")

    closedList.append(node)

    # remove listitem which has node as last element from openlist

    return openList
