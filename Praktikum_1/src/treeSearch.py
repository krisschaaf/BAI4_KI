import numpy as np
import constants


def execute(node, strategy):
    openList = [[node]]
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

    if isTargetState(node.state):
        return node

    createChildNodes(node, openList, closedList)


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
                print("Found:")
                print(intValue)
                return True

    return False


# @returns 3d array
# @param openList contains list of possible paths (nodes)
# @param closedList contains all nodes which incident Edges have been checked
def createChildNodes(node, openList, closedList):
    [[y, x]] = np.argwhere(node.state == 0)

    for allowedBlankFieldStep in constants.ALLOWED_BLANK_FIELD_STEPS:
        newX = x + allowedBlankFieldStep[1]
        newY = y + allowedBlankFieldStep[0]

        if not (newY > 3 or newY < 0 or newX > 3 or newX < 0):
            newState = np.empty_like(node.state)

            # Copy values from the original array to the new array
            newState[:] = node.state

            newState[y][x] = newState[newY][newX]
            newState[newY][newX] = 0
            childNode = Node([],newState)
            # -1 means last element of list
            if (not any(np.array_equal(childNode.state, element.state) for element in closedList) and
                    not any(np.array_equal(childNode.state, element[-1].state) for element in openList)):
                # TODO fix: get list from openlist when last item is node and append to node to that list
                for list in openList:
                    if np.array_equal(list[-1].state,node.state):
                        newList = list.copy()
                        newList.append(childNode)
                        openList.append(newList)


            #elif not any(np.array_equal(newNode, element) for element in closedList) and any(
                #    np.array_equal(newNode, element) for element in openList):
                # which node hast shortest path, add that one, delete other one
                print("tbd")

            node.childNodes.append(childNode)
    for list in openList:
        if np.array_equal(list[-1].state, node.state):
            openList.remove(list)
    closedList.append(node)

    # remove listitem which has node as last element from openlist

    return openList

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
