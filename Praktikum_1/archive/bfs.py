from src.Node import *
from collections import deque


def execute(startNode):
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

        for childNode, _ in childNodes:
            if childNode not in closedList:
                openList.append((childNode, path[:]))  # Copy the path for the child node

    return None  # Target node not found
