from collections import deque
import numpy as np
import constants
import heapq


def execute(node, strategy):
    match strategy:
        case "BFS":
            path = bfs(node)
            print(path)
        case "DFS":
            path = dfs(node)
            print(path)
        case "IDS":
            path = iddfs(node, 1000000000000000)
            print(path)
        case "AStar":
            path = astar(node, heuristic)
            print(path)
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
    def __init__(self, childNodes, state, cost=0):
        self.state = state
        self.childNodes = childNodes
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

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

            childNodes.append([Node([], newState), 0])

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

        for childNode, _ in childNodes:
            if childNode not in closedList:
                openList.append((childNode, path[:]))  # Copy the path for the child node

    return None  # Target node not found


def dfs(start_node):
    visited = set()
    stack = [(start_node, [])]

    while stack:
        node, path = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        path.append(node)

        if isTargetState(node.state):
            return path  # Return the path to the target node

        child_nodes = generateChildNodes(node)
        for child_node, _ in child_nodes:
            if child_node not in visited:
                stack.append((child_node, path[:]))  # Copy the path for the child node

    return None  # Target node not found


# IDDFS function with helper functions
def iddfs(start_node, max_depth):
    for depth in range(max_depth + 1):
        result = dfs_limit(start_node, depth)
        if result:
            return result

    return None  # Target node not found within the maximum depth


# Depth-First Search function with depth limit
def dfs_limit(start_node, max_depth):
    visited = set()
    stack = [(start_node, [], 0)]

    while stack:
        node, path, depth = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        path.append(node)

        if isTargetState(node.state):
            return path  # Return the path to the target node

        if depth >= max_depth:
            continue  # Skip this branch if we've reached the depth limit

        child_nodes = generateChildNodes(node)
        for child_node, _ in child_nodes:
            if child_node not in visited:
                stack.append((child_node, path[:], depth + 1))  # Copy the path and increase depth

    return None  # Target node not found within the specified depth limit


def heuristic(node):
    # Your implementation of the heuristic function goes here.
    # It should estimate the cost from the current node to the target node.
    return 0


# A* search function with helper functions
def astar(start_node, heuristic):
    start = [start_node, 0]
    open_set = [start]  # Priority queue for nodes (cost, node, path)
    closed_set = set()

    while open_set:
        current_node = heapq.heappop(open_set)

        if isTargetState(current_node[0].state):
            return current_node[0]  # Return the target node

        if current_node[0] in closed_set:
            continue

        closed_set.add(current_node[0])

        child_nodes = generateChildNodes(current_node[0])

        for child, cost in child_nodes:
            if child not in closed_set:
                child_node = [child, current_node[1] + cost]
                priority = child_node[1] + heuristic(child)
                heapq.heappush(open_set, child_node)

    return None  # Target node not found
