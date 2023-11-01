import heapq
from src.Node import *


def heuristic(node) -> int:
    [[y, x]] = np.argwhere(node.state == 1)

    currentValue: int = 1

    while nextValueFound(node.state, y, x, currentValue):
        currentValue += 1
        [[y, x]] = np.argwhere(node.state == currentValue)

    return 15 - currentValue


# A* search function with helper functions
def execute(start_node, heuristic):
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
                child_node = [child, heuristic(child) + current_node[1]]
                heapq.heappush(open_set, child_node)

    return None  # Target node not found
