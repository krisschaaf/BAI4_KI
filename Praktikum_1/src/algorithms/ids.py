from src.Node import *


def execute(start_node, max_depth):
    for depth in range(max_depth + 1):
        result = dfs_limit(start_node, depth)
        if result:
            return result

    print("Target node not found within the maximum depth")
    return None


# DFS with depth limit
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
            continue

        child_nodes = generateChildNodes(node)
        for child_node, _ in child_nodes:
            if child_node not in visited:
                stack.append((child_node, path[:], depth + 1))

    return None
