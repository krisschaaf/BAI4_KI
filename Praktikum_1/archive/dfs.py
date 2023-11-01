from src.Node import *


def execute(start_node):
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

