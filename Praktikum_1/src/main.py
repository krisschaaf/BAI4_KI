from src.utils import constants
from archive import bfs, dfs
from src.algorithms import ids
from src.algorithms import aStar
from Node import isTargetState, generateChildNodes


def run(node, strategy):
    match strategy:
        case "BFS":
            path = bfs.execute(node)
            print(path)
        case "DFS":
            path = dfs.execute(node)
            print(path)
        case "IDS":
            path = ids.execute(node, 4)
            print(path)
        case "A_STAR":
            path = aStar.execute(node, aStar.heuristic)
            print("Is Target State: ")
            print(isTargetState(path.state))
        case _:
            print("404 Not available!")


def treeSearch(problem, strategy, maxDepth):
    closedList = set()
    openList = [(problem, [], 0)]

    while openList:

        match strategy:
            case "BFS":
                node, path, depth = openList.pop(0)  # FiFo
            case "DFS":
                node, path, depth = openList.pop()  # FiLo (Stack)
            case "IDS":
                node, path, depth = openList.pop()
            # case "A_STAR":
            #
            case _:
                raise Exception("Strategy not available")

        if node in closedList:
            continue

        closedList.add(node)
        path.append(node)

        if isTargetState(node.state):
            return path  # Return the path to the target node

        if depth >= maxDepth:
            print("Could not find any solution, because max depth has been reached!")
            return

        childNodes = generateChildNodes(node)

        for childNode, _ in childNodes:
            if childNode not in closedList:
                openList.append((childNode, path[:], depth + 1))  # Copy the path for the child node

    return None  # Target node not found


if __name__ == '__main__':
    treeSearch(constants.thirdExample, "IDS", 1000)
