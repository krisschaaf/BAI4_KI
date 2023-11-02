import heapq
from src.classes.Node import *
from src.algorithms.heuristic import heuristic
from src.classes.TreeSearchElement import TreeSearchElement
from src.utils.enums import Strategy


def treeSearch(problem, _strategy, _maxDepth, idsDepth=0):
    closedList = set()
    openList = [TreeSearchElement(problem)]

    while openList:

        match _strategy:
            case Strategy.BFS:
                treeSearchElement = openList.pop(0)  # FiFo
            case Strategy.DFS:
                treeSearchElement = openList.pop()  # FiLo (Stack)
            case Strategy.IDS:
                treeSearchElement = openList.pop()  # FiLo (Stack)
            case Strategy.A_STAR:
                treeSearchElement = heapq.heappop(openList)  # Priority Queue
            case _:
                raise Exception("Strategy not available")

        if treeSearchElement.node in closedList:
            continue

        closedList.add(treeSearchElement.node)
        treeSearchElement.path.append(treeSearchElement.node)

        if isTargetState(treeSearchElement.node.state):
            return treeSearchElement.path

        if treeSearchElement.depth >= _maxDepth:
            print("Could not find any solution, because max depth has been reached!")
            return

        if _strategy == Strategy.IDS and treeSearchElement.depth >= idsDepth:
            continue

        childNodes = generateChildNodes(treeSearchElement.node)

        for childNode in childNodes:
            if childNode not in closedList:

                if _strategy == Strategy.A_STAR:
                    newTreeSearchElement = TreeSearchElement(
                        childNode,
                        treeSearchElement.path[:],
                        treeSearchElement.depth + 1,
                        heuristic(childNode) + treeSearchElement.node.cost
                    )
                    heapq.heappush(openList, newTreeSearchElement)
                else:
                    openList.append(
                        TreeSearchElement(childNode, treeSearchElement.path[:], treeSearchElement.depth + 1))

    return None


def iddfs(_startNode, _maxDepth, _maxIdsDepth):
    for depth in range(_maxIdsDepth + 1):
        result = treeSearch(_startNode, Strategy.IDS, _maxDepth, depth)
        if result:
            return result

    print("Could not find any solution, because target node is not within max ids depth!")
    return None
