import heapq
from Node import *


def heuristic(node) -> int:
    [[y, x]] = np.argwhere(node.state == 1)

    currentValue: int = 1

    while nextValueFound(node.state, y, x, currentValue):
        currentValue += 1
        [[y, x]] = np.argwhere(node.state == currentValue)

    return 15 - currentValue


class TreeSearchElement:
    def __init__(self, node, path=None, depth=0, cost=0):
        if path is None:
            path = []
        self.node = node
        self.path = path
        self.depth = depth
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


def treeSearch(problem, strategy, maxDepth):
    closedList = set()
    openList = [TreeSearchElement(problem)]

    while openList:

        match strategy:
            case "BFS":
                treeSearchElement = openList.pop(0)  # FiFo
            case "DFS":
                treeSearchElement = openList.pop()  # FiLo (Stack)
            case "IDS":
                treeSearchElement = openList.pop()
            case "A_STAR":
                treeSearchElement = heapq.heappop(openList)  # Priority Queue
            case _:
                raise Exception("Strategy not available")

        if treeSearchElement.node in closedList:
            continue

        closedList.add(treeSearchElement.node)
        treeSearchElement.path.append(treeSearchElement.node)

        if isTargetState(treeSearchElement.node.state):
            return treeSearchElement.path

        if treeSearchElement.depth >= maxDepth:
            print("Could not find any solution, because max depth has been reached!")
            return

        childNodes = generateChildNodes(treeSearchElement.node)

        for childNode in childNodes:
            if childNode not in closedList:

                if strategy == "A_STAR":
                    newTreeSearchElement = TreeSearchElement(
                        childNode,
                        treeSearchElement.path[:],
                        treeSearchElement.depth + 1,
                        heuristic(childNode) + treeSearchElement.node.cost
                    )
                    heapq.heappush(openList, newTreeSearchElement)
                else:
                    openList.append(TreeSearchElement(childNode, treeSearchElement.path[:], treeSearchElement.depth + 1))

    return None


if __name__ == '__main__':
    treeSearch(constants.thirdExample, "A_STAR", 1000)
