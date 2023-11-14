from src.algorithms.treeSearch import iddfs,treeSearch
from src.classes.Node import Node
from src.utils import constants
from src.utils.enums import Strategy
import time


def printResult(_path, duration):
    if _path is None:
        return
    else:
        print("Length of path: " + str(len(_path)))
        print("targetNode: \r\n" + str(_path[-1].state))
        print("time: " + str(duration) + " ns")


if __name__ == '__main__':
    strategy: Strategy = Strategy.BFS  # choose strategy here
    startNode: Node = constants.thirdExample  # choose example here
    maxDepth: int = 1000  # choose max depth for tree search
    maxIdsDepth: int = 5  # only relevant for IDS - choose max depth for iteration depth

    startTime = time.perf_counter_ns()

    if strategy == Strategy.IDS:
        path = iddfs(startNode, maxDepth, maxIdsDepth)
    else:
        path = treeSearch(startNode, strategy, maxDepth)

    endTime = time.perf_counter_ns()

    printResult(path, endTime - startTime)




