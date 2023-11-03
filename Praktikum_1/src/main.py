from src.algorithms.treeSearch import iddfs,treeSearch
from src.classes.Node import Node
from src.utils import constants
from src.utils.enums import Strategy

if __name__ == '__main__':
    strategy: Strategy = Strategy.IDS  # choose strategy here
    startNode: Node = constants.thirdExample  # choose example here
    maxDepth: int = 1000  # choose max depth for tree search
    maxIdsDepth: int = 5  # only relevant for IDS - choose max depth for iteration depth

    if strategy == Strategy.IDS:
        path = iddfs(startNode, maxDepth, maxIdsDepth)
    else:
        path = treeSearch(startNode, strategy, maxDepth)

    print(path)
