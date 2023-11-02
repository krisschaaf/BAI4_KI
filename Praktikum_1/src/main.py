from src.algorithms.treeSearch import iddfs,treeSearch
from src.utils import constants
from src.utils.enums import Strategy

if __name__ == '__main__':
    strategy = Strategy.A_STAR  # choose strategy here
    startNode = constants.thirdExample  # choose example here
    maxDepth = 1000  # choose max depth for tree search
    maxIdsDepth = 1  # only relevant for IDS - choose max depth for iteration depth

    if strategy == Strategy.IDS:
        path = iddfs(startNode, maxDepth, maxIdsDepth)
    else:
        path = treeSearch(startNode, strategy, maxDepth)

    print(path)
