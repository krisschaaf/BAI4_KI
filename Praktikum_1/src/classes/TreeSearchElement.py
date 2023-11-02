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