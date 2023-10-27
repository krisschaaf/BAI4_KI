import numpy as np
from treeSearch import *

targetArray = Node([],np.array([[15, 10, 3, 6],
                        [4, 7, 14, 11],
                        [9, 12, 5, 2],
                        [0, 1, 8, 13]]))

array = Node([],np.array([[15, 10, 3, 6],
                  [4, 7, 11, 14],
                  [9, 12, 5, 2],
                  [0, 1, 8, 13]]))

ALLOWED_SPRINGER_STEPS = np.array([
    [-2, -1],
    [-2, +1],
    [-1, -2],
    [+1, -2],
    [-1, +2],
    [+1, +2],
    [+2, -1],
    [+2, +1]]
)

ALLOWED_BLANK_FIELD_STEPS = np.array([
    [0, -1],
    [0, +1],
    [-1, 0],
    [+1, 0]
])