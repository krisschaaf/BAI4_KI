from treeSearch import *

targetArray = Node([], np.array([
    [15, 10, 3, 6],
    [4, 7, 14, 11],
    [9, 12, 5, 2],
    [0, 1, 8, 13]]))

oneStepArray = Node([], np.array([
    [15, 10, 3, 6],
    [4, 7, 14, 11],
    [9, 12, 5, 2],
    [1, 0, 8, 13]]))

threeStepArray = Node([], np.array([
    [15, 10, 3, 6],
    [4, 7, 14, 11],
    [9, 12, 5, 2],
    [1, 8, 13, 0]]))

firstExample = Node([], np.array([
    [15, 10, 3, 6],
    [4, 7, 14, 11],
    [0, 12, 5, 2],
    [9, 1, 8, 13]]))

secondExample = Node([], np.array([
    [15, 10, 3, 6],
    [4, 7, 14, 11],
    [12, 0, 5, 2],
    [9, 1, 8, 13]]))

thirdExample = Node([], np.array([
    [15, 10, 3, 6],
    [4, 14, 0, 11],
    [12, 7, 5, 2],
    [9, 1, 8, 13]]))

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
