from sympy import multiplicity
import util, math, random
from collections import defaultdict
from util import ValueIteration
from typing import List, Callable, Tuple, Any
import submission

cardValues = [1,2,3]
multiplicity = 1
threshold = 4
peekCost = 10
test = submission.BlackjackMDP(cardValues, multiplicity, threshold, peekCost)



# Peek
currentState = (0, None, (1,1,1))
result = test.succAndProbReward(currentState, 'Peek')
assert result == [(0, 0, (1, 1, 1)), (0, 1, (1, 1, 1)), (0, 2, (1, 1, 1))]

# Peek
currentState = (0, 1, (1,1,1))
result = test.succAndProbReward(currentState, 'Peek')
assert result == []

# Quit
currentState = (0, None, (1,1,1))
result = test.succAndProbReward(currentState, 'Peek')
assert result == [(0, 0, (1, 1, 1)), (0, 1, (1, 1, 1)), (0, 2, (1, 1, 1))]

# Take start
currentState = (0, None, (1,1,1))
result = test.succAndProbReward(currentState, 'Take')
assert result == [(1, None, (0, 1, 1)), (2, None, (1, 0, 1)), (3, None, (1, 1, 0))]

# Take ok
currentState = (1, None, (1,1,1))
result = test.succAndProbReward(currentState, 'Take')
assert result == [(2, None, (0, 1, 1)), (3, None, (1, 0, 1)), (4, None, (1, 1, 0))]

# Take bust
currentState = (3, None, (1,1,1))
result = test.succAndProbReward(currentState, 'Take')
assert result == [(4, None, (0, 1, 1)), (5, None, None), (6, None, None)]

# Take last card
currentState = (3, None, (1,0,0))
result = test.succAndProbReward(currentState, 'Take')
print(result)

currentState = (3, None, (0,1,0))
result = test.succAndProbReward(currentState, 'Take')
print(result)