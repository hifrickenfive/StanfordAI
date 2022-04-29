from sympy import multiplicity
import util, math, random
from collections import defaultdict
from util import ValueIteration
from typing import List, Callable, Tuple, Any
import graderUtil


grader = graderUtil.Grader()
submission = grader.load('submission')

cardValues = [1,2,3]
multiplicity = 1
threshold = 4
peekCost = 10
test = submission.BlackjackMDP(cardValues, multiplicity, threshold, peekCost)


# Peek
currentState = (0, None, (1,1,1))
result = test.succAndProbReward(currentState, 'Peek')
print(result)

# Peek
currentState = (0, 1, (1,1,1))
result = test.succAndProbReward(currentState, 'Peek')
print(result)

# Quit
currentState = (0, None, (1,1,1))
result = test.succAndProbReward(currentState, 'Peek')
print(result)

# Take start
currentState = (0, None, (1,1,1))
result = test.succAndProbReward(currentState, 'Take')
print(result)

# Take ok
currentState = (1, None, (1,1,1))
result = test.succAndProbReward(currentState, 'Take')
print(result)

# Take bust
currentState = (3, None, (1,1,1))
result = test.succAndProbReward(currentState, 'Take')
print(result)

# Take last card
currentState = (3, None, (1,0,0))
result = test.succAndProbReward(currentState, 'Take')
print(result)

currentState = (3, None, (0,1,0))
result = test.succAndProbReward(currentState, 'Take')
print(result)