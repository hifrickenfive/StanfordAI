# import submission
from sympy import multiplicity
import util, math, random
from collections import defaultdict
from util import ValueIteration
from typing import List, Callable, Tuple, Any
import graderUtil

grader = graderUtil.Grader()
submission = grader.load('submission')

mdp1 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                threshold=10, peekCost=1)
mdp1.computeStates()
startState = mdp1.startState()
preBustState = (6, None, (1, 1))
postBustState = (11, None, None)

mdp2 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                threshold=15, peekCost=1)
preEmptyState = (11, None, (1, 0))

mdp3 = submission.BlackjackMDP(cardValues=[1, 2, 3], multiplicity=3,
                                threshold=8, peekCost=1)
mdp3_startState = mdp3.startState()
mdp3_preBustState = (6, None, (1, 1, 1))

# Make sure the succAndProbReward function is implemented correctly.
tests = [
    ([((1, None, (1, 2)), 0.5, 0), ((5, None, (2, 1)), 0.5, 0)], mdp1, startState, 'Take'),
    ([((0, 0, (2, 2)), 0.5, -1), ((0, 1, (2, 2)), 0.5, -1)], mdp1, startState, 'Peek'),
    ([((0, None, None), 1, 0)], mdp1, startState, 'Quit'),
    ([((7, None, (0, 1)), 0.5, 0), ((11, None, None), 0.5, 0)], mdp1, preBustState, 'Take'),
    ([], mdp1, postBustState, 'Take'),
    ([], mdp1, postBustState, 'Peek'),
    ([], mdp1, postBustState, 'Quit'),
    ([((12, None, None), 1, 12)], mdp2, preEmptyState, 'Take'),
    ([((1, None, (2, 3, 3)), 1 / 3, 0), ((2, None, (3, 2, 3)), 1 / 3, 0), ((3, None, (3, 3, 2)), 1 / 3, 0)], mdp3,
        mdp3_startState, 'Take'),
    ([((7, None, (0, 1, 1)), 1 / 3, 0), ((8, None, (1, 0, 1)), 1 / 3, 0), ((9, None, None), 1 / 3, 0)], mdp3,
        mdp3_preBustState, 'Take'),
    ([((6, None, None), 1, 6)], mdp3, mdp3_preBustState, 'Quit')
]
for gold, mdp, state, action in tests:
    if not grader.require_is_equal(gold,
                                    mdp.succAndProbReward(state, action)):
        print(('   state: {}, action: {}'.format(state, action)))