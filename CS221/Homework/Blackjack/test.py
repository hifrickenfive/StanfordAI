from sympy import multiplicity
import util, math, random
from collections import defaultdict
from util import ValueIteration
from typing import List, Callable, Tuple, Any
import graderUtil


grader = graderUtil.Grader()
submission = grader.load('submission')


mdp = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                threshold=10, peekCost=1)
mdp.computeStates()
rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                    submission.blackjackFeatureExtractor,
                                    0)
# We call this here so that the stepSize will be 1
rl.numIters = 1

rl.incorporateFeedback((7, None, (0, 1)), 'Quit', 7, (7, None, None))

# incorporateFeedback(self, state: Tuple, action: Any, reward: int, newState: Tuple) -> None:
# rl.incorporateFeedback((7, None, (0, 1)), 'Quit', 7, (7, None, None))

# # Return the Q function associated with the weights and features
# def getQ(self, state: Tuple, action: Any) -> float:
#     score = 0
#     for f, v in self.featureExtractor(state, action):
#         score += self.weights[f] * v
#     return score
grader.require_is_equal(28, rl.getQ((7, None, (0, 1)), 'Quit'))
# grader.require_is_equal(7, rl.getQ((7, None, (1, 0)), 'Quit'))
# grader.require_is_equal(14, rl.getQ((2, None, (0, 2)), 'Quit'))
# grader.require_is_equal(0, rl.getQ((2, None, (0, 2)), 'Take'))