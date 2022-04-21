from util import *
from submission import *
from wordsegUtil import *
import graderUtil
import grader


# # Load doc. Set up language model
# corpus = 'leo-will.txt'
# possibleFills = wordsegUtil.makeInverseRemovalDictionary(corpus, 'aeiou')
# print(possibleFills('hll'))

# # Check State Definitions
# testObj = VowelInsertionProblem(queryWords, bigramCost, possibleFills)
# print(testObj.isEnd((0, '-BEGIN-')))
# print(testObj.startState())

# # Check Final Answer
# # ucs = util.UniformCostSearch(verbose=0)
# # ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
# # print(' '.join(ucs.actions))

_, bigramCost, possibleFills = grader.getRealCosts()

QUERIES_INS = [
    'strng',
    'pls',
    'hll thr',
    'whats up',
    'dudu and the prince',
    'frog and the king',
    'ran with the queen and swam with jack',
    'light bulbs need change',
    'ffcr nd prnc ndrw',
    'ffcr nd shrt prnc',
    'ntrntnl',
    'smthng',
    'btfl',
]

ucs = util.UniformCostSearch(verbose=0)
for query in QUERIES_INS:
    query = wordsegUtil.cleanLine(query)
    ws = [wordsegUtil.removeAll(w, 'aeiou') for w in wordsegUtil.words(query)]
    ucs.solve(VowelInsertionProblem(ws, bigramCost, possibleFills))
    print(ws)
    print(' '.join(ucs.actions))
    print('\n')