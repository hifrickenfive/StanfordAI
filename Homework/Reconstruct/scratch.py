from util import *
from submission import *
from wordsegUtil import *
import graderUtil


# Load doc. Set up language model
corpus = 'leo-will.txt'
possibleFills = wordsegUtil.makeInverseRemovalDictionary(corpus, 'aeiou')
print(possibleFills('hll'))

queryWords = 'wld lk t hv mr lttrs'.split()
print(queryWords)

def bigramCost(a, b):
    corpus = [wordsegUtil.SENTENCE_BEGIN] + 'beam me up scotty'.split()
    if (a, b) in list(zip(corpus, corpus[1:])):
        return 1.0
    else:
        return 1000.0

# Check State Definitions
testObj = VowelInsertionProblem(queryWords, bigramCost, possibleFills)
print(testObj.isEnd((queryWords[-2],queryWords[-1])))

# Check Final Answer
# ucs = util.UniformCostSearch(verbose=0)
# ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
# print(' '.join(ucs.actions))
