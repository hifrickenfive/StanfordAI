import util
import submission
# from wordsegUtil import *
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

def bigramCost(a, b):
    if b in ['and', 'two', 'three', 'word', 'words']:
        return 1.0
    else:
        return 1000.0

fills_ = {
    'nd': set(['and']),
    'tw': set(['two']),
    'thr': set(['three']),
    'wrd': set(['word']),
    'wrds': set(['words']),
}
fills = lambda x: fills_.get(x, set())
ucs = util.UniformCostSearch(verbose=0)
ucs.solve(submission.JointSegmentationInsertionProblem('ndthrwrds', bigramCost, fills))
actionSeq = ucs.actions
print(actionSeq)
