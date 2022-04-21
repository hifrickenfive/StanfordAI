from util import *
from submission import *
from wordsegUtil import *
import graderUtil


# Load doc. Set up language model
path2doc = 'leo-will.txt'
unigramCost, bigramModel = makeLanguageModels(path2doc)

# Define instance
query = 'word'
# query = 'twowords'
query = 'andthreewords'
print(len(query))
test = SegmentationProblem(query, unigramCost)

# Check my succAndCost
# answer = test.succAndCost(0)
# print(answer)


grader = graderUtil.Grader()
submission = grader.load('submission')
# answer = submission.segmentWords('word', unigramCost)
# print(answer)

ucs = util.UniformCostSearch(verbose=0)
ucs.solve(SegmentationProblem(query, unigramCost))
print(ucs.actions)

