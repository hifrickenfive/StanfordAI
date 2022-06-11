from typing import Callable, List, Set

import shell
import util
import wordsegUtil


############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    # State Definition: (current idx in the queryWords list)

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0 
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state == len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        actions = []
        for i in range(state, len(self.query)+1):
            actions.append((i, self.query[state:i]))
        
        result = []
        for newState, action in actions:
            cost = self.unigramCost(action)
            result.append((action, newState, cost)) # action, newState, cost
        return result
        # END_YOUR_CODE

def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE


############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords: List[str], bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    # State Definition: (current idx in the queryWords list, previous chosen word)

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        if len(self.queryWords) == 0:
            return True
        else:
            return state[0] == len(self.queryWords) # stop after final word transformed
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)

        currIdx = state[0]
        currWrd = self.queryWords[currIdx]
        prevWrd = state[1]

        actions = self.possibleFills(currWrd)
        if len(actions) == 0:
            actions = {currWrd}

        results = []
        for action in actions:
            cost = self.bigramCost(prevWrd, action)
            results.append((action, (currIdx+1, action), cost)) # action, newState, cost
        return results
        # END_YOUR_CODE

def insertVowels(queryWords: List[str], bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    return ' '.join(ucs.actions)
    # END_YOUR_CODE


############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query: str, bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    # State Definition: (current idx in the query string, previous chosen word)

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        if len(self.query) == 0:
            return True
        else:
            return state[0] == len(self.query) # stop after final word transformed
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
        startIdx = state[0]
        prevWord = state[1]

        results = []
        for i in range(startIdx, len(self.query)):
            actions = self.possibleFills(self.query[startIdx:i+1])
            for action in actions:
                # Check if only vowels
                onlyVowels = True
                for char in action:
                    if char not in 'aeiou':
                        onlyVowels = False
                if onlyVowels or action == self.query[startIdx:i+1]:
                    continue # Exclude onlyVowels and out-of-vocab words
                else:
                    results.append((action, (i+1, action), self.bigramCost(prevWord, action)))

        return results
        # END_YOUR_CODE


def segmentAndInsert(query: str, bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_CODE


############################################################

if __name__ == '__main__':
    shell.main()
