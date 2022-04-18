#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar
from collections import Counter, defaultdict

from sympy import numbered_symbols
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    wordList = x.split( )
    return dict(Counter(wordList))
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    def evalHingeLoss(featureVector, y, weights):
        margin = dotProduct(weights, featureVector)*y
        if margin < 1:
            hingeLoss = 1 - margin
        else:
            hingeLoss = 0
        return hingeLoss

    def predictClass(x):
        featureVector = featureExtractor(x)
        score = dotProduct(weights, featureVector)
        if score >= 0:
            predictedClass = 1
        else:
            predictedClass = -1
        return predictedClass

    for i in range(numEpochs):
        for item in trainExamples:
            x, y = item
            featureVector = featureExtractor(x)
            hingeLoss = evalHingeLoss(featureVector, y, weights)
            if hingeLoss > 0:
                increment(weights, eta*y, featureVector)
    
    print(f'Epoch Number: {numEpochs}')
    print(f'Training error: {evaluatePredictor(trainExamples, predictClass)}')
    print(f'Validation error: {evaluatePredictor(validationExamples, predictClass)}')
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = dict()
        for i in range(random.randint(1, len(weights))):
            randomKey = random.choice(list(weights.keys()))
            phi[randomKey] =  random.randint(1, 100)

        score = dotProduct(phi, weights)
        if score >= 0:
            y = 1
        else:
            y = -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        newString = x.replace(" ", "")
        nGrams = defaultdict(int)
        for index in range(0, len(newString)):
            nLetterWord = newString[index:index+n]
            if len(nLetterWord) == n:
                nGrams[nLetterWord] += 1 
        return nGrams
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))
    
    return validationError

############################################################
# Problem 5: k-means
############################################################

import random
import math

def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)

    def AddSparseVectors(d1:dict, d2:dict) -> dict:
        """
        @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
        @param dict d2: same as d1
        @return float: the squared euclidean distance between d1 and d2
        """
        if d1 == None:
            return d2
        if d2 == None:
            return d1
        
        if len(d1) < len(d2):
            return AddSparseVectors(d2, d1)
        else:
            d1Copy = d1.copy() # To not mutate features
            for key, value in d2.items():
                if key not in d1.keys():
                    d1Copy.update({key: value})
                else:
                    d1Copy[key] += value
            return d1Copy

    def distSquared(d1: Dict, d2: Dict) -> float:
        """
        @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
        @param dict d2: same as d1
        @return float: the squared euclidean distance between d1 and d2
        """
        # Three cases:  (delta of commons keys)^2 + (d1 uniques)^2 + (d2 uniques)^2
        sum = 0

        for key in d1.keys() & d2.keys():
            sum += (d1[key] - d2[key])**2
        
        for key in d1.keys() - d2.keys():
            sum += (d1[key])**2

        for key in d2.keys() - d1.keys():
            sum += (d2[key])**2
        
        return sum
    
    random.seed(4)
    centroidList = list()
    for i in range(K):
        centroidList.append(random.choice(examples))

    for i in range(maxEpochs):
        print(f'================ Starting Epoch {i} ================')
        # Initialise dictionaries
        data2cluster = dict.fromkeys(range(len(examples)), None)
        finalLoss = dict.fromkeys(range(len(examples)), None)
        clusterSum = dict.fromkeys(range(len(centroidList)), None)
        clusterCounts = dict.fromkeys(range(len(centroidList)), 0)
        # print(f'Dataset: {examples}')
        # print(data2cluster, clusterSum, clusterCounts)
        # print('\n')

        for featureID, featureVector in enumerate(examples):
            # print(f'Processing Data point {featureID}')

            # Initialise default cluster assignment
            assignedClusterID = 0
            lowestDistSq = math.inf

            # Evaluate the distances from current featureVector to each centroid
            for centroidID, centroidVector in enumerate(centroidList):
                # print(f'    centroidID: {centroidID}, centroidVector = {centroidVector}')
                currentDistSq = distSquared(featureVector, centroidVector)
                # print(f'    Current Distance Squared: {currentDistSq}')
                if currentDistSq < lowestDistSq:
                    assignedClusterID = centroidID
                    lowestDistSq = currentDistSq

            # Associate the current featureVector to its closest centroid
            # print(f'    assignedClusterID: {assignedClusterID}')
            data2cluster[featureID] = assignedClusterID
            finalLoss[featureID] = lowestDistSq

            # Update that cluster's sum and counts
            currentSum = clusterSum[assignedClusterID]
            # print(f'    current sum for cluster: {assignedClusterID}: {clusterSum[assignedClusterID]}')
            clusterSum[assignedClusterID] = AddSparseVectors(featureVector, currentSum)
            # print(f'    updated sum for cluster: {assignedClusterID}: {clusterSum[assignedClusterID]}')
            clusterCounts[assignedClusterID] += 1
            # print('\n')

        # print(f'Results from epoch')
        # print(f'    Each datapoints assigned cluster: {data2cluster}')
        # print(f'    Each clusters\' total sum: {clusterSum}')
        # print(f'    Each clusters\' total count: {clusterCounts}')
        # print('\n')

        # Update centroids
        # print(f'Update Centroids')
        for centroidID, centroidVector in enumerate(centroidList):
            # print(f'    Cluster Sum: {clusterSum[centroidID]}')
            # print(f'    Cluster count: {clusterCounts[centroidID]}')
            for key, value in clusterSum[centroidID].items():
                centroidVector[key] = value / clusterCounts[centroidID]
        #     print(f'        Updated centroidVector = {centroidVector}')
        #     print('\n')

        # print(f'Final centroids: {centroidList}')
        # print('\n')
    return [centroidList, data2cluster, sum(v for k, v in finalLoss.items())]

    # END_YOUR_CODE
