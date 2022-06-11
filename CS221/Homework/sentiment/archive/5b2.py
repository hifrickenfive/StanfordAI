from typing import Callable, Dict, List, Tuple, TypeVar
from collections import Counter, defaultdict

from sympy import numbered_symbols
from util import *
import random
import math

def kmeans(examples, K, maxEpochs):
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
        # print(f'    Input d1: {d1}, d2: {d2}')
        if d1 == None:
            # print(f'    Output d1: {d1}, d2: {d2}')
            return d2
        if d2 == None:
            # print(f'    Output d1: {d1}, d2: {d2}')
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
            # print(f'    Output d1: {d1}, d2: {d2}')
            return d1Copy

    def distSquared(d1: dict, d2: dict) -> float:
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
    
    def euclideanDistance(v1, v2, v1_square, v2_square):
        return v1_square + v2_square - 2 * dotProduct(v1, v2)

    # def distSquared2(v, centroid, e_square, c_square):
    #     '''
    #     assign each vector to a centroid with minimum reconstruction loss
    #     '''
    #     min_dist = math.inf
    #     dist = euclideanDistance(v, centroid, e_square, c_square)
    #     if dist < min_dist:
    #         min_dist = dist
    #     return min_dist


    random.seed(4)
    centroidList = random.sample(examples, K)
    ele_squares = [dotProduct(ele, ele) for ele in examples]

    for i in range(maxEpochs):
        print(f'================ Starting Epoch {i} ================')
        print(f'Initial centroidList: {centroidList}')
        # Initialise dictionaries
        data2cluster = dict.fromkeys(range(len(examples)), None)
        finalLoss = dict.fromkeys(range(len(examples)), None)
        clusterSum = dict.fromkeys(range(len(centroidList)), None)
        clusterCounts = dict.fromkeys(range(len(centroidList)), 0)

        c_squares = [dotProduct(c, c) for c in centroidList]

        for featureID, featureVector in enumerate(examples):

            # Initialise default cluster assignment
            assignedClusterID = 0
            lowestDistSq = math.inf

            # Evaluate the distances from current featureVector to each centroid
            for centroidID, centroidVector in enumerate(centroidList):
                print(f'    eles')
                currentDistSq = euclideanDistance(featureVector, centroidVector, ele_squares[featureID], c_squares[centroidID])
                print(f'    currentDist: {currentDistSq}')
                if currentDistSq < lowestDistSq:
                    assignedClusterID = centroidID
                    lowestDistSq = currentDistSq
            print(f'FeatureID {featureID}, {featureVector} dist: {lowestDistSq}, assignedCluster: {assignedClusterID}')

            # Associate the current featureVector to its closest centroid
            data2cluster[featureID] = assignedClusterID
            finalLoss[featureID] = lowestDistSq


            # Update that cluster's sum and counts
            currentSum = clusterSum[assignedClusterID]
            clusterSum[assignedClusterID] = AddSparseVectors(featureVector, currentSum)
            clusterCounts[assignedClusterID] += 1

        newCentroidList = []
        for idx, centroidDict in enumerate(centroidList):
            newDict = {k: v/clusterCounts[idx] for k, v in clusterSum[idx].items()}
            newCentroidList.append(newDict)

        centroidList = newCentroidList


    if i == maxEpochs or centroidList == centroidList:
        return [centroidList, data2cluster, sum(v for k, v in finalLoss.items())]


random.seed(42)
x1 = {0:0, 1:0}
x2 = {0:0, 1:1}
x3 = {0:0, 1:2}
x4 = {0:0, 1:3}
x5 = {0:0, 1:4}
x6 = {0:0, 1:5}
examples = [x1, x2, x3, x4, x5, x6]
centers, assignments, totalCost = kmeans(examples, 2, maxEpochs=2)
print(centers, assignments, totalCost)