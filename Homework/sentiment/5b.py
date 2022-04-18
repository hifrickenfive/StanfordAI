import random
import math

# def kmeans(examples: list[dict[str, float]], K: int,
#            maxEpochs: int) -> tuple[list, list, float]:

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
    
    random.seed(4)
    centroidList = list()
    for i in range(K):
        centroidList.append(random.choice(examples))


    for i in range(maxEpochs):
        print(f'================ Starting Epoch {i} ================')
        print(f'Initial centroidList: {centroidList}')
        # Initialise dictionaries
        data2cluster = dict.fromkeys(range(len(examples)), None)
        finalLoss = dict.fromkeys(range(len(examples)), None)
        clusterSum = dict.fromkeys(range(len(centroidList)), None)
        clusterCounts = dict.fromkeys(range(len(centroidList)), 0)
        # print(f'Dataset: {examples}')
        # print(data2cluster, clusterSum, clusterCounts)
        # print('\n')

        for featureID, featureVector in enumerate(examples):
            print(f'Processing Data point {featureID}: {featureVector}')

            # Initialise default cluster assignment
            assignedClusterID = 0
            lowestDistSq = math.inf

            # Evaluate the distances from current featureVector to each centroid
            for centroidID, centroidVector in enumerate(centroidList):
                print(f'    centroidID: {centroidID}, centroidVector = {centroidVector}')
                currentDistSq = distSquared(featureVector, centroidVector)
                print(f'    Current Distance Squared: {currentDistSq}')
                if currentDistSq < lowestDistSq:
                    assignedClusterID = centroidID
                    lowestDistSq = currentDistSq

            # Associate the current featureVector to its closest centroid
            data2cluster[featureID] = assignedClusterID
            finalLoss[featureID] = lowestDistSq
            print(f'    assignedClusterID: {assignedClusterID}, loss: {lowestDistSq}')
            print('\n')

            # Update that cluster's sum and counts
            currentSum = clusterSum[assignedClusterID]
            print(f'    Current sum of all datapoints in clusterID: {assignedClusterID} is {clusterSum[assignedClusterID]}')
            print(f'    Adding sparse vectors...')
            clusterSum[assignedClusterID] = AddSparseVectors(featureVector, currentSum)
            print(f'    Sum of all datapoints in clusterID: {assignedClusterID} is {clusterSum[assignedClusterID]}')
            clusterCounts[assignedClusterID] += 1
            print('\n')

            print(f'    Did datapoint mutate?: {featureVector}')
            print('\n')

        print(f'Results from epoch')
        print(f'    Each datapoint\'s cluster: {data2cluster}')
        print(f'    Each clusters\' total sum: {clusterSum}')
        print(f'    Each clusters\' total count: {clusterCounts}')
        print('\n')

        print(f'    Examples before: {examples}')
        newCentroidList = []
        for idx, centroidDict in enumerate(centroidList):
            print(f'    ITERATION {idx}')
            # print(f'    Cluster Sum: {clusterSum[centroidID]}')
            # print(f'    Cluster count: {clusterCounts[centroidID]}')
            # for key, value in clusterSum[centroidID].items():
            #     centroidVector[key] = value / clusterCounts[centroidID]
            newDict = {k: v/clusterCounts[idx] for k, v in clusterSum[idx].items()}
            print(f'    newDict: {newDict}')
            print(f'    Examples after: {examples}')
            newCentroidList.append(newDict)
            print(f'    Updating centroidList: {centroidList}')
            print(f'    Examples after: {examples}')
            print('\n')
        centroidList = newCentroidList

        print(f'    Final centroidList: {centroidList}')
        print(f'    Examples after: {examples}')
        # print('\n')

        print(f' Final loss: {finalLoss}')
        # totalLoss = sum(v for k, v in finalLoss.items())
        # print(f' New Assignments: {data2cluster}, Loss: {totalLoss}')
        # print(f' New centroids: {centroidList}')

      
    return [centroidList, data2cluster, sum(v for k, v in finalLoss.items())]


random.seed(42)
x1 = {0:0, 1:0}
x2 = {0:0, 1:1}
x3 = {0:0, 1:2}
x4 = {0:0, 1:3}
x5 = {0:0, 1:4}
x6 = {0:0, 1:5}
examples = [x1, x2, x3, x4, x5, x6]
centers, assignments, totalCost = kmeans(examples, 2, maxEpochs=10)
print(centers, assignments, totalCost)