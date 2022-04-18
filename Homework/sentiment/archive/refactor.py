from typing import Callable, Dict, List, Tuple, TypeVar
from collections import Counter, defaultdict

from sympy import numbered_symbols
from util import *
import random

def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[list, list, float]:

    def euclideanDistance(v1, v2, v1_square, v2_square):
        return v1_square + v2_square - 2 * dotProduct(v1, v2)

    def assign(v, centroids, e_square, c_squares):
        '''
        assign each vector to a centroid with minimum reconstruction loss
        '''
        min_dist = 1000000
        for c in range(len(centroids)):
            dist = euclideanDistance(v, centroids[c], e_square, c_squares[c])
            if dist < min_dist:
                min_dist = dist
                min_c = c
        return min_c, min_dist

    def avgV(vs):
        size = float(len(vs))
        result = defaultdict(float)
        for vec in vs:
            for k, v in vec.items():
                result[k] += v / size
        return result

    centroids = random.sample(examples, K)
    ele_squares = [dotProduct(ele, ele) for ele in examples]
    example_centroid_map = [0 for _ in examples]
    it = 0
    while True:
        loss = 0.0
        it += 1

        c_squares = [dotProduct(c, c) for c in centroids]

        # assign each ele to centroids and get reconstruction loss
        centroid_example_map = {x: [] for x in range(len(centroids))}
        new_example_centroid_map = []

        for i, (ele, e_square) in enumerate(zip(examples, ele_squares)):
            c, l = assign(ele, centroids, e_square, c_squares) # assignment and loss
            print(c, l)
            centroid_example_map[c].append(ele)
            new_example_centroid_map.append(c)
            loss += l

        # import pdb;pdb.set_trace()
        if example_centroid_map == new_example_centroid_map or it >= maxEpochs:
            return centroids, example_centroid_map, loss

        example_centroid_map = new_example_centroid_map
        # compute new centroids
        centroids = [avgV(eles) for eles in centroid_example_map.values()]
    # END_YOUR_CODE

random.seed(42)
x1 = {0:0, 1:0}
x2 = {0:0, 1:1}
x3 = {0:0, 1:2}
x4 = {0:0, 1:3}
x5 = {0:0, 1:4}
x6 = {0:0, 1:5}
examples = [x1, x2, x3, x4, x5, x6]
maxEpochs = 2
centers, assignments, totalCost = kmeans(examples, 2, maxEpochs)
print(centers, assignments, totalCost)