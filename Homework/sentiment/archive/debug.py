from typing import Callable, Dict, List, Tuple, TypeVar
from collections import Counter, defaultdict

from sympy import numbered_symbols
from util import *
import random
import math


def euclideanDistance(v1, v2, v1_square, v2_square):
    return v1_square + v2_square - 2 * dotProduct(v1, v2)



ele_squares = [dotProduct(ele, ele) for ele in examples]