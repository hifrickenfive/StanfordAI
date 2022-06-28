import numpy as np
import prettyprinter as pp
import scipy.spatial
import time
from tqdm import tqdm

# Data types
arrayA = np.array([1, 2, 3])
print(f'\t {arrayA} Data type is {arrayA.dtype} \n')

arrayB = np.array([1, 2, 3.0])
print(f'\t {arrayB} Data type is {arrayB.dtype} \n')

# Random np from a uniform sample
rand1 = np.random.random(3)
print(f'\t Uniform sampling {rand1} \n')

# Random np from a normal distribution
rand2 = np.random.randn(3)
print(f'\t Normal sampling {rand2} \n')

# Dimensions - Easiest way to debug is to check dimensions
array_1d = np.array([1,2,3,4]) # 4
array_1by4 = np.array([[1,2,3,4]]) # 1x4
array_2by4 = np.array([[1,2,3,4], [5,6,7,8]]) # 2,4
print(f'\t dimensions are {array_1d.shape}, {array_1by4.shape}, {array_2by4.shape}')

# If dimensions aren't correct then use reshape
array_1d = array_1d.reshape(-1, 4) # 4,1
array_1by4 = array_1by4.reshape(2, 2)
array_2by4 = array_2by4.reshape(4,2)
print(f'\t dimensions are {array_1d.shape}, {array_1by4.shape}, {array_2by4.shape} \n')

large_array = np.array([i for i in range(400)])
large_array = large_array.reshape(20,20)
print(f'\t Return all the rows of column 5 {large_array[:, 5]} \n')

# 3D array
large_array = np.array([i for i in range(30)])
large_array = large_array.reshape(3,2,5)
print('3D Array')
print(f' {large_array} \n')
print('Return slice 0')
print(f' {large_array[0, :, :]} \n')
print('Return slice 1')
print(f' {large_array[1, :, :]} \n')
print('Return slice 2')
print(f' {large_array[2, :, :]} \n')
 
# Reshape by column or by row?
small_array = np.arange(4)
print(np.reshape(small_array, (2,2), order='C')) #default rows
print(np.reshape(small_array, (2,2), order='F')) #columns
print('\n')

# np math are element wise. You shouldn't need to do for loops.
array_2d = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(f'Given {array_2d}')
print(f'Sum of entire array {np.sum(array_2d)}')
print(f'Sum row-wise {np.sum(array_2d, axis=1)}')
print(f'Sum col-wise {np.sum(array_2d, axis=0)}')
print('\n')

# np dot as array
array_1 = np.array([1,2,3])
array_2 = np.array([4,5,6])
print('np dot product as array')
print(f'dim: {array_1.shape}, {array_2.shape}')
print(np.dot(array_1, array_2))
print('\n')

# np dot product as matrix
# CAUTION: The numpy.dot() method works separately for a matrix and an array.
array_1 = np.array([[1,2,3]])
array_2 = np.array([[4,5,6]])
array_2 = array_2.T
print('np dot product as matrices')
print(f'dim: {array_1.shape}, {array_2.shape}')
print(np.dot(array_1, array_2))
print('\n')

# transpose with .T
print(f'Transpose {array_1.T}')
print('\n')

# Ax = b with matmul
A = np.array([1,2,3,4]).reshape(2,2)
x = np.array([[50,60]]).T
b = np.matmul(A, x)
print('# Ax = b with matmul')
print(b) # [[170], [390]]
print('\n')

# Broadcasting
# The most confusing and useful thing about numpy
# np can perfom operations on different shapes by inference and by expanding
op1 = np.array([i for i in range(9)]).reshape(3,3)
op2 = np.array([[1,2,3,]])
op3 = np.array([1,2,3]) 
print('Broadcasting')
print('op1 is...')
pp.pprint(op1)
print(f'op1 has dim {op1.shape}') 
print(f'op2 is {op2} with dim {op2.shape}') 
print(f'op3 is {op3} with dim {op3.shape}') 
print('\n')

print('op1+op2 equals... ')
pp.pprint(op1+op2)
print('Notice how op2 is added to EVERY ROW of op1')
print('\n')

print('op1+op2.T equals... ')
pp.pprint(op1+op2.T)
print('Notice how op2 is added to EVERY COLUMN of op1')
print('\n')
# The vector that gets broadcasted is that with the (1, x) dimension

# Tiling equivalent to repmat

# Expand. Adds extra dimension

# Squeeze. Removes extra dimension

# Pairwise distance
print('Pairwise distance')
distances = scipy.spatial.distance.cdist(np.array([[1,2,3]]), np.array([[1,2,4]]))
print(distances)

# TDQM is equivalent to tic and toc

# Vectorsation >> for loops. Use numpy operations!!
# If still slow, use OpenBLAS which speeds up numpy 
# https://github.com/xianyi/OpenBLAS