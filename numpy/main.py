import numpy as np

"""

my_list = np.array([1,2,3]) # list but with C performence

my_list *= 2 # multiply each element of an array by 2

print(type(my_list))

print(my_list)
 
"""

"""

array = np.array([
    ["A", 'B', 'C'],
    ["A", 'B', 'C'],
    ["A", 'B', 'C'],
    ])

print(array.ndim)
print(array.shape)
print(array[0, 0]) # multidimensional indexing instead of chain indexing

word = array[0, 0] + array[1, 2]
print(word)

"""
"""
array = np.array([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12]])

print(array[0:4:2]) #array[start:end:step]
print(array[::2]) 
print(array[::-1])

print(array[:, 1:4]) # I select all rows, and elements from 1 to 4 column

"""


# #scalar arithmetic

# arr = np.array([1,2,3,4])

# print(arr * 2)

# # Vectorized math funcs

# array = np.array([1.5, 2.34, 3.67])

# print(np.sqrt(array))
# print(np.round(arr))


# #Element-wise arithmetic

# array1 = np.array([1,2,3])
# array2 = np.array([2,4,6])

# print(array1 + array2)

# # Comparison operators

# scores = np.array([91, 55, 100, 73, 82, 64])

# print(scores == 100)

# Broadcasting


# array1 = np.array([[1,2,3,4]])

# array2 = np.array([[1], [2], [3], [4]])

# print(array1 * array2)

# Random numbers

# rng = np.random.default_rng(seed=1)

# print(rng.integers(1, 7, size=(3, 2)))

print(np.round(np.random.uniform(low=-1, high=1, size=(5,5))))

