import numpy as np
from functools import reduce
# lambdas

add_1 = lambda x, y: x + y

print(add_1(1, 2))

# examples
arr = np.arange(10, 100, 5)
squares = np.array(list(map(lambda x: x**2, arr)))
div_by_15 = np.array(list(filter(lambda x: x % 15 == 0, arr)))
sum_of_numbers = reduce(lambda acc, x: acc + x, div_by_15)