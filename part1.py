import numpy as np
from heapq import nsmallest

matrix = np.around(np.random.rand(5, 6) * 9)
smallest = nsmallest(3, matrix[4, :])
index = np.argwhere(matrix[4, :]==smallest[2])

print('3rd smallest number in 5th row was ',int(smallest[2]))
matrix[4, index[0]] = 10
print(matrix)
