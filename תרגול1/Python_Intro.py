import numpy as np

'''
DANIEL KIGLI - TAU - VIDEO PROCESSING 2020
'''

# Function that return
# dot product of two vector array.
def dotProduct(vec_A, vec_B):
    # if I will uncomment the next line, the value of 'i' in this function will effect it's value outside the function
    # global i

    num_of_elments_in_A = len(vec_A)
    num_of_elments_in_B = len(vec_B)

    # We need to validate that the two vector have the same size
    assert (num_of_elments_in_A == num_of_elments_in_B)

    product_result = 0

    # Loop for calculate cot product
    for i in range(0, num_of_elments_in_A):
        product_result += + vec_A[i] * vec_B[i]

    return product_result


# for loops
for i in range(3):
    print(i)

for i in range(1, 3):
    print(i)

for i in [4, 5, 6]:
    print(i)

# Define a toy array
vector_a = np.array([1, 2, 3], dtype=np.int)
print("Vector values are: ", vector_a)
# Prints the same line
print("Vector data type is: {}".format(vector_a.dtype))
print("Vector data type is: " + str(vector_a.dtype))
print("Vector data type is:", vector_a.dtype)

# Define a second vector and do a dot product between them
vector_b = np.array([1, 1, 1], dtype=np.int)

# Let's try to do a dot product between the vector
r1 = np.dot(vector_a, vector_b)
r2 = vector_a.dot(vector_b)
r3 = dotProduct(vector_a, vector_b)  # My function!
if r1 == r2 == r3:
    print('yay')
else:
    assert False  # Code will throw an assertion if we reached here

# A few additional examples
type(vector_a)  # Tells you the type of the variable
np.shape(vector_a)  # Prints the shape of the numpy array
vector_a.shape  # Prints the shape of the numpy array


vector_c = np.arange(12)
matrix_c = vector_c.reshape((4, 3))

# We can see the array in PyCharm's variable viewer

# Dot product between the vector and matrix
np.dot(matrix_c, vector_b)