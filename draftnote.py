import numpy as np

# print(np.ones((2,2)) * 2 ** 3)
# print(np.ones((2, 2)) > 2)
# print(np.ones(5)[-3:])
#
# a = [np.ones((2, 2)), np.ones((2, 2))]
# b = np.concatenate(a, axis=1)
# print(b.reshape((-1)))
# print(b > 2)

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.mean(a, axis=1,keepdims=True)
print(b)
print(a - b)
print(np.tile(b, 20))
