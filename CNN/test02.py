import numpy as np

a = np.arange(16).reshape(2,2,2,2)
b = np.reshape(a,(-1,8))
print(a)
print(b)