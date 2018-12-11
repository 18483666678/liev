import numpy as np
#onehot形式
def _one_hot():
    z = np.zeros(shape=(4,10))
    for i in range(4):
        index = int("1 2 3 4".split(" ")[i])
        z[i][index] += 1
    return z
if __name__ == '__main__':
    result = _one_hot()
    print(result)