import tensorflow as tf
import numpy as np

Q = np.zeros(shape=(6,6))
R = np.array([[-1,-1,-1,-1,0,-1],
              [-1,-1,-1,0,-1,100],
              [-1,-1,-1,0,-1,-1],
              [-1,0,0,-1,0,-1],
              [0,-1,-1,0,-1,100],
              [-1,0,-1,-1,0,100]])

print("更新前的Q值：\n",Q)
print("R回报：\n",R)
for state in range(6):
    for action in range(6):
        Q[state][action] = R[state][action] + 0.9*np.max(Q[action])
print("更新后的Q值：\n",Q)


