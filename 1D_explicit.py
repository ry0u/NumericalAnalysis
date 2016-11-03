import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

STEP = 100
delta_t = 0.1
delta_x = 1.0
k = 1.0
S = [0.0, 0.0, 0.0, 0.0]
K = k * delta_t / (delta_x * delta_x)
title = 'STEP:' + str(STEP) + ' delta_t:' + str(delta_t) + ' delta_x:' + str(delta_x) + ' k:' + str(k)

n = 4
mat = [[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
t = [0.0, 0.0, 0.0, 100.0]

fig = plt.figure()
ims = [plt.plot(t, 'r')]

line = [t]

for i in range(STEP):
    next = copy.deepcopy(t)

    for j in range(n):
        res = 0.0
        for k in range(n):
            res = res + mat[j][k] * t[k]
        next[j] = next[j] + res * K + S[j] * delta_t
    t = copy.deepcopy(next)

    ims.append(plt.plot(t, 'r'))

plt.title(title)
plt.ylabel('T')
plt.xlabel('x')
ani = animation.ArtistAnimation(fig, ims)
# ani.save("n4_S4.gif", writer="imagemagick")
