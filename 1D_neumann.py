import copy
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
from matplotlib import animation

n = 10
STEP = 100
delta_t = 0.1
delta_x = 1.0
k = 1.0
K = k * delta_t / (delta_x * delta_x)

mat = np.zeros([n, n])
mat[0][0] = 1 - K
mat[0][1] = K
for i in range(1, n-1):
    mat[i][i-1] = K
    mat[i][i] = (1 - 2 * K)
    mat[i][i+1] = K

mat[n-1][n-2] = K
mat[n-1][n-1] = 1 - K

t = np.zeros(n)
t[n-1] = 100

fig = plt.figure()
ims = [plt.plot(t, 'r')]

for i in range(STEP):
    next = np.zeros(n)
    for j in range(n):
        res = 0.0
        for k in range(n):
            res = res + mat[j][k] * t[k]
        next[j] = next[j] + res

    t = copy.deepcopy(next)
    ims.append(plt.plot(t, 'r'))

plt.title('Neumann')
plt.ylabel('T')
plt.xlabel('x')
ani = animation.ArtistAnimation(fig, ims)
plt.show()
# ani.save('1D_Neumann.gif', writer='imagemagick')
