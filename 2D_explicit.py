import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import time

STEP = 100
delta_t = 0.3
delta_x = 1
k = 1
S = 10
EPS = 1e-1

K = k * delta_t / (delta_x * delta_x)

n = 20
mat = [[0,  1, 0],
       [1, -4, 1],
       [0,  1, 0]]

t = np.zeros([n, n])
for i in range(n):
    t[i][n-1] = 100
    t[n-1][i] = 100

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

ax.set_xlim(0, n-1)
ax.set_ylim(0, n-1)
ax.set_zlim(0, 100)

# ax.view_init(30, 240)

x = np.arange(0, n, 1)
y = np.arange(0, n, 1)
X, Y = np.meshgrid(x, y)

ims = ax.plot_wireframe(X, Y, t, rstride=1, cstride=1)
tstart = time.time()

for step in range(STEP):
    next = copy.deepcopy(t)

    for i in range(1, n-1):
        for j in range(1, n-1):
            res = 0.0
            for ty in range(-1, 2):
                for tx in range(-1, 2):
                    res = res + t[i+ty][j+tx] * mat[1+ty][1+tx]

            next[i][j] = next[i][j] + res * K

    flag = 1
    for i in range(n):
        for j in range(n):
            if abs(t[i][j] - next[i][j]) < EPS:
                continue
            else:
                flag = 0

    if flag == 1:
        print 'not diff ', step
        break

    t = copy.deepcopy(next)
    oldcol = ims
    ims = ax.plot_wireframe(X, Y, t, rstride=1, cstride=1)
    ax.collections.remove(oldcol)

    # plt.draw()
    # plt.savefig('./test9/' + '{0:03d}'.format(step) + '.png')
    plt.pause(0.0001)

