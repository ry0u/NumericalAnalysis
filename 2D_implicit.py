import numpy as np
import copy
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import time

STEP = 100
delta_t = 0.1
delta_x = 1
k = 1
S = 10
EPS = 1e-1

K = k * delta_t / (delta_x * delta_x)
n = 3
t = np.zeros(n * n)
for i in range(n):
    t[i * n + n - 1] = 100
    t[(n-1)*n + i] = 100

A = np.zeros([n * n, n * n])
for i in range(n*n):
    A[i][i] = 1
for i in range(n + 1, n * n - n - 1):
    if i % n == n - 1 or i % n == 0:
        continue
    A[i][i-1] = - k * delta_t / (delta_x * delta_x)
    A[i][i+1] = - k * delta_t / (delta_x * delta_x)
    A[i][i] = (1.0 + 4 * k * delta_t / (delta_x * delta_x))
    A[i][i-n] = - k * delta_t / (delta_x * delta_x)
    A[i][i+n] = - k * delta_t / (delta_x * delta_x)

print A
LU = linalg.lu_factor(A)

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

tt = []
for i in range(n):
    tt.append(t[i*n:i*n+n])

ims = ax.plot_wireframe(X, Y, tt, rstride=1, cstride=1)
tstart = time.time()

for step in range(STEP):
    next = linalg.lu_solve(LU, t)
    t = copy.deepcopy(next)
    oldcol = ims

    tt = []
    for i in range(n):
        tt.append(t[i*n:i*n+n])

    ims = ax.plot_wireframe(X, Y, tt, rstride=1, cstride=1)
    ax.collections.remove(oldcol)
    # plt.savefig('./test4/' + '{0:03d}'.format(step) + '.png')
    plt.pause(0.0001)
