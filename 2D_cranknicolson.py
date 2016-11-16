import copy
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# ------------------
# parameter
# ------------------
n = 20
STEP = 100
delta_t = 0.3
delta_x = 1.0
k = 1.0
K = k * delta_t /  (delta_x * delta_x)

t = np.zeros(n * n)
for i in range(n):
    t[i * n + n - 1] = 100
    t[(n-1)*n + i] = 100

# ------------------
# matrix
# ------------------
A = np.zeros([n*n, n*n])
for i in range(n * n):
    A[i][i] = 1

for i in range(n + 1, n * n - n - 1):
    if i % n == n - 1 or i % n == 0:
        continue
    A[i][i-n] = -K
    A[i][i-1] = -K
    A[i][i] = 2 + 4 * K
    A[i][i+1] = -K
    A[i][i+n] = -K

LU = linalg.lu_factor(A)

mat = np.zeros([n*n, n*n])
for i in range(n * n):
    mat[i][i] = 1

for i in range(n + 1, n * n - n - 1):
    if i % n == n - 1 or i % n == 0:
        continue
    mat[i][i-n] = K
    mat[i][i-1] = K
    mat[i][i] = 2 - 4 * K
    mat[i][i+1] = K
    mat[i][i+n] = K

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

for i in range(STEP):
    next = np.zeros(n * n)
    oldcol = ims
    
    for j in range(n * n):
        res = 0.0
        for k in range(n * n):
            res = res + mat[j][k] * t[k]
        next[j] = next[j] + res

    t = linalg.lu_solve(LU, next)

    tt = []
    for j in range(n):
        tt.append(t[j*n:j*n+n])

    ims = ax.plot_wireframe(X, Y, tt, rstride=1, cstride=1)
    ax.collections.remove(oldcol)
    # plt.savefig('./2D/n20_1/' + '{0:03d}'.format(i) + '.png')
    plt.pause(0.0001)
