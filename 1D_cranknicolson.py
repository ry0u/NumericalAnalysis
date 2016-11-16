import copy
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import animation

# ------------------
# parameter
# ------------------
n = 4
STEP = 30
delta_t = 0.1
delta_x = 1.0
k = 1.0
K = k * delta_t / (delta_x * delta_x)

t = np.zeros(n)
t[n-1] = 100
S = np.zeros(n)


# ------------------
# matrix
# ------------------
A = np.zeros([n, n])
A[0][0] = 1
A[n-1][n-1] = 1
for i in range(1, n-1):
    A[i][i-1] = -K
    A[i][i] = 2 + 2 * K
    A[i][i+1] = -K

LU = linalg.lu_factor(A)
mat = np.zeros([n, n])
mat[0][0] = 1
mat[n-1][n-1] = 1
for i in range(1, n-1):
    mat[i][i-1] = K
    mat[i][i] = 2 - 2 * K
    mat[i][i+1] = K

fig = plt.figure()
ims = [plt.plot(t, 'r')]

for i in range(STEP):
    next = np.zeros(n)

    for j in range(n):
        res = 0.0
        for k in range(n):
            res = res + mat[j][k] * t[k]
        next[j] = next[j] + res

    t = linalg.lu_solve(LU, next)

    ims.append(plt.plot(t, 'r'))

plt.title('crank-nicolson')
plt.ylabel('T')
plt.xlabel('x')
ani = animation.ArtistAnimation(fig, ims)
plt.show()
# ani.save("1D_002.gif", writer="imagemagick")
