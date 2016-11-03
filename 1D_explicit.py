import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# -------------- parameter ----------------
n = 4
STEP = 100
dt = 0.1
dx = 1.0
k = 1.0
K = k * dt / (dx * dx)

S = np.zeros(n)

T = np.zeros(n)
T[n-1] = 100

# -------------- matrix ----------------
mat = np.zeros([n, n])
for i in range(1, n-1):
    mat[i][i-1] = 1
    mat[i][i] = -2
    mat[i][i+1] = 1

fig = plt.figure()
ims = [plt.plot(T, 'r')]

for i in range(STEP):
    next = copy.deepcopy(T)

    for j in range(n):
        res = 0.0
        for k in range(n):
            res = res + mat[j][k] * T[k]
        next[j] = next[j] + res * K + S[j] * dt
    T = copy.deepcopy(next)
    ims.append(plt.plot(T, 'r'))

plt.title('1D - explicit')
plt.ylabel('T')
plt.xlabel('x')
ani = animation.ArtistAnimation(fig, ims)
# ani.save("n4_S4.gif", writer="imagemagick")

plt.show()
