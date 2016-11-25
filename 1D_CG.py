import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from matplotlib import cm
from matplotlib import animation

def cgm(A, b):
    x = np.zeros(len(b))
    r0 = b - np.dot(A, x)
    p = r0

    for i in range(200):
        a = float(np.dot(r0.T, r0) / np.dot(np.dot(p.T, A), p))
        x = x + p * a
        r1 = r0 - np.dot(A * a, p)

        if linalg.norm(r1) < 1.0e-10:
            return x
        
        b = float(np.dot(r1.T, r1) / np.dot(r0.T, r0))
        p = r1 + b * p
        r0 = r1

    return x

n = 10
STEP = 100

k = 1.0
delta_t = 0.66
delta_x = 1.0

T = np.zeros(n)
T[n-1] = 100
S = np.zeros(n)

A = np.zeros([n, n])
A[0][0] = 1
A[n-1][n-1] = 1
for i in range(1, n-1):
    A[i][i-1] = - k * delta_t / (delta_x * delta_x)
    A[i][i] = (1.0 + 2 * k * delta_t / (delta_x * delta_x))
    A[i][i+1] = - k * delta_t / (delta_x * delta_x)

fig = plt.figure()
ims = [plt.plot(T, 'r')]

for i in range(STEP):
    b = copy.deepcopy(T)
    x = cgm(A, b)
    T = copy.deepcopy(x)
    ims.append(plt.plot(T, 'r'))

plt.title('CG method')
plt.ylabel('T')
plt.xlabel('x')
plt.xlim([0, n-1])

ani = animation.ArtistAnimation(fig, ims)
# ani.save('cgm.gif', writer='imagemagick')
plt.show()
