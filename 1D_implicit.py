import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# ---------- parameter ----------
n = 4
STEP = 100
A = np.zeros([n, n])
A[0][0] = 1
A[n-1][n-1] = 1

k = 1.0
delta_t = 0.66
delta_x = 1.0

T = np.zeros(n)
T[n-1] = 100
S = np.zeros(n)

for i in range(1, n-1):
    A[i][i-1] = - k * delta_t / (delta_x * delta_x)
    A[i][i] = (1.0 + 2 * k * delta_t / (delta_x * delta_x))
    A[i][i+1] = - k * delta_t / (delta_x * delta_x)
LU = linalg.lu_factor(A)

fig = plt.figure()
ims = [plt.plot(T, 'r')]

for i in range(STEP):
    T = linalg.lu_solve(LU, T + S * delta_t)
    ims.append(plt.plot(T, 'r'))

plt.title('STEP:'+str(STEP)+' delta_t:'+str(delta_t)+' delta_x:'+str(delta_x)+' k:'+str(k))
plt.ylabel('T')
plt.xlabel('x')
plt.xlim([0, n-1])

ani = animation.ArtistAnimation(fig, ims)
plt.show()
# ani.save("n4_S5.gif", writer="imagemagick")
