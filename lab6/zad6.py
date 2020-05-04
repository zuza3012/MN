# rysuje atraktor lorentza - metoda RK4
import numpy as np
import matplotlib.pyplot as plt


def attractor(x):
    sigma = 10
    b = 8 / 3
    r = 28
    return np.array([sigma * (x[1] - x[0]), -x[0] * x[2] + r * x[0] - x[1], x[0] * x[1] - b * x[2]])


def rungekutta4(f, x0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    nt = t.size

    nx = x0.size
    x = np.zeros((nx, nt))

    x[:, 0] = x0

    for k in range(nt - 1):
        k1 = dt * f(t[k], x[:, k])
        k2 = dt * f(t[k] + dt / 2, x[:, k] + k1 / 2)
        k3 = dt * f(t[k] + dt / 2, x[:, k] + k2 / 2)
        k4 = dt * f(t[k] + dt, x[:, k] + k3)
        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[:, k + 1] = x[:, k] + dx
    return x, t


f = lambda t, x: attractor(x)

x0 = np.array([1, 1, 1])
t0 = 0
tf = 100
dt = 0.01

x, t = rungekutta4(f, x0, t0, tf, dt)

X = x[0, :]
Y = x[1, :]
Z = x[2, :]

ax = plt.axes(projection='3d')
ax.plot3D(X, Y, Z, 'blue')
ax.set_xlabel('x', fontsize=10, rotation = 0)
ax.set_ylabel('y', fontsize=10, rotation = 0)
ax.set_zlabel('z', fontsize=10, rotation = 0)

plt.show()
