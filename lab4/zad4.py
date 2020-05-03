# Najwieksza wartosc wlasna i wektor jej odpowiadajacy dla macierzy n x n
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

A = np.array([(6, 5, -5), (2, 6, -2), (2, 5, -1)])

x0 = np.array([[2, 2, 1]]).T


def calculate_lambda(A, x0, max):
    i = 1
    y = np.zeros((3, 1), dtype=float, order='C')
    yy = np.zeros((3, 1), dtype=float, order='C')
    I = []
    l = []
    precision = 0.01
    while i != max:
        yy = y
        I.append(i)
        if i == 1:
            y = x0
            y = y / y[0]
        y = np.matmul(A, y)
        l.append(y[0])
        y = y / y[0]
        if np.abs(yy[0] - y[0]) < precision and np.abs(yy[1] - y[1]) < precision and np.abs(yy[2] - y[2]) < precision:
            print("Lambda max = ", l[i-1])
            print("Eigenvector: ")
            print(y)
            break
        i = i + 1

    return l, i , I

print("Matrix A")
print(A)

x, i, y = calculate_lambda(A, x0, 20)

plt.plot(y,x, 'o', color='blue', )
plt.ylabel('lambda')
plt.show()
print()
print("LA.eig(A)")
w, v = LA.eig(A)
print("Eigenvalues: ")
print(w)
print("Eigenvectors")
print(v)