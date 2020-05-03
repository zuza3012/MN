# aproksymacja danych wielomianem dowolnego stopnia
import numpy as np
import matplotlib.pyplot as plt

X = np.array([(1, 2, 3, 4, 5)])
Y = np.array([(52, 5, -5, -40, 10)])


def approx(X, Y, m):
    if len(X[0,:]) != len(Y[0,:]):
        print("Sizes of X[0,:] and Y[0,:] should be the same!")
        return
    m = m + 1
    xsum = np.zeros([2 * m])
    w, h = m, m
    A = [[0 for x in range(w)] for y in range(h)]
    b = np.zeros([m])
    n = X.size

    for j in range(0, 2 * m, 1):
        sum = 0
        for i in range(0, n, 1):
            sum += X[0, i] ** j
        xsum[j] = sum
    n = X.size

    for j in range(0, m, 1):
        for i in range(0, m, 1):
            A[i][j] = xsum[i + j]
        bb = 0
        for k in range(0, n, 1):
            bb += X[0, k] ** j * Y[0, k]
        b[j] = bb

    print("b:", b)
    print("A",A)

    a = np.linalg.solve(A, b)
    print("wspolczynniki", a)
    return a


w = approx(X, Y, 3)

p1 = np.poly1d(list(reversed(w)))

x_new = np.linspace(1, 5, 300)
p = np.zeros([x_new.size])

for i in range(x_new.size):
    p[i] = p1(x_new[i])


plt.plot(X[0,:],Y[0,:], 'o', color='red', )
plt.plot(x_new,p, color='blue', )

plt.ylabel('y')
plt.xlabel('x')
plt.show()


