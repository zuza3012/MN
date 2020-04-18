import numpy as np
import scipy.linalg
A = np.array([(9,8,-2,2,-2), (7,-3,-2,7,2), (2,-2,2,-7,6), (4,8,-3,3,-1), (2,2,-1,1,4)])




def LU(N,A):
    P, L_, U_ = scipy.linalg.lu(A)

    L = np.zeros((N, N), dtype=float, order='C')
    U = np.zeros((N, N), dtype=float, order='C')

    w = A.shape
    if(w[0] != w[1]):
        print("It's not square matrix")
        return



    for i in range(N):
        for j in range(i, N):
            sum = 0;
            for k in range(0, i):
                # liczymy sume
                sum += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - sum

        for j in range(i, N):

            if i == j:
                L[i, i] = 1
            else:
                sum = 0
                for k in range(0, i):
                    sum += L[j, k] * U[k, i]
                L[j, i] = (A[j, i] - sum) / U[i, i]
    print("Matrix A:")
    print(A)

    print("Matrix L:")
    print(L)

    print("Matrix U:")
    print(U)

    print("Matrix L_ scipy:")
    print(L_)

    print("Matrix U_ scipy:")
    print(U_)

    precision = 1e-9
    u_arr = np.zeros((N, N), dtype=bool)
    l_arr = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(N):
            if np.abs(U[i, j] - U_[i, j] < precision):
                u_arr[i, j] = True
            if np.abs(L[i, j] - L_[i, j] < precision):
                l_arr[i, j] = True

    # sprawdzenie czy U_ i U daja podobne wyniki z dokladnoscia do precision
    print(u_arr)
    # sprawdzenie czy L_ i L daja podobne wyniki z dokladnoscia do precision
    print(l_arr)



# # N = input ("Number of equations:")
# # N = int(N)
# # B = np.zeros((N,N),dtype=float, order='C')
# # for i in range(N):
# #     eq = input("Write matrix coefficients in row " + str(i))
# #     B[i,:] = list(map(int, eq))
# print("test")
# print(B)
N = 5
LU(N,A)
