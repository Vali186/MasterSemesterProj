import numpy as np
import math
from scipy.linalg import block_diag



def Pmat(k,n):
    if (n%k != 0):
        raise ValueError(f"Error: {k} does not divide {n}. Please provide valid inputs for permutation matrix.")
    mat = np.zeros((n,n))
    for i in range(n):
        mat[i,int(i%k*n/k+math.floor(i/k))] = 1
    return mat


def GSmat(L,R,b):
    P = Pmat(b,L.shape[1])
    mat = np.linalg.multi_dot([P.T,L,P,R])
    return mat

def Cailey(K):
    if not np.allclose(K, -K.T):
        return "Matrix is not skew-symmetric."
    I = np.identity(K.shape[0])
    Q = np.dot(I+K,np.linalg.inv(I-K))
    return Q



# print(Pmat(3,9))

K = np.array([
    [0, -1, 0],
    [1, 0, -1],
    [0, 1, 0]
])

B = Cailey(K)
L = block_diag(B,B,B)
R = L
P = Pmat(3,9)

A = GSmat(L,R,3)
print(B)





