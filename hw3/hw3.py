from scipy import sparse
import numpy as np
import math

def gridLaplacian(m1, m2):
    def index(i1,i2):
        return i1 + m1*i2
    def degree(i1,i2):
        return (1 if i1 > 0 else 0) + (1 if i1+1 < m1 else 0) \
             + (1 if i2 > 0 else 0) + (1 if i2+1 < m2 else 0)
    L = sparse.dok_matrix((m1*m2, m1*m2))
    for i2 in range(m2):
        for i1 in range(m1):
            i = index(i1,i2)
            L[i,i] = degree(i1,i2)
    for i2 in range(m2):
        for i1 in range(m1-1):
            i = index(i1,  i2)
            j = index(i1+1,i2)
            L[i,j] = -1
            L[j,i] = -1
    for i2 in range(m2-1):
        for i1 in range(m1):
            i = index(i1,i2)
            j = index(i1,i2+1)
            L[i,j] = -1
            L[j,i] = -1
    return L

def incompleteCholesky(A):
    m = A.shape[0]
    A 
    R = sparse.dok_matrix((m,m))
    for i in range(m):
        R[i,i] = math.sqrt(A[i,i])
        R[i+1:,i] = A[i+1:,i]/R[i,i]
        b = A[i+1:,i]
        b = (b@b.conj().T)/A[i,i]
        # print(b,b.shape,"khatam")
        for j in range(m-i-1):
            # print(i)
            (x,y) = A[i+1:,i+1:].getrow(j).nonzero()
            # print(b)
            # print(x,y,j)
            for k in range(len(x)):
                A[i+1+j,y[k]+i+1] = A[j+i+1,y[k]+i+1]-b[j,y[k]]
        # print(A,"A khatam")
    print(R)
    # print(A)


m1 = 2
m2 = 2
L = gridLaplacian(m1,m2)
A = L + 20*sparse.eye(m1*m2)/(m1**2 + m2**2)
# test = sparse.dok_matrix([[3,0,-1,-1,0,-1],[0,2,0,-1,0,0],[-1,0,3,0,-1,0],[-1,-1,0,2,0,-1],[0,0,-1,0,3,-1],[-1,0,0,-1,-1,4]],dtype=np.float32)
incompleteCholesky(test)
# print(A)