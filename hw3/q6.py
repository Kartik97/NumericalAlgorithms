from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import math
import matplotlib.pyplot as plt

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
    T = A.copy()
    m = A.shape[1]
    R = sparse.dok_matrix((m,m))
    for i in (range(m)):
        R[i,i] = math.sqrt(T[i,i])
        R[i,i+1:] = T[i,i+1:]/R[i,i]
        (x,y) = T[i+1:,i+1:].nonzero()
        for k in range(len(x)):
            if(T[i,y[k]+i+1]==0 or T[i,x[k]+i+1]==0):
                pass
            elif(x[k]<=y[k]):
                T[x[k]+i+1,y[k]+i+1] = T[x[k]+i+1,y[k]+i+1]-T[i,x[k]+i+1]*T[i,y[k]+i+1]/T[i,i]
    return R

def icIteration(A, b, R, x):
    M = R.conj().T@R
    N = M-A
    t = N@x + b
    y = linalg.spsolve_triangular(R.conj().T.tocsr(),t,lower=True)
    sol = linalg.spsolve_triangular(R.tocsr(),y,lower=False)
    return sol

def conjugateGradient(A,b,tolerance):
    x = sparse.dok_matrix((A.shape[0],1))
    r = np.reshape(b.copy(),(A.shape[0],1))
    p = np.reshape(r.copy(),(A.shape[0],1))
    norms = []
    while True:
        if ((np.linalg.norm(r,ord=2)/np.linalg.norm(b,ord=2)) <= tolerance):
            break
        if(np.linalg.norm(r,ord=2)/np.linalg.norm(b,ord=2) != 0):
            norms.append(np.linalg.norm(r,ord=2)/np.linalg.norm(b,ord=2))
        alpha = (r.conj().T@r)/(p.conj().T@A@p)
        x = x + alpha*p
        r_n = r - alpha*(A@p)
        beta = (r_n.conj().T@r_n)/(r.conj().T@r)
        p = r_n + beta*p
        r = r_n
    return x,norms

def preconditionedConjugateGradient(A, b, R, tolerance):
    Rinv = sparse.linalg.inv(R.tocsc())
    y,norms = conjugateGradient(Rinv.conj().T@A@Rinv,Rinv.conj().T@b,tolerance)
    x = linalg.spsolve_triangular(R.tocsr(),y,lower=False)
    return x,norms

def drawGraph(iterNorms,cg,precg):
    plt.figure(0)
    ax = plt.gca()
    ax.plot(range(len(iterNorms)),abs(np.log(iterNorms)),label="Stationary Method Iterations")
    ax.plot(range(len(cg)),abs(np.log(cg)),label="Conjugate Gradient Iterations")
    ax.plot(range(len(precg)),abs(np.log(precg)),label="Preconditioned Conjugate Gradient Iterations")
    ax.legend()
    ax.set_xlabel("Number of iterations (n)")
    ax.set_ylabel("Error: ||r|| / ||b|| (On negative log scale)")
    plt.show()