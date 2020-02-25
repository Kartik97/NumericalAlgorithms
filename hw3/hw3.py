from scipy import sparse
from scipy.sparse import linalg,SparseEfficiencyWarning
import numpy as np
import math
import time
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
    m = A.shape[1]
    R = sparse.dok_matrix((m,m))
    for i in range(m):
        R[i,i] = math.sqrt(A[i,i])
        R[i,i+1:] = A[i,i+1:]/R[i,i]
        # R[i+1:,i] = A[i+1:,i]/R[i,i]
        b = A[i,i+1:]
        # b = A[i+1:,i]
        b = (b.conj().T@b)/A[i,i]
        # b = (b@b.conj().T)/A[i,i]
        for j in range(m-i-1):
            (x,y) = A[i+1:,i+1:].getrow(j).nonzero()
            for k in range(len(x)):
                A[i+1+j,y[k]+i+1] = A[j+i+1,y[k]+i+1]-b[j,y[k]]
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
    r = b
    p = r
    norms = []
    for _ in range(A.shape[0]):
        if ((np.linalg.norm(r,ord=2)/np.linalg.norm(b,ord=2)) <= tolerance):
            break
        alpha = (r.conj().T@r)/(p.conj().T@A@p)
        x = x + alpha*p
        r_n = r - alpha*(A@p)
        beta = (r_n.conj().T@r_n)/(r.conj().T@r)
        p = r_n + beta*p
        r = r_n
        norms.append(np.linalg.norm(r,ord=2)/np.linalg.norm(b,ord=2))
    return x,norms

def preconditionedConjugateGradient(A, b, R, tolerance):
    Rinv = sparse.linalg.inv(R.tocsc())
    y,norms = conjugateGradient(Rinv.conj().T@A@Rinv,Rinv.conj().T@b,tolerance)
    x = linalg.spsolve_triangular(R.tocsr(),y,lower=False)
    return x,norms

def drawGraph(iterNorms,cg,precg):
    plt.figure(0)
    ax = plt.gca()
    ax.plot(range(len(iterNorms)),abs(np.log(iterNorms)),label="Stationary Method Iterations",color="blue")
    ax.plot(range(len(cg)),abs(np.log(cg)),label="Conjugate Gradient Iterations",color="red")
    ax.plot(range(len(precg)),abs(np.log(precg)),label="Preconditioned Conjugate Gradient Iterations",color="green")
    ax.legend()
    ax.set_xlabel("Number of iterations (n)")
    ax.set_ylabel("Error: ||r|| / ||b|| (On negative log scale)")
    plt.show()

m1 = 2
m2 = 2
L = gridLaplacian(m1,m2)
A = L + 20*sparse.eye(m1*m2)/(m1**2 + m2**2)
b = 2.5*np.ones((4,1))
# test = sparse.dok_matrix([[3,0,-1,-1,0,-1],[0,2,0,-1,0,0],[-1,0,3,0,-1,0],[-1,-1,0,2,0,-1],[0,0,-1,0,3,-1],[-1,0,0,-1,-1,4]],dtype=np.float32)
st = time.time()
R = incompleteCholesky(A)
print(time.time()-st)
# print(R)
# SOL = all 1
x0 = np.zeros((4,1),dtype=float)
st = time.time()
x1 = icIteration(A,b,R,x0)
print(time.time()-st)

iterNorms = []
x = np.zeros((4,1),dtype=float)
while True:
# for i in range(A.shape[0]):
    if((np.linalg.norm(A@x-b,ord=2)/np.linalg.norm(b,ord=2)) < 1e-15):
        break
    x = icIteration(A,b,R,x)
    iterNorms.append(np.linalg.norm(A@x-b,ord=2)/np.linalg.norm(b,ord=2))
# print(iterNorms)
# print(x1)
st = time.time()
x1,norms1 = conjugateGradient(A,b,1e-20)
print(time.time()-st)
st = time.time()
x2,norms2 = preconditionedConjugateGradient(A,b,R,1e-20)
print(time.time()-st)
# print(x1,"\n",norms1)
# print(x2,"\n",norms2)

drawGraph(iterNorms,norms1,norms2)