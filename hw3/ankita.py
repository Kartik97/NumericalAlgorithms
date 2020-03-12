import math
from tqdm import tqdm
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
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
    a = A.copy()
    n=a.shape[0]
    for k in (range(n)):
        a[k,k] = math.sqrt(a[k,k])
        for i in range ((k+1),n):
            if (a[i,k]!=0):
                a[i,k] = a[i,k]/a[k,k]
        for i in range((k+1),n):
            if(a[i,k]==0):
                pass
            for j in (a[k+1:,k+1:].getrow(i-k-1).nonzero()[1]):
                if(a[k,k+1+j] == 0):
                    pass
                a[i,k+1+j] -= a[i,k]*a[k,k+1+j]/a[k,k]
    return sparse.triu(a.T)

def conjugateGradient(A, b, tolerance):
    r1=b.copy()
    p=r1.copy()
    x=np.zeros((b.shape))
    lst=[]
    while(True):
        alpha=(r1.T.dot(r1))/((p.T.dot(A.toarray())).dot(p))
        x = x+alpha *(p)
        r2=r1-alpha*A.dot(p)
        beta=(r2.T.dot(r2))/(r1.T.dot(r1))
        lst.append(np.linalg.norm(r2)/ np.linalg.norm(b))
        p=r2+beta*p
        r1=r2
        
        if (np.linalg.norm(r2)/ np.linalg.norm(b)<tolerance):
            break
    return(x,lst)

def preconditionedConjugateGradient(A, b, r, tolerance):
    r_inv=sparse.linalg.inv(r)
    (x,y)=conjugateGradient((r_inv.T.dot(A)).dot(r_inv), (r_inv.T).dot(b), tolerance)
    ans=sparse.linalg.spsolve_triangular(r, x,lower=False)
    return (ans,y)

def icIteration(A, b, R, x):
    M=R.T.dot(R)
    N=M-A
    z=N.dot(x)+b
    y=sparse.linalg.spsolve_triangular(R.T.tocsr(), z,lower=True)
    ans=sparse.linalg.spsolve_triangular(R.tocsr(), y,lower=False)
    return ans

m1 = 10
m2 = 10
L = gridLaplacian(m1,m2)
A = L + 20*sparse.eye(m1*m2)/(m1**2 + m2**2)
# b = np.ones((100,1))
b = np.random.randn(A.shape[1])
# A = sparse.dok_matrix([[3.0,0,-1,-1,0,-1],[0,2,0,-1,0,0],[-1,0,3,0,-1,0],[-1,-1,0,2,0,-1],[0,0,-1,0,3,-1],[-1,0,0,-1,-1,4]],dtype=np.float32)
# R = incompleteCholesky(A)
# b = np.ones(A.shape[1])
# print(R.toarray())
# st = time.time()
R = incompleteCholesky(A)
# print(time.time()-st)
# print(R)
# SOL = all 1
x0 = np.zeros(A.shape[1],dtype=float)
# st = time.time()
x1 = icIteration(A,b,R,x0)
# print(time.time()-st)

iterNorms = []
x = np.zeros(A.shape[1],dtype=float)
# while True:
for i in range(A.shape[0]):
    if((np.linalg.norm(A@x-b,ord=2)/np.linalg.norm(b,ord=2)) < 1e-15):
        break
    x = icIteration(A,b,R,x)
    iterNorms.append(np.linalg.norm(A@x-b,ord=2)/np.linalg.norm(b,ord=2))
# # print(iterNorms)
# # print(x1)
# st = time.time()
x1,norms1 = conjugateGradient(A,b,1e-20)
# print(time.time()-st)
# st = time.time()
x2,norms2 = preconditionedConjugateGradient(A,b,R,1e-20)
# print(time.time()-st)
print(len(norms1))
print(len(norms2))
print(len(iterNorms))

plt.figure()
plt.plot(range(len(norms1)),np.log(norms1),label="Conjugrate Gradient")
plt.plot(range(len(norms2)),np.log(norms2),label="Preconditioned Conjugrate Gradient")
plt.plot(range(len(iterNorms)),np.log(iterNorms),label="Iterations")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Residuals")
plt.show()