import numpy as np

def createAW(n):
    A = np.identity(n)
    one = [-1]*(n-1)
    W = ((A*2)+np.diag(one,1)+np.diag(one,-1))
    return A,W

def weighted_qr(A,W):
    V = A.astype('float')
    n = A.shape[0]
    m = A.shape[1]
    g = max(m,n)
    l = min(m,n)
    R = np.zeros((m,m))
    Q = np.zeros((n,m))
    for i in range(0,l):
        temp = np.dot(V[:,i:i+1].conj().T,W)
        R[i][i] = np.sqrt(np.dot(temp,V[:,i:i+1]))
        Q[:,i:i+1] = V[:,i:i+1]/R[i][i]
        for j in range(i+1,l):
            temp = np.dot(Q[:,i:i+1].conj().T,W)
            R[i][j] = np.dot(temp,V[:,j:j+1])
            V[:,j:j+1] = V[:,j:j+1]-(R[i][j]*Q[:,i:i+1])
    return [Q,R]