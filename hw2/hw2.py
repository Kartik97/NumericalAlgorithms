import numpy as np

def createAW(n):
    A = np.identity(n)
    one = [-1]*(n-1)
    W = ((A*2)+np.diag(one,1)+np.diag(one,-1))
    return A,W

def weightedModifiedGramSchmidt(A,W):
    V = A
    n = A.shape[0]
    m = A.shape[1]
    g = max(m,n)
    l = min(m,n)
    R = np.zeros((m,m))
    Q = np.zeros((n,m))
    for i in range(0,l):
        # R[i][i] = np.linalg.norm(V[:,i:i+1],ord=2)
        temp = np.dot(V[:,i:i+1].T,W)
        R[i][i] = np.sqrt(np.dot(temp,V[:,i:i+1]))
        Q[:,i:i+1] = V[:,i:i+1]/R[i][i]
        for j in range(i+1,l):
            temp = np.dot(Q[:,i:i+1].T,W)
            R[i][j] = np.dot(temp,V[:,j:j+1])
            V[:,j:j+1] = V[:,j:j+1]-(R[i][j]*Q[:,i:i+1])
    return Q,R

if __name__ == '__main__':
    n=5
    A,W = createAW(n)
    # A = np.array([[1,2,3],[-1,1,1],[1,1,1]])
    # A = np.array([[0,0,1],[0,1,1],[1,1,0],[1,0,0]])

    Q,R = weightedModifiedGramSchmidt(A,W)
    print(Q)
    print(R)