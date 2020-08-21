import numpy as np

def calculateGradient(X,var,links,k,m,n,p):
    # X = (m*k), links=[[i,j]....[]]
    grad = np.zeros((n,k))
    for i in links:
        xi=var[i[0]-m-1] if i[0]>m else X[i[0]-1]
        xj=var[i[1]-m-1] if i[1]>m else X[i[1]-1]
        if(i[0]>m):
            grad[i[0]-m-1]+= p*(np.linalg.norm(xi-xj,ord=2)**(p-2))*(xi-xj)
        if(i[1]>m):
            grad[i[1]-m-1]+= -p*(np.linalg.norm(xi-xj,ord=2)**(p-2))*(xi-xj)
    # print(grad)
    return grad

def f(X,var,links):
    cost=0
    for i in links:
        xi=var[i[0]-m-1] if i[0]>m else X[i[0]-1]
        xj=var[i[1]-m-1] if i[1]>m else X[i[1]-1]
        cost+=np.linalg.norm(xi-xj,ord=2)**p
    return cost

def lineSearch(X,var,links,deltax,k,m,n,p):
    t=1
    alpha,beta=0.4,0.8
    grad=calculateGradient(X,var,links,k,m,n,p).flatten()
    while f(X,var+t*deltax,links)>f(X,var,links)+alpha*t*grad@deltax.flatten().T:
        t=beta*t
    return t

def gradientDescent(X,links,k,m,n,p,eplison=1e-5):
    k=X.shape[1]
    # X=X.flatten()
    gradnorm = 1
    var = np.zeros((n,k)) # Initialising all n points at zero
    while gradnorm>eplison:
        d=-calculateGradient(X,var,links,k,m,n,p)
        t=lineSearch(X,var,links,d,k,m,n,p)
        var=var+t*d
        gradnorm=np.linalg.norm(calculateGradient(X,var,links,k,m,n,p).flatten(),ord=2)
        print(gradnorm)
    return var

X=np.array([[1,1],[2,1],[2,2],[1,2]])
links=np.array([[5,3],[5,6],[5,2],[5,1],[6,4],[6,1],[6,2]])
# links=np.array([[5,2]])
k,m,n=2,4,2
p=2

print(gradientDescent(X,links,k,m,n,p))