import numpy as np

def calculateGradient(X,var,links,m,n,p):
    # X = (m*k), links=[[i,j]....[]]
    k=X.shape[1]
    grad = np.zeros((n,k))
    for i in links:
        xi=var[i[0]-1] if i[0]>m else X[i[0]-1]
        xj=var[i[0]-1] if i[1]>m else X[i[1]-1]
        print(xi,xj)


# def lineSearch(var,grad):

def gradientDescent(X,links,m,n,p,eplison=1e-4):
    gradnorm = 1
    k=X.shape[1]
    var = np.zeros((n,k)) # Initialising all n points at zero
    # while gradnorm>eplison:
    grad=calculateGradient(X,var,links,m,n,p)
        # t=lineSearch(grad,X)
        # var=var+t*grad
        # gradnorm=norm(grad)
    return var

X=np.array([[1,1],[1,2],[2,1],[2,2]])
links=np.array([[5,3],[5,6],[5,2],[5,1],[6,4],[6,1],[6,2]])
m,n=4,2
p=2
gradientDescent(X,links,m,n,p)