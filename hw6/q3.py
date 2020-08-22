import numpy as np
import random

def calculateGradient(X,var,links,k,m,n,p):
    # X = (m*k), links=[[i,j]....[]]
    grad = np.zeros((n,k))
    var=var.reshape((n,k))
    for i in links:
        xi=var[i[0]-m-1] if i[0]>m else X[i[0]-1]
        xj=var[i[1]-m-1] if i[1]>m else X[i[1]-1]
        if(i[0]>m):
            grad[i[0]-m-1]+= p*(np.linalg.norm(xi-xj,ord=2)**(p-2))*(xi-xj)
        if(i[1]>m):
            grad[i[1]-m-1]+= -p*(np.linalg.norm(xi-xj,ord=2)**(p-2))*(xi-xj)
    return grad

def f(X,var,links,p):
    cost=0
    for i in links:
        xi=var[i[0]-m-1] if i[0]>m else X[i[0]-1]
        xj=var[i[1]-m-1] if i[1]>m else X[i[1]-1]
        # print(xi,xj,xi-xj,np.linalg.norm(xi-xj,ord=2)**p)
        cost+=np.linalg.norm(xi-xj,ord=2)**p
    return cost

def lineSearch(X,var,links,deltax,k,m,n,p):
    t=1
    alpha,beta=0.4,0.8
    grad=calculateGradient(X,var,links,k,m,n,p).flatten()
    while f(X,var+t*deltax,links,p)>f(X,var,links,p)+alpha*t*grad@deltax.flatten().T:
        t=beta*t
    return t 

def gradientDescent(X,links,k,m,n,p,eplison=1e-3,maxiter=1e2):
    X = X.reshape((m,k))
    # X=X.flatten()
    gradnorm = 1
    # var = np.zeros((n,k)) # Initialising all n points at zero
    var = np.random.uniform(1.0,2.0,(n,k)) # Random Initialisation of points
    i=1
    while gradnorm>eplison and i<maxiter:
        deltax=-calculateGradient(X,var,links,k,m,n,p)
        t=lineSearch(X,var,links,deltax,k,m,n,p)
        var=var+t*deltax
        gradnorm=np.linalg.norm(calculateGradient(X,var,links,k,m,n,p).flatten(),ord=2)
        # print(gradnorm)
        i+=1
    print(i)
    return var

def calculateHessian(X,var,links,k,m,n,p):
    # Using Centered Difference
    H = np.zeros((n*k,n*k))
    h = 1e-5
    row=0
    for i in range(n):
        for j in range(k):
            var[i,j]+=h
            grad1=calculateGradient(X,var,links,k,m,n,p).reshape((1,-1))
            var[i,j]-=2*h
            grad2=calculateGradient(X,var,links,k,m,n,p).reshape((1,-1))
            var[i,j]+=h
            H[row]=((grad1-grad2)/(2*h))
            row+=1
    H=(H+H.T)/2
    # print(H)
    # H=np.zeros((n,n))
    # for i in links:
    #     xi=var[i[0]-m-1] if i[0]>m else X[i[0]-1]
    #     xj=var[i[1]-m-1] if i[1]>m else X[i[1]-1]
    #     diff=(xi-xj).reshape((-1,1))
    #     if(i[0]>m):
    #         H[i[0]-m-1,i[0]-m-1]+=p*np.linalg.norm(xi-xj,ord=2)
    #     if(i[1]>m):
    #         H[i[1]-m-1,i[1]-m-1]+=p*(p-1)*np.linalg.norm(xi-xj,ord=2)
    #     if(i[0]>m and i[1]>m):
    #         H[i[0]-m-1,i[1]-m-1]+=-p*(p-1)*np.linalg.norm(xi-xj,ord=2)
    #         H[i[1]-m-1,i[0]-m-1]+=-p*(p-1)*np.linalg.norm(xi-xj,ord=2)
    # print(H)
    return H

def newton(X,links,k,m,n,p,epsilon=1e-3,maxiter=1e3):
    var = np.random.uniform(1.0,2.0,(n,k)) # Random Initialisation of points
    # var=np.array([[0.0949275,0.34440541],[0.70923956,0.43162053],[0.80891097,0.94935611]])
    decrement=1
    i=1
    # print(var)
    while decrement>epsilon and i<maxiter:
        H = calculateHessian(X,var,links,k,m,n,p)
        # print(H)
        grad = calculateGradient(X,var,links,k,m,n,p).reshape((-1,1))
        deltax = -np.linalg.inv(H)@grad
        deltax=deltax.reshape((n,k))
        t=lineSearch(X,var,links,deltax,k,m,n,p)
        var=var+t*deltax
        decrement=grad.T@H@grad
        i+=1
        # print(H)
        # break
    print(i)
    return var

# X=np.array([[1,1],[1,2],[2,2],[2,1]])
X = np.random.uniform(1,10,(8,2))
m,n,k,p=8,6,2,1.2
links=np.array([[14,1],[14,8],[14,7],[14,9],[14,13],[14,12],[13,1],[13,12],[13,2],[13,3],[13,11],[12,1],[12,11],[12,9],[11,3],[11,4],[11,10],[10,4],[10,14],[10,9],[10,6],[10,5],[9,6],[9,5],[9,7]])
result=gradientDescent(X,links,k,m,n,p)
print(result,f(X,result,links,p))
# result=newton(X,links,k,m,n,p)
# print(result,f(X,result,links,p))

# print(links)
# links=np.array([[7,2],[7,4],[7,3]])
# k,m,n,p=2,6,6,1.2
# all_links,fixed_points=[],[]
# for i in range(1,m+n+1):
#     fixed_points.append(i)
#     for j in range(i+1,m+n+1):
#         all_links.append([i,j])
# links=[]
# for i in range(1,n+1):
#     links.append([i+m,random.sample(fixed_points,1)[0]])
# links=np.array(random.sample(all_links,12))
# links=np.array(links)
# print(links)

# X = np.array([[0.23017621, 0.55718181],
#        [0.71442189, 0.16026443],
#        [0.2634742 , 0.69211105],
#        [0.89201334, 0.13123567],
#        [0.20364701, 0.73184623],
#        [0.71984995, 0.01871416]])
# m,n,k,p=6,5,2,1.2
# links=[]
# for i in range(1,m+1):
#     for j in range(i+1,m+n+1):
#         links.append([i,j])
# print(links)
# print(gradientDescent(X,links,k,m,n,p))
# result=newton(X,links,k,m,n,p)
# print(result)

# X=np.array([[0.23017621, 0.55718181],
#        [0.71442189, 0.16026443],
#        [0.2634742 , 0.69211105],
#        [0.89201334, 0.13123567],
#        [0.20364701, 0.73184623],
#        [0.71984995, 0.01871416]])
# m,n,p,k=6,3,1.2,2
# links=[]
# for i in range(1,m+1):
#     for j in range(i+1,m+n+1):
#         links.append([i,j])
# links=np.array(links)
# res=newton(X,links,k,m,n,p)
# print(res)
# print(f(X,res,links,p))
# res=gradientDescent(X,links,k,m,n,p)
# print(res)