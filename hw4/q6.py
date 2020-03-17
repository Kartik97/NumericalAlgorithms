from scipy import linalg
import math
import numpy as np

def f(A0,A1,y1,x):
    return min(np.linalg.eigvals((A0+x*A1)))-y1

def findRoot1(A0,A1,y1,etol):
    A0eval = linalg.eigvals(A0)
    A1eval = linalg.eigvals(A1)
    a = (y1-min(A0eval))/min(A1eval)
    b = (y1-max(A0eval))/max(A1eval)
    count=0
    data = np.array([[count,a,b,f(A0,A1,y1,a),f(A0,A1,y1,b),abs(b-a)]])
    while(abs(b-a) > etol):
        mid = a+(b-a)/2
        if(np.sign(min(np.linalg.eigvals((A0+mid*A1)))-y1) == np.sign(min(np.linalg.eigvals((A0+b*A1)))-y1)):
            b = mid 
        else:
            a = mid 
        count+=1
        data = np.vstack([data,[count,a,b,f(A0,A1,y1,a),f(A0,A1,y1,b),abs(b-a)]])
    # print(data.real)
    if(abs(min(np.linalg.eigvals((A0+a*A1)))-y1) <  abs(min(np.linalg.eigvals((A0+b*A1)))-y1)):
        return a
    else:
        return b

def func(A0,A1,A2,x):
    tempVal,tempVec = linalg.eig(A0+A1*x[0]+A2*x[1])
    return np.array([[min(tempVal)],[max(tempVal)]])

def findRoot2(A0,A1,A2,y,x0,rtol,maxiter):
    count = 0
    y = np.reshape(np.array(y),(-1,1))
    x0 = np.reshape(np.array(x0),(-1,1))
    A1Val,A1Vec = linalg.eig(A1)
    A2Val,A2Vec = linalg.eig(A2)
    jacobian = np.array([[min(A1Val),min(A2Val)],[max(A1Val),max(A2Val)]])
    data = np.array([[count,x0.real,func(A0,A1,A2,x0).real,np.linalg.norm(func(A0,A1,A2,x0)-y,ord=2)]])
    while True:
        l,u = linalg.lu(jacobian,permute_l=True)
        b = func(A0,A1,A2,x0)-y
        w = linalg.solve_triangular(l,b,lower=True)
        x = linalg.solve_triangular(u,w,lower=False)
        # x = linalg.solve(jacobian,func(A0,A1,A2,x0)-y)
        x0 = x0-x
        count+=1
        data = np.vstack([data,[count,x0.real,func(A0,A1,A2,x0).real,np.linalg.norm(func(A0,A1,A2,x0)-y,ord=2)]])
        if(count>maxiter or np.linalg.norm(func(A0,A1,A2,x0)-y,ord=2)<rtol):
            break
    # print(data.real)
    return x0