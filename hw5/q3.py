import numpy as np
from scipy import linalg

def createMatrix(t0,t1,y0,y1,m0,m1):
    A=np.zeros((4,4))
    A[0,0]=A[1,0]=A[2,1]=A[3,1]=1
    A[0,1]=t0
    A[0,2]=t0**2
    A[0,3]=t0**3
    A[1,1]=t1
    A[1,2]=t1**2
    A[1,3]=t1**3
    A[2,2]=2*t0
    A[2,3]=3*t0*t0
    A[3,2]=2*t0
    A[3,3]=3*t1*t1

    b = np.array([y0,y1,m0,m1])
    return A,b

def search(t,tnew):
    # Using Linear Search
    t=t.T[0]
    for i in range(0,t.shape[0]-1):
        if(t[i]<=tnew and t[i+1]>tnew):
            return i
    return t.shape[0]-2
    # Using Binary Search
    # l,r=0,t.shape[0]-1
    # while l<=r:
    #     m=l+(r-l)//2
    #     if(m==0 and t[m+1]>tnew):
    #         return 0
    #     elif(m==t.shape[0]-1 and t[m-1]<=tnew):
    #         return t.shape[0]-2
    #     elif (t[m]<=tnew and t[m+1]>tnew):
    #         return m
    #     elif (t[m]>tnew):
    #         r=m-1
    #     else:
    #         l=m+1
    # return l

def piecewiseCubic(t,y,m,tnew):
    # Using Monomial Basis
    t=np.reshape(t,(-1,1))
    y=np.reshape(y,(-1,1))
    m=np.reshape(m,(-1,1))
    n=t.shape[0]
    coeffs=[]
    for i in range(0,n-1):
        A,b = createMatrix(t[i],t[i+1],y[i],y[i+1],m[i],m[i+1])
        a = linalg.solve(A,b)
        coeffs.append(a)
    i=search(t,tnew)
    factor=np.array([[1.0],[tnew],[tnew**2],[tnew**3]])
    return coeffs[i].T@factor



print(piecewiseCubic([1.3,1.6,1.9],[0.620086,0.4554022,0.2818186],[-0.5220232,-0.5648959,-0.5811571],1.5))  #0.5118277-0.512165
print(piecewiseCubic([1.3,1.6,1.9],[0.620086,0.4554022,0.2818186],[-0.5220232,-0.5648959,-0.5811571],1.5))  #0.5118277