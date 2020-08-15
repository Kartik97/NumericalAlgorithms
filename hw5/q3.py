import numpy as np
from scipy import linalg

def createMatrix(t0,t1,y0,y1,m0,m1):
    A=np.zeros((4,4))
    A[0,0]=A[1,0]=A[2,1]=A[3,1]=1
    A[1,1]=A[3,2]=t1-t0
    A[2,2]=t0-t1
    A[3,3]=(t1-t0)**2

    b = np.array([y0,y1,m0,m1])
    return A,b

def search(t,tnew):
    # Using Linear Search
    t=t.T[0]
    n=t.shape[0]
    for i in range(0,t.shape[0]-1):
        if(t[i]<=tnew and t[i+1]>tnew):
            return i,t[i],t[i+1]
    return n-2,t[n-2],t[n-1]

def piecewiseCubic(t,y,m,tnew):
#     # Using Newton's Basis
    t=np.reshape(t,(-1,1))
    y=np.reshape(y,(-1,1))
    m=np.reshape(m,(-1,1))
    n=t.shape[0]
    coeffs=[]
    for i in range(0,n-1):
        A,b = createMatrix(t[i],t[i+1],y[i],y[i+1],m[i],m[i+1])
        a = linalg.solve_triangular(A,b,lower=True)
        coeffs.append(a)
    i,t0,t1=search(t,tnew)
    factor=np.array([[1.0],[tnew-t0],[(tnew-t0)*(tnew-t1)],[(tnew-t0)*(tnew-t0)*(tnew-t1)]])
    return (coeffs[i].T@factor)[0][0]

def denominator(t0,t1,t2):
    return (t0-t1)*(t0-t2)

def estimateDerivatives(t,y):
    # Assuming list length would be atleast 3
    t,y=list(t),list(y)
    n=len(t)
    derivatives=[]
    t0,t1,t2=t[0],t[1],t[2]
    term1=y[0]*(2*t0-t1-t2)/denominator(t0,t1,t2)
    term2=y[1]*(t0-t2)/denominator(t1,t0,t2)
    term3=y[2]*(t0-t1)/denominator(t2,t0,t1)
    derivatives.append(term1+term2+term3)
    for i in range(1,n-1):
        term1=y[i-1]*(t[i]-t[i+1])/denominator(t[i-1],t[i],t[i+1])
        term2=y[i]*(2*t[i]-t[i-1]-t[i+1])/denominator(t[i],t[i-1],t[i+1])
        term3=y[i+1]*(t[i]-t[i-1])/denominator(t[i+1],t[i-1],t[i])
        derivatives.append(term1+term2+term3)
    term1=y[n-3]*(t[n-1]-t[n-2])/denominator(t[n-3],t[n-2],t[n-1])
    term2=y[n-2]*(t[n-1]-t[n-3])/denominator(t[n-2],t[n-3],t[n-1])
    term3=y[n-1]*(2*t[n-1]-t[n-2]-t[n-3])/denominator(t[n-1],t[n-2],t[n-3])
    derivatives.append(term1+term2+term3)
    derivatives=np.array(derivatives)
    derivatives=np.reshape(derivatives,(-1,1))
    return derivatives