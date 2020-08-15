import numpy as np
from scipy import linalg

def createMatrix(t0,t1,y0,y1,m0,m1):
    A=np.zeros((4,4))
    # A[0,0]=A[1,0]=A[2,1]=A[3,1]=1
    # A[0,1]=t0
    # A[0,2]=t0**2
    # A[0,3]=t0**3
    # A[1,1]=t1
    # A[1,2]=t1**2
    # A[1,3]=t1**3
    # A[2,2]=2*t0
    # A[2,3]=3*t0*t0
    # A[3,2]=2*t0 
    # A[3,3]=3*t1*t1
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

# def piecewiseCubic(t,y,m,tnew):
#     # Using Monomial Basis
#     t=np.reshape(t,(-1,1))
#     y=np.reshape(y,(-1,1))
#     m=np.reshape(m,(-1,1))
#     n=t.shape[0]
#     coeffs=[]
#     for i in range(0,n-1):
#         A,b = createMatrix(t[i],t[i+1],y[i],y[i+1],m[i],m[i+1])
#         a = linalg.solve(A,b)
#         coeffs.append(a)
#     i=search(t,tnew)
#     factor=np.array([[1.0],[tnew],[tnew**2],[tnew**3]])
#     return coeffs[i].T@factor

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
    return coeffs[i].T@factor

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
    return derivatives

# print(piecewiseCubic([1.3,1.6,1.9],[0.620086,0.4554022,0.2818186],[-0.5220232,-0.5648959,-0.5811571],1.9))  #0.5118277-0.512165
# print(estimateDerivatives([1.3,1.6,1.9],[0.620086,0.4554022,0.2818186]))

# hw5-agp
t=[-1.,-0.60685724,-0.50658655,-0.49578285,-0.28211784,-0.15838871,0.19500793,0.30415757,0.45340956,0.48473086,0.68048472,1.]
y=[-2.71828635,-1.83630269,-1.66703767,-1.65051344,-1.55952159,-4.49011781,0.83272048,0.73928382,0.63557943,0.61593433,0.50637405,0.36787945]
m=[2.71834964,1.85931608,1.77017546,1.77167932,4.06696031,9.04051714,-0.65758507,-0.71165818,-0.63339109,-0.61464934,-0.50632795,-0.36787925]
# for i in t:
#     print(piecewiseCubic(t,y,m,i))
print(piecewiseCubic(t,y,m,-0.7))
print(estimateDerivatives(t,y))    
