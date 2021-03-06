import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

x = np.array([10.2,9.5,8.7,7.7,6.7,5.6,4.4,3.0,1.6,0.1])
y = np.array([3.9,3.2,2.7,2.2,1.8,1.5,1.3,1.2,1.3,1.5])

def fcontour(f,xrange,yrange,theta,**kwargs):
    xs = np.linspace(xrange[0], xrange[1])
    ys = np.linspace(yrange[0], yrange[1])
    fs = [[f(x,y,theta) for x in xs] for y in ys]
    plt.contour(xs, ys, fs, [0], **kwargs)
    plt.axis('scaled')

def testF(x,y,theta):
    return (1+theta[0][0])*x**2+(1-theta[0][0])*y**2+2*theta[1][0]*x*y+theta[2][0]*x+theta[3][0]*y+theta[4][0]

def createAb(x,y):
    return np.array([x**2-y**2,2*x*y,x,y,np.ones(x.shape)]).T,np.array([-x**2-y**2]).T

def back_subs(A,b):
    res = np.zeros(b.shape)
    for i in range(A.shape[0]-1,-1,-1):
        res[i][0] = b[i][0]/A[i][i]
        for j in range(0,i):
            b[j][0] = b[j][0]-res[i][0]*A[j][i]
    return res

def findCond(A,b,theta):
    k = np.linalg.cond(A,p=2)
    eta = np.linalg.norm(A,ord=2)*np.linalg.norm(theta,ord=2)/np.linalg.norm(np.dot(A,theta),ord=2)
    cos = np.linalg.norm(np.dot(A,theta),ord=2)/np.linalg.norm(b,ord=2)

    KbTheta = k/(eta*cos)

    tan = np.linalg.norm(np.dot(A,theta)-b)/np.linalg.norm(np.dot(A,theta))
    KATheta = k+(tan*k**2)/(eta)

    v = np.append(x,y)
    kv = KbTheta*2*np.linalg.norm(x+y,ord=2)*np.linalg.norm(v,ord=2)/np.linalg.norm(b,ord=2)

def ellipse_fit(x,y):
    A,b = createAb(x,y)
    q,r = np.linalg.qr(A)
    prod = np.dot(q.conj().T,b)
    theta = back_subs(r,prod)
    return theta

    # print(theta)
    # xr = [-5,12]
    # yr = [0,10]
    # fcontour(testF,xr,yr,theta)
    # plt.scatter(x,y)
    # plt.show()