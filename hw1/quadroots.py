import math
import matplotlib.pyplot as plt
import numpy as np

def b(h):
    return 100+1.11*h

def a(h):
    return (1/4)

def c(h):
    # return (100+1.11h)**2 - (100+1.1*h)**2
    return 0.01*h*(200+2.21*h)

# Given a value h-calculates the expected root values using the original formula
def calRoots(h):
    return ((-b(h)-math.sqrt(b(h)**2-4*a(h)*c(h)))/(2*a(h)),(-b(h)+math.sqrt(b(h)**2-4*a(h)*c(h)))/(2*a(h)))

# Returns the smaller root value of the taken function
def xStar(h):
    return -0.02*h

# Part (c)- Quardroots function
def quadroots(a,b,c):
    if(b>=0):
        return (((-b-math.sqrt(b**2-4*a*c))/(2*a)),(2*c/(-b-math.sqrt(b**2-4*a*c))))
    else:
        return ((((-b+math.sqrt(b**2-4*a*c))/(2*a))),(2*c/(-b+math.sqrt(b**2-4*a*c))))

if __name__ == '__main__':

    h=np.linspace(1e-2,1e-5,1e4)
    x=h
    y1=[]
    y2=[]
    for i in h:
        f,s = calRoots(i)
        F,S = quadroots(a(i),b(i),c(i))
        y1.append(abs(xStar(i)-s))
        y2.append(abs(xStar(i)-S))

    # Part (b)

    xlog = (np.log(x[:500]))
    ylog1 = (np.log(y1[:500]))
    ax = plt.gca()
    l1, = plt.plot(xlog, ylog1, linewidth=2, color='navy')
    l1.set_label('Original Formula')
    plt.xlabel(r'log(h)')
    plt.ylabel(r'log(y)')
    plt.legend()
    ax.grid(True)
    plt.title(r'Log-Log plot')
    plt.show()
    plt.clf()

    # Part (c)

    xlog = (np.log(x[:500]))
    ylog1 = (np.log(y1[:500]))
    ylog2 = (np.log(y2[:500]))
    ax = plt.gca()
    l1, = plt.plot(xlog, ylog1, linewidth=2, color='navy')
    l1.set_label('Original Formula')
    l2, = plt.plot(xlog, ylog2, linewidth=2, color='green')
    l2.set_label('Modified Formula')
    plt.legend()
    plt.xlabel(r'log(h)')
    plt.ylabel(r'log(y)')
    ax.grid(True)
    plt.title(r'Log-Log plot')
    plt.show()
    plt.clf()
    plt.close()