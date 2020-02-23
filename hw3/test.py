import numpy as np

A = np.array([[2,1,1],[1,2,1],[1,1,2]])
v = np.array([[1],[1],[1]])

MJ = np.array([[2,0,0],[0,2,0],[0,0,2]])
NJ = np.array([[0,1,0],[1,0,1],[0,1,0]])

MG = np.array([[2,0,0],[-1,2,0],[0,-1,2]])
NG = np.array([[0,-1,0],[0,0,-1],[0,0,0]])

x=np.linalg.norm(np.linalg.inv(MG)@NG,ord=1)

print(x)
