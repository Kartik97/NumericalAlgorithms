import numpy as np

x = np.array([[1,0,0],[2,6,0],[3,4,5]])
v = np.array([[1],[1],[1]])
print(np.linalg.cholesky(x@x.T+v@v.T))

