import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

def f(X,var,links,m,p):
    cost=0
    for i in links:
        xi=var[i[0]-m-1] if i[0]>m else X[i[0]-1]
        xj=var[i[1]-m-1] if i[1]>m else X[i[1]-1]
        cost+=np.linalg.norm(xi-xj,ord=2)**p
    return cost

def lineSearch(X,var,links,deltax,k,m,n,p):
    t=1
    alpha,beta=0.3,0.8
    grad=calculateGradient(X,var,links,k,m,n,p).reshape((1,-1))
    while f(X,var+t*deltax,links,m,p)>f(X,var,links,m,p)+alpha*t*grad@deltax.reshape((-1,1)):
        t=beta*t
    return t 

def gradientDescent(X,links,k,m,n,p,eplison=1e-4,maxiter=1e3):
    X = X.reshape((m,k))
    gradnorm,i=1,1
    var = np.random.uniform(4.0,6.0,(n,k)) # Random Initialisation of points
    normList=[]
    while gradnorm>eplison and i<maxiter:
        deltax=-calculateGradient(X,var,links,k,m,n,p)
        t=lineSearch(X,var,links,deltax,k,m,n,p)
        var=var+t*deltax
        gradnorm=np.linalg.norm(calculateGradient(X,var,links,k,m,n,p).flatten(),ord=2)
        normList.append(gradnorm)
        i+=1
    return var,normList

def calculateHessian(X,var,links,k,m,n,p):
    # Using Centered Difference
    H = np.zeros((n*k,n*k))
    h = 1e-4
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
    return H

def newton(X,links,k,m,n,p,epsilon=1e-4,maxiter=1e3):
    var = np.random.uniform(4.0,6.0,(n,k)) # Random Initialisation of points
    decrement,i=1,1
    normList=[]
    while decrement>epsilon and i<maxiter:
        H = calculateHessian(X,var,links,k,m,n,p)  # Calculate Hessian
        grad = calculateGradient(X,var,links,k,m,n,p).reshape((-1,1))
        deltax = -np.linalg.inv(H)@grad
        deltax=deltax.reshape((n,k))
        t=lineSearch(X,var,links,deltax,k,m,n,p)
        var=var+t*deltax
        decrement=grad.T@H@grad
        gradnorm=np.linalg.norm(calculateGradient(X,var,links,k,m,n,p).flatten(),ord=2)
        normList.append(gradnorm)
        i+=1
    return var,normList

def testCase():
    X = np.array([[1.0,1.0],[2.0,0.7],[2.7,0.5],[3,1.5],[2.5,3],[2.5,3],[1,3],[1,2]])
    links=np.array([[14,1],[14,8],[14,7],[14,9],[14,13],[14,12],[13,1],[13,12],[13,2],[13,3],[13,11],[12,1],[12,11],[12,9],[11,3],[11,4],[11,10],[10,4],[10,14],[10,9],[10,6],[9,6],[9,7]])
    m,n,k,p=8,6,2,1.2
    result,normList=gradientDescent(X,links,k,m,n,p)
    print("Gradient Descent : ",result,f(X,result,links,m,p),len(normList))
    result,normList=newton(X,links,k,m,n,p)
    print("Newton's Method : ",result,f(X,result,links,m,p),len(normList))


testCase()
# X=np.array([[1,1],[1,2],[2,2],[2,1]])
# X = np.random.uniform(3,8,(8,2))
# X = np.array([[7.15237843,4.70701317],
#  [5.15542796,4.50275655],
#  [4.07949618, 3.21611181],
#  [7.51799915, 3.79230366],
#  [5.53265132, 7.0027515 ],
#  [4.2840709 , 3.78633124],
#  [6.7790279 , 5.03089901],
#  [7.5224723 , 3.47052628]])
# X = np.array([[1.0,1.0],[2.7,0.5],[3,1.5],[2.5,3],[2.5,3],[1,3],[1,2]])   #CLASSIC CONVERGENCE
# links=np.array([[13,1],[13,8],[13,7],[13,9],[13,12],[13,2],[13,3],[13,11],[12,1],[12,11],[12,9],[11,3],[11,4],[11,10],[10,4],[10,9],[10,6],[9,6],[9,7]])
# X = np.array([[0.7,2.3],[2,3],[3,3.2],[3.7,3.3],[4,2],[3.3,1.1],[2,1],[1,1.5]])
# links = np.array([[1,11],[1,12],[1,14],[2,12],[2,13],[2,11],[3,13],[3,14],[8,11],[7,11],[7,10],[4,14],[5,14],[5,9],[6,9],[6,14],[6,10],[9,10],[13,11],[12,14],[10,13],[10,11],[10,14],[4,5],[7,8]])
# m,n,k,p=8,6,2,1.2
# result,normList=gradientDescent(X,links,k,m,n,p)
# print(result,f(X,result,links,p),len(normList))
# result,normList=newton(X,links,k,m,n,p)
# print(result,f(X,result,links,p),len(normList))

# fig, ax = plt.subplots()
# G = nx.Graph()
# color_map=[]
# for i in range(m+n):
#     if(i+1<=m):
#         color_map.append('red')
#     else:
#         color_map.append('blue')
#     G.add_node(i+1)

# for i in links:
#     G.add_edge(i[0],i[1])

# pos={}
# for i,pt in enumerate(X):
#     pos[i+1]=list(pt)
# for i,pt in enumerate(result):
#     pos[i+m+1]=list(pt)

# options = {'node_size' : 100, 'node_color' : 'k'}
# nx.draw_networkx_nodes(G, pos, node_size=20, node_color=color_map,ax=ax)
# nx.draw_networkx_edges(G, pos, alpha=0.2)
# # plt.axis('off')
# plt.title('Graph')
# ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
# plt.show()
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