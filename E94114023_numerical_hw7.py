import numpy as np
A=np.array([
    [4,-1,0,-1,0,0],
    [-1,4,-1,0,-1,0],
    [0,-1,4,0,1,-1],
    [-1,0,0,4,-1,-1],
    [0,-1,0,-1,4,-1],
    [0,0,-1,0,-1,4]
])
U=np.array([
    [0,-1,0,-1,0,0],
    [0,0,-1,0,-1,0],
    [0,0,0,0,1,-1],
    [0,0,0,0,-1,-1],
    [0,0,0,0,0,-1],
    [0,0,0,0,0,0]
])
L=np.array([
    [0,0,0,0,0,0],
    [-1,0,0,0,0,0],
    [0,-1,0,0,0,0],
    [-1,0,0,0,0,0],
    [0,-1,0,-1,0,0],
    [0,0,-1,0,-1,0]
])
D=np.array([
    [4,0,0,0,0,0],
    [0,4,0,0,0,0],
    [0,0,4,0,0,0],
    [0,0,0,4,0,0],
    [0,0,0,0,4,0],
    [0,0,0,0,0,4]
])
bt=np.array([[0,-1,9,4,8,6]])
B=bt.T
def jacobi(a,u,l,d,b):
    xt=np.array([[1,1,1,1,1,1]])
    X=xt.T
    Dv=np.linalg.inv(d)
    T=-1*(Dv@(u+l))
    C=Dv@b
    print('X0:')
    print(X)
    for i in range(1,50):
        Xnew=T@X+C
        print('X'+str(i)+':')
        print(Xnew)
        if np.linalg.norm(Xnew-X)<0.001:
           return Xnew
        X=Xnew
def gauss_seidel(a,u,l,d,b):
    xt=np.array([[1,1,1,1,1,1]])
    X=xt.T
    T=-1*(np.linalg.inv(d+l)@u)
    C=np.linalg.inv(d+l)@b
    print('X0:')
    print(X)
    for i in range(1,50):
        Xnew=T@X+C
        print('X'+str(i)+':')
        print(Xnew)
        if np.linalg.norm(Xnew-X)<0.001:
           return Xnew
        X=Xnew
def SOC(a,u,l,d,b,w):
    xt=np.array([[1,1,1,1,1,1]])
    X=xt.T
    T=np.linalg.inv(d+w*l)@((1-w)*d-w*u)
    C=w*np.linalg.inv(d+l)@b
    print('X0:')
    print(X)
    for i in range(1,50):
        Xnew=T@X+C
        print('X'+str(i)+':')
        print(Xnew)
        if np.linalg.norm(Xnew-X)<0.001:
           return Xnew
        X=Xnew
def TCG(a,b):
    xt=np.array([[1,1,1,1,1,1]])
    X=xt.T
    print('X0:')
    print(X)
    for i in range(1,50):
        V=b-a@X
        t=(V.T@V)/(V.T@a@V)
        Xnew=X+t*V
        print('X'+str(i)+':')
        print(Xnew)
        if np.linalg.norm(Xnew-X)<0.001:
           return Xnew
        X=Xnew
print('A=')
print(A)
print('B=')
print(B)
print('(a)Jacobi method:')
jacobi(A, U, L, D, B)
print('(b)Gauss_seidel method:')
gauss_seidel(A, U, L, D, B)
print('(c)SOC method:')
SOC(A, U, L, D, B,0.5)
print('(d)The conjugate gradient method:')
TCG(A, B)







    
