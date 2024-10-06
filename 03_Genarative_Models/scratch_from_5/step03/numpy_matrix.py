import numpy as np

A = np.array([[1,2,3],[4,5,6]])
print(A)
print(A.T)

#行列式
A = np.array([[3,4],[5,6]])
d = np.linalg.det(A)
print(A)

#逆行列
A = np.array([[3,4],[5,6]])
B = np.linalg.inv(A)

print(B)
print('---')
print(A@B)

def multivariate_normal(x,mu,cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1/np.sqrt((2*np.pi)**D*det)
    y = z*np.exp((x-mu).T@inv@(x-mu)/-2.0)
    return y

x = np.array([[0],[0]])
mu = np.array([[1],[2]])
cov = np.array([[1,0],
                [0,1]])
y = multivariate_normal(x,mu,cov)
print(y)