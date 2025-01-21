import numpy as np
x = np.array([1,2,3])

print(x.__class__)
print(x.shape)
print(x.ndim)

w = np.array([[1,2,3],
             [4,5,6]])
print(w.ndim)
print(w.shape)

w = np.array([[1,2,3],[4,5,6]])
x = np.array([[0,1,2],[3,4,5]])

print(w+x)
print('---')
print(w*x)

#ベクトルの内積
a = np.array([1,2,3])
b = np.array([4,5,6])
y = np.dot(a,b)
print(y)

#行列の積
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
Y = np.dot(A,B)
print(Y)
