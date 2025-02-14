import numpy as np
import matplotlib.pyplot as plt
N=500
X = 2 * np.random.rand(N, 1)
y = 4 + 3 * X +6*X**2+ np.random.randn(N, 1)
xr=np.linspace(0,2,10)
yr= 4 + 3 * xr+6*xr**2
unos=np.ones((len(X), 1))
A = np.hstack((unos, X,X**2))  # montar la matriz
b=y
print(np.shape(b))
print(np.shape(A))

w=np.matmul(np.linalg.pinv(A),b)
print(w)

lr = 0.1
n_iteraciones = 1000
m = len(A)
w = np.random.randn(3,1)
loss=[]
for iteraciones in range(n_iteraciones):
    gradiente = 2/m * A.T.dot(A.dot( w) - y)
    w= w - lr* gradiente
    ye=w[0]+w[1]*X+w[2]*(X**2)
    errores=(ye-y)**2
    loss.append(np.sum(errores))
print(w)
plt.plot(loss)
plt.show()

yer=w[0]+w[1]*xr+w[2]*(xr**2)
plt.plot(xr,yer,color="green")

plt.plot(xr,yr)
plt.scatter(X,y,alpha=0.3)
plt.show()

