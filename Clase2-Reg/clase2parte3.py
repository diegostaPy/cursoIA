import numpy as np
from sklearn.linear_model import SGDRegressor
N=500
X = 2 * np.random.rand(N, 1)
y = 4 + 3 * X + np.random.randn(N, 1)

sgd_reg = SGDRegressor(max_iter=1000, eta0=0.1  , random_state=42)
sgd_reg.fit(X, y.ravel())

print(sgd_reg.coef_)

print(sgd_reg.intercept_)
print(sgd_reg.score(X,y))