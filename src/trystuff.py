import numpy as np
import scipy.sparse

a = np.array([1, 2, 3, 4, 5])


n = 5
x = (np.random.rand(n) * 2).astype(int).astype(float) # 50% sparse vector
x_csr = scipy.sparse.csr_matrix(x)

print x.shape
print x_csr.shape
print x_csr.dot(a)
