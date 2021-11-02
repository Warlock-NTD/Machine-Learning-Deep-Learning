import numpy as np

a = np.arange(15).reshape(3, 5)
# print(a)
# print(a.shape)
# print(a.ndim)
# print(a.dtype)
# print(a.size)
# print(type(a))

c = np.array([[1, 2], [3, 4]], dtype=complex)
# print(c)
# print(np.zeros((3, 4)))
# print(np.ones((2, 3, 4), dtype=np.int16))
# print(np.arange(10, 30, 5))
# print(np.linspace(0, 2, 9))
# print(np.zeros_like(a))
# print(np.ones_like(a))
# np.set_printoptions(threhold= sys.maxsize) # print hole big array
# print(np.random.random((2, 3)))
# print(np.linspace(0, np.pi, 3))
a = np.random.random((2, 3))
# print(a.sum(), a.min(), a.max())
b = np.arange(12).reshape(3, 4)
# print(b.sum(axis=1))
# axis=0 : col, axis=1 : row
# print(b.cumsum(axis=1)) # prefix sum by row

a = np.arange(10)**3
a[1::3] = -1000 # [0:6:3] ... [::3] ... [::]
# print(a)
# print(a[::-1]) # reverse a
a = a[::-1]
# print(a)

def f(x,y):
    return 10*x + y

b = np.fromfunction(f, (5, 4), dtype=int)
# print(b[2, 3])
# print(b[0:5, 1])
# print(b[:, 1]) same above line
# print(b[1:3, :])

for el in b.flat:
    print(el)

a = np.floor(10*np.random.random((3, 4)))
# print(a.ravel()) # similar to flat
# print(a.reshape(6, 2))
a = a.reshape(3, -1)
# print(a)