from __future__ import print_function
import numpy as np

res = np.zeros((1,1), dtype='float64')

for i in range(0, 53):
    res[0][0] = res[0][0] + pow(2, -i)

print(res[0][0])
