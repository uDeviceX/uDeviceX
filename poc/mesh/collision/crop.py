import numpy as np
import matplotlib.pyplot as plt


# crop coordinates to get in the unit triangle


def crop(xin):
    u = xin[0]
    v = xin[1]
    
    # gamma region
    if (v > u - 1) and (v < u + 1) and (v > 1 - u):
        a = 0.5 * (u + v - 1)
        return [u-a, v-a]
    u = max([min([1.0, u]), 0.0])
    v = max([min([v, 1-u]), 0.0])
    return [u, v]

n = 100

input = 4 * (np.random.rand(n, 2) - 0.5) 
output = [crop(xin) for xin in input]

plt.figure()

plt.plot([0, 1, 0, 0], [0, 0, 1, 0], '-k')

for in_, out_ in zip(input, output):
    plt.plot(in_[0], in_[1], '+r')
    plt.plot(out_[0], out_[1], '+b')
    plt.plot([in_[0], out_[0]], [in_[1], out_[1]], '--b')
    
plt.show()
