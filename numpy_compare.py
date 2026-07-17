import numpy as np
import time

n = 10_000_000

x = np.random.rand(n)

# Python loop
start=time.time()
y=np.zeros(n)

for i in range(n):
    y[i]=x[i]*2

print(time.time()-start)


# NumPy
start=time.time()

y=x*2

print(time.time()-start)