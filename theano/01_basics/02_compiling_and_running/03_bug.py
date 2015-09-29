# Something weird happens when you run this code.
# Find something that is not quite right.
# Figure out which compilation modes make the problem more obvious.
# Explain why what is happening is happening.
import numpy as np
from theano import function
from theano import tensor as T
x = T.vector()
y = T.vector()
z = T.ones_like(y)
a = x + z
f = function([x, y], a, mode='FAST_COMPILE')
output = f(np.ones((2,), dtype=x.dtype), np.zeros((3,), dtype=y.dtype))
print output
