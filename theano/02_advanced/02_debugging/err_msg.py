import numpy as np
import theano
import theano.tensor as T

#theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'

x = T.vector()
y = T.vector()
z = x + x
z = z + y
f = theano.function([x, y], z)#,mode='FAST_COMPILE')
f(np.ones((2,)), np.ones((3,)))