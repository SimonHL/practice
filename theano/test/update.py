# -*- coding: utf-8 -*-
import numpy
import theano


def _step(x, a):
    return a*x 


dtype=theano.config.floatX
x = theano.tensor.vector()

a0 = numpy.array([1])
b0 = numpy.array([2])
a = theano.shared(a0.astype(dtype), name='a')
b = theano.shared(b0.astype(dtype), name='b')

y, updates = theano.scan(_step, 
                         sequences = x, 
                         non_sequences = a)

print updates

y_f = theano.function([x], outputs=y, updates=[(a,a+1), (b,b+2)])

x_data = numpy.array([1,2,3])



print y_f(x_data)
print a.get_value()
print b.get_value()

print y_f(x_data)
print a.get_value()
print b.get_value()


