# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T
import theano

a = np.ones([3,4])

b = np.zeros([4,3])

x = T.scalar()

y = T.scalar()

c = x + y

f_test = theano.function([x,y],c)
print  f_test(3,4)

print a
print b

mystr = 'a b cde'
print mystr.upper()
print mystr.lower()
print mystr.capitalize()
print mystr.title()
