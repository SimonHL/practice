# -*- coding: utf-8 -*-
"""

"""

import numpy
import theano

def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]

x = theano.tensor.dvector('x')

x_printed = theano.printing.Print('this is a very important value')(x)

f = theano.function([x], [x * 5], mode=theano.compile.MonitorMode(
                        pre_func=inspect_inputs,
                        post_func=inspect_outputs))
print f([3.3,4])
print f([1,5])
              
f_with_print = theano.function([x], x_printed * 5)
print f_with_print(range(10))

#this runs the graph without any printing
#assert numpy.all( f([1, 2, 3]) == [5, 10, 15])

#this runs the graph with the message, and value printed
#assert numpy.all( f_with_print([1, 2, 5]) == [5, 10, 25])

theano.printing.debugprint(f)
