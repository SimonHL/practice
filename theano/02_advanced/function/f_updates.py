import theano
import numpy as np
import theano.tensor as T 

# theano vector variable 
x = T.ivector('x')
# shared variable to check update
y = theano.shared(np.array([2,2]),name='y_shared')
# the update increment variable
inc = theano.shared(np.array([1,1],dtype=np.int32))
# the symbolic expression to calculate output
output = x + y
test = theano.function([],updates=[(y,y+inc)])

# 设置 x = 0 这样其实output 的值就是 output 计算时的Y值
# 如果 updates 先执行 这样 应该是 output == y
# 如果 output 先计算 这样 updates != output
# 借用这样的一组简单的例子说明 output 和 updates 计算时机

print 'Output value is : ',test(),test()

print 'The shared variable : ',y.get_value()
