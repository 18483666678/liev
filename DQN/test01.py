import tensorflow as tf

# a = tf.Variable([1,2,3],name='a')
# b = tf.Variable([1,2,3],name='a')
# c = tf.Variable([1,2,3],name='a')
# print(a)
# print(b)
# print(c)
a = tf.get_variable('a',shape=[2])
b = tf.get_variable('a',shape=[2])
print(a)
print(b)
#使用tf.Variable如果检测到命名冲突，系统自动处理
#tf.get_variable检测到命名冲突，系统不会处理

