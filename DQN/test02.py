#共享空间 共享变量
import tensorflow as tf

with tf.variable_scope('abc'):
    a = tf.get_variable('a',shape=[2])
with tf.variable_scope('abc',reuse=True):  #为True时表示tf.get-variable得到的变量可以在别的地方重复使用
    b = tf.get_variable('a',shape=[2])
print(a)
print(b)

op = tf.assign(a,[1,2])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(op)
    print(sess.run(b))