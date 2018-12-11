import tensorflow as tf

with tf.variable_scope('xyz') as scope:
    a = tf.get_variable('a',shape=[2])
    b = tf.get_variable('b',shape=[2])
c = tf.get_variable('c',shape=[2],collections=['abc'])
print(tf.get_collection('abc'))
print(tf.get_collection(tf.GraphKeys.VARIABLES,scope='xyz'))