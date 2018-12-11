import tensorflow as tf

#增加维度

a =  tf.constant([[1,2],[3,4]])
b = tf.expand_dims(a,axis=1) #增加维度
c = tf.tile(b,[1,4,1]) #张量扩张 tf.tile张量赋值  b的shape(2,1,2)乘以一个[1,4,1]变成c的shape（2,4,2）

with tf.Session() as sess:
    print(sess.run(a).shape)
    print(sess.run(b).shape)
    # print(sess.run(a))
    # print(sess.run(b))
    print(sess.run(c).shape)