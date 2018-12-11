import tensorflow as tf

a = tf.Variable(0)

c = tf.assign_add(a,1)
d = tf.summary.scalar('d',a)  #收集标量
serged = tf.summary.merge_all()   #自动收集

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs',sess.graph) #将图片存入到logs中
    sess.run(init)

    for i in range(100):
        print(sess.run(c))
        summery = sess.run(serged)
        writer.add_summary(summery,i)  #写入文件